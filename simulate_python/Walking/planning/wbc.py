import numpy as np
import mujoco
import proxsuite


class WBCController:
    def __init__(self, mj_model, mj_data, ssp_duration=0.4, dsp_duration=0.1, dt=0.005):
        self.model = mj_model
        self.data = mj_data
        self.dt = dt
        
        # System dimensions
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.state = 2  
        
        # Walking state tracking
        self.step_counter = 0
        self.side = 0
        self.phase_time = 0.0
        self.dsp_duration = dsp_duration
        self.ssp_duration = ssp_duration

        # Body IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.left_ankle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_ankle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

        # Contact offsets relative to ankle links
        self.contact_offsets = [
            np.array([-0.05, -0.025, -0.03]),
            np.array([-0.05,  0.025, -0.03]),
            np.array([ 0.12,  0.030, -0.03]),
            np.array([ 0.12, -0.030, -0.03])
        ]
        self.n_contacts = 2 * len(self.contact_offsets)
        
        # QP Dimensions
        self.n_vars = self.nv + self.nv + 3 * self.n_contacts
        self.n_eq = self.nv
        self.n_in = 5 * self.n_contacts
        
        self.A = np.zeros((self.n_eq, self.n_vars))
        self.b = np.zeros(self.n_eq)
        self.C = np.zeros((self.n_in, self.n_vars))
        self.dmin = np.zeros(self.n_in)
        self.dmax = np.zeros(self.n_in)

        self.M = np.zeros((self.nv, self.nv))
        self._setup_friction_cone()

        # Task setup: Contacts (24) + CoM (3) + Pelvis Rot (3) + Torso Rot (3) + Posture (nu)
        self.n_task_rows = 3 * self.n_contacts + 3 + 3 + 3 + self.nu
        self.W = np.zeros((self.n_task_rows, self.n_vars))
        self.t = np.zeros(self.n_task_rows)
        
        # Regularization matrices (Tikhonov / Numerical Damping)
        self.reg = np.zeros((self.n_vars, self.n_vars))
        np.fill_diagonal(self.reg[:self.nv, :self.nv], 1e-2)
        np.fill_diagonal(self.reg[self.nv:2*self.nv, self.nv:2*self.nv], 1e-4)
        np.fill_diagonal(self.reg[2*self.nv:, 2*self.nv:], 1e-5)

# Normalized Task Weights (Prevents ill-conditioned QP matrix)
        self.w_contact = 10000.0   # Highest priority (rigid ground contact)
        self.w_com = 1000.0        # Reduced so it doesn't fight the contact constraint
        self.w_pelvis = 500.0
        self.w_torso = 200.0
        self.w_posture = 1.0       # Minimal interference

        # Stable PD Gains (Prevents torque explosions)
        self.Kp_contact = 100.0
        self.Kd_contact = 20.0
        
        # Anisotropic CoM gains: Stiff enough to stand, soft enough to simulate
        self.Kp_com = np.array([40.0, 40.0, 300.0]) 
        self.Kd_com = np.array([15.0, 15.0, 50.0])
        
        self.Kp_pelvis = 100.0
        self.Kd_pelvis = 20.0
        self.Kp_torso = 100.0
        self.Kd_torso = 20.0
        self.Kp_posture = 40.0
        self.Kd_posture = 10.0

        self.qp = proxsuite.proxqp.dense.QP(self.n_vars, self.n_eq, self.n_in)
        self.qp_init = False
        self.initialized = False

    def _setup_friction_cone(self):
        mu = 0.6
        n = np.array([0, 0, 1])
        t1 = np.array([1, 0, 0])
        t2 = np.array([0, 1, 0])
        
        for i in range(self.n_contacts):
            col = 2 * self.nv + 3 * i
            row = 5 * i
            self.C[row, col:col+3] = n
            self.dmin[row] = 1e-4
            self.dmax[row] = 1e20
            
            self.C[row+1, col:col+3] = t1 - mu * n
            self.dmin[row+1] = -1e20
            self.dmax[row+1] = 0
            
            self.C[row+2, col:col+3] = t1 + mu * n
            self.dmin[row+2] = 0
            self.dmax[row+2] = 1e20
            
            self.C[row+3, col:col+3] = t2 - mu * n
            self.dmin[row+3] = -1e20
            self.dmax[row+3] = 0
            
            self.C[row+4, col:col+3] = t2 + mu * n
            self.dmin[row+4] = 0
            self.dmax[row+4] = 1e20

    def swing(self, phase_time: float, start_pos: np.ndarray, target_pos: np.ndarray, duration: float, 
              a: float = 0.05, start_quat: np.ndarray = None, target_quat: np.ndarray = None):
        s = np.clip(phase_time / duration, 0.0, 1.0)
        s_prime = 1.0 - np.cos(np.pi * s)
        
        p_des = start_pos + (target_pos - start_pos) * (0.5 * s_prime)
        z_clearance = a * np.sin(np.pi * s)
        p_des[2] = start_pos[2] + (target_pos[2] - start_pos[2]) * s + z_clearance
        
        ds_dt = np.pi / duration * np.sin(np.pi * s)
        v_xy_scale = 0.5 * (target_pos[:2] - start_pos[:2]) * ds_dt
        vz = (target_pos[2] - start_pos[2]) / duration + a * (np.pi / duration) * np.cos(np.pi * s)
        v_des = np.array([v_xy_scale[0], v_xy_scale[1], vz])
        
        d2s_dt2 = (np.pi ** 2 / duration ** 2) * np.cos(np.pi * s)
        a_xy_scale = 0.5 * (target_pos[:2] - start_pos[:2]) * d2s_dt2
        az = -a * (np.pi ** 2 / duration ** 2) * np.sin(np.pi * s)
        a_des = np.array([a_xy_scale[0], a_xy_scale[1], az])
        
        if start_quat is not None and target_quat is not None:
            q0 = np.array(start_quat, dtype=float)
            q1 = np.array(target_quat, dtype=float)
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot
            if dot > 0.9995:
                quat_des = q0 + s * (q1 - q0)
                quat_des /= np.linalg.norm(quat_des)
            else:
                theta_0 = np.arccos(dot)
                theta = theta_0 * s
                q2 = q1 - q0 * dot
                q2 /= np.linalg.norm(q2)
                quat_des = q0 * np.cos(theta) + q2 * np.sin(theta)
        else:
            quat_des = np.array([1.0, 0.0, 0.0, 0.0])
            
        return p_des, v_des, a_des, quat_des

    def compute_torques(self, current_state, com_target=None, foot_target=None, angle_target=0):
        if self.state == 1:
            return self.singlesupport(current_state, com_target, foot_target, angle_target)
        elif self.state == 2:
            return self.doublesupport(current_state, com_target, foot_target, angle_target)

    def doublesupport(self, current_state, com_target=None, foot_target=None, angle_target=0.0, zmp_target=None):
        if not self.initialized:
            self.q0 = current_state["joint_pos"].copy()
            self.pelvis_quat_des = current_state["base_quat"].copy()
            
            torso_quat = np.zeros(4)
            mujoco.mju_mat2Quat(torso_quat, self.data.xmat[self.torso_id])
            self.torso_quat_des = torso_quat.copy()
            
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self.com_initial_pos = self.data.subtree_com[0].copy()
            
            self.contact_initial_pos = []
            for body_id in [self.left_ankle_id, self.right_ankle_id]:
                for offset in self.contact_offsets:
                    p_world = self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3, 3) @ offset
                    self.contact_initial_pos.append(p_world.copy())
            self.initialized = True

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        # Print diagnostics for DSP
        com_current = self.data.subtree_com[0].copy()
        com_des_debug = com_target.copy() if com_target is not None else self.com_initial_pos.copy()
        
        left_sole_pos = self.data.xpos[self.left_ankle_id].copy()
        right_sole_pos = self.data.xpos[self.right_ankle_id].copy()
        
        left_sole_quat = np.zeros(4)
        right_sole_quat = np.zeros(4)
        mujoco.mju_mat2Quat(left_sole_quat, self.data.xmat[self.left_ankle_id])
        mujoco.mju_mat2Quat(right_sole_quat, self.data.xmat[self.right_ankle_id])

        print(f"[DSP Debug] Phase Time: {self.phase_time:.3f}s")
        print(f"  CoM Pos: X={com_current[0]:.3f}, Y={com_current[1]:.3f}, Z={com_current[2]:.3f}")
        print(f"  Target CoM: X={com_des_debug[0]:.3f}, Y={com_des_debug[1]:.3f}, Z={com_des_debug[2]:.3f}")
        print(f"  Left Sole Pos: X={left_sole_pos[0]:.3f}, Y={left_sole_pos[1]:.3f}, Z={left_sole_pos[2]:.3f} | Quat: {left_sole_quat}")
        print(f"  Right Sole Pos: X={right_sole_pos[0]:.3f}, Y={right_sole_pos[1]:.3f}, Z={right_sole_pos[2]:.3f} | Quat: {right_sole_quat}")
        print(f"  Target Footstep: {foot_target}")
        
        mujoco.mj_fullM(self.model, self.data, self.M)
        self.A[:, :] = 0
        self.A[:, :self.nv] = self.M
        S = np.zeros((self.nv, self.nv))
        np.fill_diagonal(S[6:, 6:], 1.0)
        self.A[:, self.nv:2*self.nv] = -S
        
        self.b = -self.data.qfrc_bias.copy()
        self.W[:, :] = 0
        self.t[:] = 0
        
        row_idx = 0
        contact_idx = 0
        
        for body_id in [self.left_ankle_id, self.right_ankle_id]:
            for offset in self.contact_offsets:
                p_world = self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3, 3) @ offset
                jacp = np.zeros((3, self.nv))
                mujoco.mj_jac(self.model, self.data, jacp, None, p_world, body_id)
                
                self.A[:, 2*self.nv + contact_idx*3 : 2*self.nv + (contact_idx+1)*3] = -jacp.T
                
                self.W[row_idx:row_idx+3, :self.nv] = self.w_contact * jacp
                v_contact = jacp @ self.data.qvel
                p_err = self.contact_initial_pos[contact_idx] - p_world
                acc_des = self.Kp_contact * p_err - self.Kd_contact * v_contact
                self.t[row_idx:row_idx+3] = self.w_contact * acc_des
                
                row_idx += 3
                contact_idx += 1

        jac_com = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac_com, self.pelvis_id)
        
        self.W[row_idx:row_idx+3, :self.nv] = self.w_com * jac_com
        
        com_vel_current = jac_com @ self.data.qvel
        com_err = com_des_debug - com_current
        acc_des_com = self.Kp_com * com_err - self.Kd_com * com_vel_current
        self.t[row_idx:row_idx+3] = self.w_com * acc_des_com
        row_idx += 3

        jacp_pelvis = np.zeros((3, self.nv))
        jacr_pelvis = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp_pelvis, jacr_pelvis, self.pelvis_id)
        
        self.W[row_idx:row_idx+3, :self.nv] = self.w_pelvis * jacr_pelvis
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, self.pelvis_quat_des, current_state["base_quat"])
        w_err = 0.0 - (jacr_pelvis @ self.data.qvel)
        acc_des_rot = self.Kp_pelvis * err_rot + self.Kd_pelvis * w_err
        self.t[row_idx:row_idx+3] = self.w_pelvis * acc_des_rot
        row_idx += 3
        
        jacr_torso = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, np.zeros((3, self.nv)), jacr_torso, self.torso_id)
        
        self.W[row_idx:row_idx+3, :self.nv] = self.w_torso * jacr_torso
        torso_quat = np.zeros(4)
        mujoco.mju_mat2Quat(torso_quat, self.data.xmat[self.torso_id])
        err_rot_torso = np.zeros(3)
        mujoco.mju_subQuat(err_rot_torso, self.torso_quat_des, torso_quat)
        w_err_torso = 0.0 - (jacr_torso @ self.data.qvel)
        acc_des_torso = self.Kp_torso * err_rot_torso + self.Kd_torso * w_err_torso
        self.t[row_idx:row_idx+3] = self.w_torso * acc_des_torso
        row_idx += 3
        
        arm_act_start = 12
        W_posture_diag = np.ones(self.nu) * self.w_posture
        W_posture_diag[arm_act_start:] = 500.0  
        Kp_vec = np.ones(self.nu) * self.Kp_posture
        Kd_vec = np.ones(self.nu) * self.Kd_posture
        Kp_vec[arm_act_start:] = 300.0  
        Kd_vec[arm_act_start:] = 50.0   

        q_err = self.q0 - current_state["joint_pos"]
        v_err = 0.0 - current_state["joint_vel"]
        acc_des_posture = Kp_vec * q_err + Kd_vec * v_err

        self.W[row_idx:row_idx+self.nu, 6:self.nv] = np.diag(W_posture_diag)
        self.t[row_idx:row_idx+self.nu] = W_posture_diag * acc_des_posture
        row_idx += self.nu

        Q = self.W.T @ self.W + self.reg
        q_o = -self.W.T @ self.t
        
        if not self.qp_init:
            self.qp.init(Q, q_o, self.A, self.b, self.C, self.dmin, self.dmax)
            self.qp_init = True
        else:
            self.qp.update(Q, q_o, self.A, self.b, self.C, self.dmin, self.dmax)
            
        self.qp.solve()

        self.phase_time += self.dt
        if self.phase_time >= self.dsp_duration:
            self.state = 1  # Transition to Single Support Phase
            self.phase_time = 0.0
            self.initialized = False

        sol = self.qp.results.x
        tau = sol[self.nv : 2*self.nv]
        return tau[6:].copy()

    def singlesupport(self, current_state, com_target=None, foot_target=None, angle_target=0.0):
        if self.side == 0:
            swing_id = self.left_ankle_id
            stance_id = self.right_ankle_id
            swing_indices = list(range(0, 4))
            stance_indices = list(range(4, 8))
        else:
            swing_id = self.right_ankle_id
            stance_id = self.left_ankle_id
            swing_indices = list(range(4, 8))
            stance_indices = list(range(0, 4))

        if not self.initialized:
            self.q0 = current_state["joint_pos"].copy()
            self.pelvis_quat_des = current_state["base_quat"].copy()
            
            torso_quat = np.zeros(4)
            mujoco.mju_mat2Quat(torso_quat, self.data.xmat[self.torso_id])
            self.torso_quat_des = torso_quat.copy()
            
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self.com_initial_pos = self.data.subtree_com[0].copy()
            
            self.ssp_stance_contact_pos = []
            for offset in self.contact_offsets:
                p_world = self.data.xpos[stance_id] + self.data.xmat[stance_id].reshape(3, 3) @ offset
                self.ssp_stance_contact_pos.append(p_world.copy())

            self.swing_start_pos = self.data.xpos[swing_id].copy()
            self.swing_start_quat = np.zeros(4)
            mujoco.mju_mat2Quat(self.swing_start_quat, self.data.xmat[swing_id])
            
            self.swing_target_quat = np.array([
                np.cos(angle_target / 2.0), 0.0, 0.0, np.sin(angle_target / 2.0)
            ])
            
            if foot_target is not None and not np.allclose(foot_target[:2], 0.0):
                self.swing_target_pos = np.array(foot_target, dtype=float)
                self.swing_target_pos[2] = max(self.swing_start_pos[2], 0.035)
            else:
                self.swing_target_pos = self.swing_start_pos.copy()
                self.swing_target_pos[0] += 0.15
                self.swing_target_pos[2] = 0.035
                
            self.initialized = True

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        # Print diagnostics for SSP
        com_current = self.data.subtree_com[0].copy()
        com_des_debug = com_target.copy() if com_target is not None else self.com_initial_pos.copy()
        
        left_sole_pos = self.data.xpos[self.left_ankle_id].copy()
        right_sole_pos = self.data.xpos[self.right_ankle_id].copy()
        
        left_sole_quat = np.zeros(4)
        right_sole_quat = np.zeros(4)
        mujoco.mju_mat2Quat(left_sole_quat, self.data.xmat[self.left_ankle_id])
        mujoco.mju_mat2Quat(right_sole_quat, self.data.xmat[self.right_ankle_id])

        swing_pos_current = self.data.xpos[swing_id].copy()
        swing_quat_current = left_sole_quat if self.side == 0 else right_sole_quat

        print(f"[SSP Debug] Side: {'Left' if self.side == 0 else 'Right'} | Phase Time: {self.phase_time:.3f}s")
        print(f"  CoM Pos: X={com_current[0]:.3f}, Y={com_current[1]:.3f}, Z={com_current[2]:.3f}")
        print(f"  Target CoM: X={com_des_debug[0]:.3f}, Y={com_des_debug[1]:.3f}, Z={com_des_debug[2]:.3f}")
        print(f"  Left Sole Pos: X={left_sole_pos[0]:.3f}, Y={left_sole_pos[1]:.3f}, Z={left_sole_pos[2]:.3f} | Quat: {left_sole_quat}")
        print(f"  Right Sole Pos: X={right_sole_pos[0]:.3f}, Y={right_sole_pos[1]:.3f}, Z={right_sole_pos[2]:.3f} | Quat: {right_sole_quat}")
        print(f"  Swing Sole Pos: X={swing_pos_current[0]:.3f}, Y={swing_pos_current[1]:.3f}, Z={swing_pos_current[2]:.3f} | Quat: {swing_quat_current}")
        print(f"  Target Footstep Pos: X={self.swing_target_pos[0]:.3f}, Y={self.swing_target_pos[1]:.3f}, Z={self.swing_target_pos[2]:.3f} | Target Quat: {self.swing_target_quat}")

        mujoco.mj_fullM(self.model, self.data, self.M)
        self.A[:, :] = 0
        self.A[:, :self.nv] = self.M
        S = np.zeros((self.nv, self.nv))
        np.fill_diagonal(S[6:, 6:], 1.0)
        self.A[:, self.nv:2*self.nv] = -S
        
        self.b = -self.data.qfrc_bias.copy()
        self.W[:, :] = 0
        self.t[:] = 0

        row_idx = 0

        # Stance Foot Contact Constraints
        for i_idx, contact_idx in enumerate(stance_indices):
            offset = self.contact_offsets[contact_idx % 4]
            p_world = self.data.xpos[stance_id] + self.data.xmat[stance_id].reshape(3, 3) @ offset
            jacp = np.zeros((3, self.nv))
            mujoco.mj_jac(self.model, self.data, jacp, None, p_world, stance_id)
            
            self.A[:, 2*self.nv + contact_idx*3 : 2*self.nv + (contact_idx+1)*3] = -jacp.T
            
            self.W[row_idx:row_idx+3, :self.nv] = self.w_contact * jacp
            v_contact = jacp @ self.data.qvel
            p_err = self.ssp_stance_contact_pos[i_idx] - p_world
            acc_des = self.Kp_contact * p_err - self.Kd_contact * v_contact
            self.t[row_idx:row_idx+3] = self.w_contact * acc_des
            row_idx += 3
        jacr_stance = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, np.zeros((3, self.nv)), jacr_stance, stance_id)
        
        self.W[row_idx:row_idx+3, :self.nv] = self.w_contact * jacr_stance
        stance_quat = np.zeros(4)
        mujoco.mju_mat2Quat(stance_quat, self.data.xmat[stance_id])
        
        err_rot_stance = np.zeros(3)
        # Force the foot to strictly maintain a flat [1, 0, 0, 0] orientation
        mujoco.mju_subQuat(err_rot_stance, np.array([1.0, 0.0, 0.0, 0.0]), stance_quat)
        w_stance_curr_rot = 0.0 - (jacr_stance @ self.data.qvel)
        
        acc_des_stance_r = 500.0 * err_rot_stance + 50.0 * w_stance_curr_rot
        self.t[row_idx:row_idx+3] = self.w_contact * acc_des_stance_r
        row_idx += 3
        # Zero-out swing foot contact forces
        for contact_idx in swing_indices:
            self.A[:, 2*self.nv + contact_idx*3 : 2*self.nv + (contact_idx+1)*3] = 0.0

        # Swing Foot Position & Orientation Tasks
        s_phase = np.clip(self.phase_time / self.ssp_duration, 0.0, 1.0)
        ramp = np.sin(0.5 * np.pi * np.clip(s_phase / 0.2, 0.0, 1.0))

        p_des_sw, v_des_sw, a_des_sw, quat_des_sw = self.swing(
            self.phase_time, self.swing_start_pos, self.swing_target_pos, self.ssp_duration,
            a=0.05, start_quat=self.swing_start_quat, target_quat=self.swing_target_quat
        )

        jacp_sw = np.zeros((3, self.nv))
        jacr_sw = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp_sw, jacr_sw, swing_id)

        w_swing = 1000.0 * ramp  # Increased from 200.0 to force forward step progression
        v_sw_curr = jacp_sw @ self.data.qvel
        
        # Increased PD gains for the swing foot trajectory tracking
        acc_des_sw_p = a_des_sw + 400.0 * ramp * (p_des_sw - swing_pos_current) - 50.0 * (v_sw_curr - v_des_sw)
        
        self.W[row_idx:row_idx+3, :self.nv] = w_swing * jacp_sw
        self.t[row_idx:row_idx+3] = w_swing * acc_des_sw_p
        row_idx += 3

        # Swing Foot Orientation Task (Flat landing)
        swing_quat = np.zeros(4)
        mujoco.mju_mat2Quat(swing_quat, self.data.xmat[swing_id])
        err_rot_sw = np.zeros(3)
        mujoco.mju_subQuat(err_rot_sw, quat_des_sw, swing_quat)
        w_sw_curr_rot = 0.0 - (jacr_sw @ self.data.qvel)
        
        # Increased rotational tracking stiffness and weight
        acc_des_sw_r = 500.0 * err_rot_sw + 50.0 * w_sw_curr_rot
        
        self.W[row_idx:row_idx+3, :self.nv] = 1000.0 * jacr_sw  # Increased from 300.0
        self.t[row_idx:row_idx+3] = 1000.0 * acc_des_sw_r
        row_idx += 3


        # CoM Task
        jac_com = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac_com, self.pelvis_id)
        self.W[row_idx:row_idx+3, :self.nv] = self.w_com * jac_com
        com_err = com_des_debug - com_current
        acc_des_com = self.Kp_com * com_err - self.Kd_com * (jac_com @ self.data.qvel)
        self.t[row_idx:row_idx+3] = self.w_com * acc_des_com
        row_idx += 3

        # Pelvis & Torso Orientation Tasks
        jacr_pelvis = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, np.zeros((3, self.nv)), jacr_pelvis, self.pelvis_id)
        self.W[row_idx:row_idx+3, :self.nv] = self.w_pelvis * jacr_pelvis
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, self.pelvis_quat_des, current_state["base_quat"])
        self.t[row_idx:row_idx+3] = self.w_pelvis * (self.Kp_pelvis * err_rot + self.Kd_pelvis * (-jacr_pelvis @ self.data.qvel))
        row_idx += 3

        jacr_torso = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, np.zeros((3, self.nv)), jacr_torso, self.torso_id)
        self.W[row_idx:row_idx+3, :self.nv] = self.w_torso * jacr_torso
        torso_quat = np.zeros(4)
        mujoco.mju_mat2Quat(torso_quat, self.data.xmat[self.torso_id])
        err_rot_torso = np.zeros(3)
        mujoco.mju_subQuat(err_rot_torso, self.torso_quat_des, torso_quat)
        self.t[row_idx:row_idx+3] = self.w_torso * (self.Kp_torso * err_rot_torso + self.Kd_torso * (-jacr_torso @ self.data.qvel))
        row_idx += 3

        # Unified Posture Control (Waist + Arms starting at index 12)
        arm_act_start = 15
        W_posture_diag = np.ones(self.nu) * self.w_posture
        W_posture_diag[arm_act_start:] = 2000.0  

        Kp_vec = np.ones(self.nu) * self.Kp_posture
        Kd_vec = np.ones(self.nu) * self.Kd_posture
        Kp_vec[arm_act_start:] = 600.0  
        Kd_vec[arm_act_start:] = 100.0   

        q_err = self.q0 - current_state["joint_pos"]
        q_err[14] = 0.0  # Prevent waist pitch folding
        v_err = 0.0 - current_state["joint_vel"]
        self.W[row_idx:row_idx+self.nu, 6:self.nv] = np.diag(W_posture_diag)
        self.t[row_idx:row_idx+self.nu] = W_posture_diag * (Kp_vec * q_err + Kd_vec * v_err)
        row_idx += self.nu

        Q = self.W.T @ self.W + self.reg
        q_o = -self.W.T @ self.t
        
        if not self.qp_init:
            self.qp.init(Q, q_o, self.A, self.b, self.C, self.dmin, self.dmax)
            self.qp_init = True
        else:
            self.qp.update(Q, q_o, self.A, self.b, self.C, self.dmin, self.dmax)
            
        self.qp.solve()

        self.phase_time += self.dt
        if self.phase_time >= self.ssp_duration:
            self.step_counter += 1
            self.side = 1 - self.side
            self.state = 2  # Return to Double Support Phase
            self.phase_time = 0.0
            self.initialized = False

        sol = self.qp.results.x
        tau = sol[self.nv : 2*self.nv]
        joint_torques = tau[6:].copy()

        # Direct Override for Arms
        arm_joint_start_idx = 15  
        for i in range(arm_joint_start_idx, self.nu):
            q_cur = current_state["joint_pos"][i]
            v_cur = current_state["joint_vel"][i]
            q_ref = self.q0[i]
            
            joint_torques[i] = 200.0 * (q_ref - q_cur) - 20.0 * v_cur

        return joint_torques