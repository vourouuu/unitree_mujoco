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
        self.state =2  
        
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

        # Task Weights
        self.w_contact = 50
        self.w_com = 100
        self.w_pelvis = 10
        self.w_torso = 5
        self.w_posture = 0.1

        # PD Gains
        self.Kp_contact = 100
        self.Kd_contact = 20
        self.Kp_com = 40
        self.Kd_com = 20
        self.Kp_pelvis = 100
        self.Kd_pelvis = 20
        self.Kp_torso = 100
        self.Kd_torso = 20
        self.Kp_posture = 40
        self.Kd_posture = 10

        self.qp = proxsuite.proxqp.dense.QP(self.n_vars, self.n_eq, self.n_in)
        self.qp_init = False
        self.initialized = False

    def _setup_friction_cone(self):
        mu = 0.8
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
#not transitioning here yet
    def compute_torques(self, current_state, com_target=None, foot_target=None, angle_target=0):
        if self.state == 0:
            return self.standing(current_state)
        elif self.state == 1:
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
        
        #Contact Constraints
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

        #CoM Position Task
        jac_com = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac_com, self.pelvis_id)
        
        self.W[row_idx:row_idx+3, :self.nv] = self.w_com * jac_com
        com_current = self.data.subtree_com[0]
        
        com_des = com_target.copy() if com_target is not None else self.com_initial_pos.copy()
        
        com_vel_current = jac_com @ self.data.qvel
        com_err = com_des - com_current
        acc_des_com = self.Kp_com * com_err - self.Kd_com * com_vel_current
        self.t[row_idx:row_idx+3] = self.w_com * acc_des_com
        row_idx += 3

        #Pelvis Orientation Task
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
        
        #Torso Orientation Task
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
        
        #Joint Posture Task
        arm_act_start = 15
        W_posture_diag = np.ones(self.nu) * 0.01
        W_posture_diag[arm_act_start:] = 5
        #the arms fell and dropped the robot i put distinct weight
        Kp_vec = np.ones(self.nu) * 20
        Kd_vec = np.ones(self.nu) * 5
        Kp_vec[arm_act_start:] = 100
        Kd_vec[arm_act_start:] = 20

        q_err = self.q0 - current_state["joint_pos"]
        v_err = 0.0 - current_state["joint_vel"]
        acc_des_posture = Kp_vec * q_err + Kd_vec * v_err

        self.W[row_idx:row_idx+self.nu, 6:self.nv] = np.diag(W_posture_diag)
        self.t[row_idx:row_idx+self.nu] = W_posture_diag * acc_des_posture
        row_idx += self.nu

        # QP Solve
        Q = self.W.T @ self.W + self.reg
        q_o = -self.W.T @ self.t
        
        if not self.qp_init:
            self.qp.init(Q, q_o, self.A, self.b, self.C, self.dmin, self.dmax)
            self.qp_init = True
        else:
            self.qp.update(Q, q_o, self.A, self.b, self.C, self.dmin, self.dmax)
            
        self.qp.solve()

        sol = self.qp.results.x
        tau = sol[self.nv : 2*self.nv]
        return tau[6:].copy()

    def singlesupport(self, current_state, com_target=None, foot_target=None, angle_target=0):
        return np.zeros(self.nu)

    def standing(self, current_state):
        if not self.initialized:
            self.q0=current_state["joint_pos"].copy()
            self.pelvis_pos_des=current_state["base_pos"].copy()
            self.pelvis_quat_des=current_state["base_quat"].copy()
            
            torso_quat = np.zeros(4)
            mujoco.mju_mat2Quat(torso_quat,self.data.xmat[self.torso_id])
            self.torso_quat_des=torso_quat.copy()
            
            self.contact_initial_pos=[]
            for body_id in [self.left_ankle_id, self.right_ankle_id]:
                for offset in self.contact_offsets:
                    p_world = self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3,3) @ offset
                    self.contact_initial_pos.append(p_world.copy())
            self.initialized=True

        # Ensure kinematics are up to date
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # Compute Mass matrix and Bias
        mujoco.mj_fullM(self.model, self.data, self.M)
        self.A[:,:] = 0
        self.A[:,:self.nv]=self.M
        #The floating base is 0
        S=np.zeros((self.nv,self.nv))
        np.fill_diagonal(S[6:,6:],1)
        self.A[:,self.nv:2*self.nv]=-S
        
        self.b =-self.data.qfrc_bias.copy()
        self.W[:,:]=0
        self.t[:]=0
        
        row_idx=0
        contact_idx=0
        
        # Contact Task  Dynamics Jacobian
        for body_id in [self.left_ankle_id, self.right_ankle_id]:
            for offset in self.contact_offsets:
                #global position
                p_world = self.data.xpos[body_id]+self.data.xmat[body_id].reshape(3,3)@offset
                #the jacobian that maps velocity to operation space
                jacp=np.zeros((3,self.nv))
                mujoco.mj_jac(self.model,self.data,jacp,None,p_world,body_id)
                

                self.A[:,2*self.nv+contact_idx*3:2*self.nv+(contact_idx+1)*3]=-jacp.T
                
                # Add contact Task
                self.W[row_idx:row_idx+3,:self.nv]=self.w_contact*jacp
                
                v_contact = jacp@self.data.qvel
                p_err=self.contact_initial_pos[contact_idx]-p_world
                acc_des=self.Kp_contact*p_err-self.Kd_contact*v_contact
                
                self.t[row_idx:row_idx+3] = self.w_contact * acc_des
                
                row_idx+=3
                contact_idx+=1

        # Pelvis Task
        jacp_pelvis = np.zeros((3, self.nv))
        jacr_pelvis = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp_pelvis, jacr_pelvis, self.pelvis_id)
        
        # Pelvis Position
        self.W[row_idx:row_idx+3, :self.nv] = self.w_pelvis * jacp_pelvis
        p_err = self.pelvis_pos_des - current_state["base_pos"]
        v_err = 0.0 - (jacp_pelvis @ self.data.qvel)
        acc_des = self.Kp_pelvis * p_err + self.Kd_pelvis * v_err
        self.t[row_idx:row_idx+3] = self.w_pelvis * acc_des
        row_idx += 3
        
        # Pelvis Orientation
        self.W[row_idx:row_idx+3, :self.nv] = self.w_pelvis * jacr_pelvis
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, self.pelvis_quat_des, current_state["base_quat"])
        w_err = 0.0 - (jacr_pelvis @ self.data.qvel)
        acc_des_rot = self.Kp_pelvis * err_rot + self.Kd_pelvis * w_err
        self.t[row_idx:row_idx+3] = self.w_pelvis * acc_des_rot
        row_idx += 3
        
        # Torso Task
        jacp_torso = np.zeros((3, self.nv))
        jacr_torso = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp_torso, jacr_torso, self.torso_id)
        
        self.W[row_idx:row_idx+3, :self.nv] = self.w_torso * jacr_torso
        torso_quat = np.zeros(4)
        mujoco.mju_mat2Quat(torso_quat, self.data.xmat[self.torso_id])
        err_rot_torso = np.zeros(3)
        mujoco.mju_subQuat(err_rot_torso, self.torso_quat_des, torso_quat)
        w_err_torso = 0.0 - (jacr_torso @ self.data.qvel)
        acc_des_torso = self.Kp_torso * err_rot_torso + self.Kd_torso * w_err_torso
        self.t[row_idx:row_idx+3] = self.w_torso * acc_des_torso
        row_idx += 3
        
        # Joint Task (Includes scaled arm tracking)
        arm_act_start = 15
        W_posture_diag = np.ones(self.nu) * self.w_posture
        W_posture_diag[arm_act_start:] = 5.0
        
        Kp_vec = np.ones(self.nu) * self.Kp_posture
        Kd_vec = np.ones(self.nu) * self.Kd_posture
        Kp_vec[arm_act_start:] = 100.0
        Kd_vec[arm_act_start:] = 20.0

        q_err = self.q0 - current_state["joint_pos"]
        v_err = 0.0 - current_state["joint_vel"]
        acc_des_posture = Kp_vec * q_err + Kd_vec * v_err

        self.W[row_idx:row_idx+self.nu, 6:self.nv] = np.diag(W_posture_diag)
        self.t[row_idx:row_idx+self.nu] = W_posture_diag * acc_des_posture
        row_idx += self.nu
        
        # QP Solve
        Q = self.W.T @ self.W + self.reg
        q_o = -self.W.T @ self.t
        
        if not self.qp_init:
            self.qp.init(Q,q_o,self.A,self.b, self.C,self.dmin,self.dmax)
            self.qp_init=True
        else:
            self.qp.update(Q, q_o,self.A,self.b,self.C,self.dmin,self.dmax)
            
        self.qp.solve()
        
        sol=self.qp.results.x
        tau=sol[self.nv:2*self.nv]
        
        return tau[6:].copy()

