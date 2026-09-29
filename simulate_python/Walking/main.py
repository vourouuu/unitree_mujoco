import time, math, mujoco, threading
import mujoco.viewer
from threading import Thread
import numpy as np
import config
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from Walking.footsteps.footsteps_generator import FootstepGenerator
from Walking.estimation.state import StateEstimator
from Walking.planning.mpc import MPCPlanner
from Walking.planning.wbc import WBCController

locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

shared_plan = {"next_foot_pose": None, "target_angle": 0.0, "com_tr": None, "zmp_tr": None, 'side': 0, 'future_steps': []}

def update_footprints(model, data, future_steps):
    """Updates current foot pos and future planned steps."""

    left_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_support_poly")
    right_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_support_poly")

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    if left_geom_id != -1 and left_foot_id != -1:
        left_pos = data.xpos[left_foot_id]
        model.geom_pos[left_geom_id] = [left_pos[0], left_pos[1], 0.002] 
        model.geom_quat[left_geom_id] = data.xquat[left_foot_id]

    if right_geom_id != -1 and right_foot_id != -1:
        right_pos = data.xpos[right_foot_id]
        model.geom_pos[right_geom_id] = [right_pos[0], right_pos[1], 0.002]
        model.geom_quat[right_geom_id] = data.xquat[right_foot_id]


    for i in range(10): # Matches the 10 step_X geoms in XML
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"step_{i}")
        if geom_id != -1:
            if future_steps is not None and i < len(future_steps):
                model.geom_pos[geom_id] = [future_steps[i][0], future_steps[i][1], 0.002]
            else:
                model.geom_pos[geom_id] = [0, 0, -1]

def PlannerThread():
    footstep_gen = FootstepGenerator(step_duration=0.4)
    state_estimator = StateEstimator(mj_model, mj_data)
    mpc_planner = MPCPlanner(N=15, dt=0.002, z_com=0.72)
    current_angle = 0.0
    
    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    
    # Pre-fetch foot IDs
    left_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    
    while viewer.is_running():
        start_time = time.perf_counter()
        
        locker.acquire()
        current_state = state_estimator.update()
        current_side = shared_plan.get("current_side", 0)
        s = 1 if current_side == 0 else -1
        
        base_pos = mj_data.xpos[pelvis_id].copy()
        
        if current_side==0:
            stance_foot_id=right_foot_id
        else:
            stance_foot_id=left_foot_id

        stance_pos = mj_data.xpos[stance_foot_id][:2]
        
        offset = np.array([0.,s*(footstep_gen.w/2)])
        R_angle = np.array([[np.cos(current_angle), -np.sin(current_angle)], 
                            [np.sin(current_angle), np.cos(current_angle)]])
        center_anchor=stance_pos+R_angle@offset
        

        forward_vec = np.array([np.cos(current_angle), np.sin(current_angle)])
        lateral_vec = np.array([-np.sin(current_angle), np.cos(current_angle)])
        nominal_base_pos = (np.dot(base_pos[:2], forward_vec) * forward_vec + 
                            np.dot(center_anchor, lateral_vec) * lateral_vec)
        
        com_pos=mj_data.subtree_com[0].copy()
        com_vel=current_state["base_vel"][:3]
        
        omega=mpc_planner.omega
        xi_meas=com_pos[:2]+com_vel[:2]/omega
        locker.release()
        
        v_current=current_state["base_vel"][:2]
        v_desired=np.array([0.4,0])
        k_feedback=np.array([0.05,0.05])


        support_polys = footstep_gen.nominal_footstep_polygons(
            mj_model, mj_data, nominal_base_pos, v_current, v_desired, 
            k_feedback, s, mpc_planner.N, 0, current_angle, 0, footstep_gen.max_turn)

        # Extract centers of all future polygons for visualization
        future_centers = []
        for p in support_polys:
            cx=0.5*(p[0]+p[1])
            cy=0.5*(p[2]+p[3])
            future_centers.append([cx, cy])

        target_pos= np.array([future_centers[0][0], future_centers[0][1], 0.0])
        target_angle = current_angle

        traj = np.array(future_centers)
        com_traj, opt_zmp = mpc_planner.compute_trajectory(com_pos, xi_meas, traj, support_polys)

        locker.acquire()
        shared_plan["next_foot_pose"] = target_pos
        shared_plan["target_angle"] = target_angle
        shared_plan["com_tr"] = com_traj
        shared_plan["zmp_tr"] = opt_zmp
        shared_plan["future_steps"] = future_centers
        locker.release()

        elapsed = time.perf_counter() - start_time
        sleep_time = 0.02 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

def SimulationThread():
    global mj_data, mj_model
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    state_estimator = StateEstimator(mj_model, mj_data)
    wbc_controller = WBCController(mj_model, mj_data, ssp_duration=0.4, dsp_duration=0.1, dt=config.SIMULATE_DT)
    
    while viewer.is_running():
        step_start = time.perf_counter()
        
        locker.acquire()
        current_state = state_estimator.update()
        current_foot_target = shared_plan["next_foot_pose"]
        current_angle_target = shared_plan["target_angle"]
        com_trajectory = shared_plan["com_tr"]
        zmp_trajectory = shared_plan["zmp_tr"]
        shared_plan["current_side"] = wbc_controller.side
        locker.release()

        # CoM target interpolation
        if com_trajectory is None or len(com_trajectory) == 0:
            com_target = None
        else:
            fraction = min(wbc_controller.phase_time / wbc_controller.ssp_duration, 1.0)
            horizon_idx = int(fraction * (len(com_trajectory) - 1))
            com_target = com_trajectory[horizon_idx]

        # ZMP target
        if zmp_trajectory is None or len(zmp_trajectory) == 0:
            zmp_target = None
        else:
            zmp_target = zmp_trajectory[0]

        # Swing foot target
        if wbc_controller.state==1 and current_foot_target is not None:
            swing_target=current_foot_target.copy()
        else:
            swing_target=None

        torques = wbc_controller.compute_torques(
            current_state,
            com_target=com_target,
            foot_target=swing_target,
            angle_target=current_angle_target
        )
        
        locker.acquire()
        mj_data.ctrl[:] = torques
        mujoco.mj_step(mj_model, mj_data)
        locker.release()
        
        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        future_steps = shared_plan.get("future_steps", [])
        update_footprints(mj_model, mj_data, future_steps)
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    planner_thread = Thread(target=PlannerThread)
    viewer_thread.start()
    sim_thread.start()
    planner_thread.start()