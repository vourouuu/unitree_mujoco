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
# physics time
mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)
shared_plan = {"next_foot_pose": None, "target_angle": 0.0, "com_tr": None, "zmp_tr": None,'side':0}

def PlannerThread():
    footstep_gen = FootstepGenerator(step_duration=0.4)
    state_estimator = StateEstimator(mj_model, mj_data)
    mpc_planner = MPCPlanner(N=15, dt=0.002, z_com=0.72)
    current_angle = 0.0
    
    # Get the pelvis body ID for root tracking
    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    
    while viewer.is_running():
        start_time = time.perf_counter()
        
        locker.acquire()
        current_state = state_estimator.update()
        
        # Read the current active swing side from the WBC controller
        current_side = shared_plan.get("current_side", 0)
        
        # Track the center of the robot rather than a specific hip link
        base_pos = mj_data.xpos[pelvis_id].copy()
        com_pos = mj_data.subtree_com[0].copy()
        com_vel = current_state["base_vel"][:3]
        
        omega = mpc_planner.omega
        xi_meas = com_pos[:2] + com_vel[:2] / omega
        locker.release()
        
        # Map controller side (0=Left Swing, 1=Right Swing) to multiplier (1=Left, -1=Right)
        s = 1 if current_side == 0 else -1

        v_current = current_state["base_vel"][:2]
        v_desired = np.array([0.4, 0.0])
        k_feedback = np.array([0.05, 0.05])

        # Generate support polygons relative to the pelvis center
        support_polys = footstep_gen.nominal_footstep_polygons(
            mj_model, mj_data, base_pos[:2], v_current, v_desired, 
            k_feedback, s, mpc_planner.N, 0, current_angle, 0, footstep_gen.max_turn
        )

        target_pos = np.array([
            0.5 * (support_polys[0][0] + support_polys[0][1]),
            0.5 * (support_polys[0][2] + support_polys[0][3]),
            0.0
        ])
        target_angle = current_angle

        traj = np.array([[0.5 * (p[0] + p[1]), 0.5 * (p[2] + p[3])] for p in support_polys])
        com_traj, opt_zmp = mpc_planner.compute_trajectory(com_pos, xi_meas, traj, support_polys)

        locker.acquire()
        shared_plan["next_foot_pose"] = target_pos
        shared_plan["target_angle"] = target_angle
        shared_plan["com_tr"] = com_traj
        shared_plan["zmp_tr"] = opt_zmp
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
    t0 = time.perf_counter()

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

        # Map phase progress safely across the N=15 horizon length
        if com_trajectory is None or len(com_trajectory) == 0:
            com_target = None
        else:
            fraction = min(wbc_controller.phase_time / wbc_controller.ssp_duration, 1.0)
            horizon_idx = int(fraction * (len(com_trajectory) - 1))
            com_target = com_trajectory[horizon_idx]

        if zmp_trajectory is None or len(zmp_trajectory) == 0:
            zmp_target = None
        else:
            zmp_target = zmp_trajectory[0]

        # Inject parabolic swing clearance (0.06m peak height)
        swing_target = current_foot_target.copy() if current_foot_target is not None else None
        if swing_target is not None:
            step_height = 0.06 
            swing_target[2] += step_height * (4 * fraction * (1 - fraction))

        torques = wbc_controller.compute_torques(
            current_state,
            com_target=com_target,
            foot_target=swing_target, # Pass the dynamically updated target
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