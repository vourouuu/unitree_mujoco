import time,math,mujoco,threading
import mujoco.viewer
from threading import Thread
import numpy as np
import config
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from Walking.footsteps.footsteps_generator import FootstepGenerator
from Walking.estimation.state import StateEstimator
# from Walking.planning.mpc import MPCPlanner
# from Walking.planning.wbc import WBCController



locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
#physics time
mj_model.opt.timestep = config.SIMULATE_DT
#actuators from xml 29
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)
shared_plan = {"next_foot_pose": None,"target_angle": 0.0,"com_tr": None,"zmp_tr": None}

def PlannerThread():

    footstep_gen=FootstepGenerator(step_duration=0.4)
    state_estimator=StateEstimator(mj_model, mj_data)
    # mpc_planner = MPCPlanner()

    while viewer.is_running():
        start_time = time.perf_counter()
        locker.acquire()
    
        current_state = state_estimator.update()
        
        hip_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
        hip_pos = mj_data.xpos[hip_body_id].copy()
        locker.release()

        # Compute footsteps and trajectories 
        v_current = current_state["base_vel"][:2]
        v_desired=np.array([0.2,0])
        k_feedback=np.array([0.05,0.05])
        
        target_pos,target_angle = footstep_gen.process_step(mj_model, mj_data, hip_pos[:2], v_current, 
        v_desired, k_feedback, s=1, desired_angle_delta=0, angle=0, z_start=1)
        
        # com_traj, zmp_traj = mpc_planner.compute_trajectory(target_pos)

        # Update shared plan safely
        locker.acquire()
        shared_plan["next_foot_pose"] = target_pos
        shared_plan["target_angle"] = target_angle
        # shared_plan["com_tr"] = com_traj
        # shared_plan["zmp_tr"] = zmp_traj
        locker.release()

        # Regulate planner frequency 
        elapsed = time.perf_counter() - start_time
        sleep_time = 0.02 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

def SimulationThread():
    global mj_data, mj_model
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    state_estimator=StateEstimator(mj_model, mj_data)
    l_shoulder_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch")
    r_shoulder_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch")
    t0 = time.perf_counter()

    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()
        
        current_state=state_estimator.update()

        elapsed=step_start-t0
        
        #Fetch plan
        current_foot_target = shared_plan["next_foot_pose"]
        current_angle_target = shared_plan["target_angle"]


        target_pos = -50 * math.sin(2*math.pi*elapsed) 
        if l_shoulder_id != -1:
            mj_data.ctrl[l_shoulder_id] = target_pos
        if r_shoulder_id != -1:
            mj_data.ctrl[r_shoulder_id] = -target_pos  

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
                
        mujoco.mj_step(mj_model, mj_data)
        locker.release()
        
        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
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