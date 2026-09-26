import time,math,mujoco,threading
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
    mpc_planner = MPCPlanner(N=5,dt=0.002,z_com=mj_data.subtree_com[0][2])#check actual z_com
    s=1
    current_angle=0
    while viewer.is_running():
        start_time = time.perf_counter()
        locker.acquire()
    
        current_state = state_estimator.update()
        
        hip_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
        hip_pos = mj_data.xpos[hip_body_id].copy()
        com_pos=mj_data.subtree_com[0]
        com_vel=current_state["base_vel"][:3]
        omega=mpc_planner.omega
        xi_meas=com_pos[:2]+com_vel[:2]/ omega
        locker.release()

        # Compute footsteps and trajectories 
        v_current = current_state["base_vel"][:2]
        v_desired=np.array([0.2,0])
        k_feedback=np.array([0.05,0.05])

        support_polys = footstep_gen.nominal_footstep_polygons(mj_model, mj_data, hip_pos[:2],v_current,v_desired, 
            k_feedback,s,mpc_planner.N,0,current_angle,0,footstep_gen.max_turn)


        target_pos =np.array([0.5*(support_polys[0][0]+support_polys[0][1]),
            0.5*(support_polys[0][2]+support_polys[0][3]),0])
        target_angle = current_angle

        traj = np.array([[0.5*(p[0]+p[1]),0.5*(p[2]+p[3])] for p in support_polys])
        com_traj, opt_zmp = mpc_planner.compute_trajectory(xi_meas,traj, support_polys)


        # Update shared plan safely
        locker.acquire()
        shared_plan["next_foot_pose"] = target_pos
        shared_plan["target_angle"] = target_angle
        shared_plan["com_tr"] = com_traj
        shared_plan["zmp_tr"] = opt_zmp
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

    state_estimator=StateEstimator(mj_model, mj_data)
    wbc_controller=WBCController(mj_model, mj_data, ssp_duration=0.4, dsp_duration=0.1, dt=config.SIMULATE_DT)
    t0 = time.perf_counter()

    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()


        
        current_state=state_estimator.update()

        elapsed=step_start-t0
        
        #Fetch plan
        current_foot_target = shared_plan["next_foot_pose"]
        current_angle_target = shared_plan["target_angle"]
        com_trajectory = shared_plan["com_tr"]
        zmp_trajectory = shared_plan["zmp_tr"]
        if com_trajectory is None:
            com_target=None
        else:
            com_target = com_trajectory[0]
        if zmp_trajectory is None:
            zmp_target=None
        else:
            zmp_target = zmp_trajectory[0]

        # com_measured = mj_data.subtree_com[0].copy() 
        # print(f"[CoM Check] Measured: {com_measured} | Target: {com_target}")
        torques = wbc_controller.compute_torques(
            current_state,
            com_target=com_target,
            foot_target=current_foot_target,
            angle_target=current_angle_target)
        mj_data.ctrl[:] = torques

        mujoco.mj_step(mj_model, mj_data)
        locker.release()
        
        time_until_next_step=mj_model.opt.timestep-(time.perf_counter() - step_start)
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