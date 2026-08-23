import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
import config
#new import
import numpy as np
import math
locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
#offline 
#maybe remove 
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

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():

    global mj_data, mj_model
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    l_shoulder_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch")
    r_shoulder_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch")
    
    t0 = time.perf_counter()

    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()
        elapsed = step_start - t0
        target_pos = 0.5 * math.sin(2.0 * math.pi * elapsed) 
        if l_shoulder_id != -1:
            mj_data.ctrl[l_shoulder_id] = target_pos
        if r_shoulder_id != -1:
            mj_data.ctrl[r_shoulder_id] = -target_pos  # Inverse motion for opposite arm
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


#here i get the dimensions of the robot
def get_measurements(mj_model):
    left_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    hip_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
    hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
    knee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
   # if there is a better way we can change this
    foot_geom_id = [g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] == left_foot_id][0]
    #*2 because it stores half sizes
    sole_length_x = float(mj_model.geom_aabb[foot_geom_id][3] * 2.0)
    sole_width_y = float(mj_model.geom_aabb[foot_geom_id][4] * 2.0)
    leg_length = float(np.linalg.norm(mj_data.xpos[hip_body_id] - mj_data.xpos[left_foot_id]))
    #assumption
    max_reach = leg_length*0.50
    hip_range = mj_model.jnt_range[hip_pitch_id]
    knee_range = mj_model.jnt_range[knee_id]
    #the values seem to be what is expected 0,06 0,0756 0,3282 [-2.5307  2.8798] [-0.087267  2.8798  ]
    return sole_length_x, sole_width_y, hip_range, knee_range, max_reach, leg_length


sole_length_x,sole_width_y,hip_range,knee_range,max_reach,leg_length=get_measurements(mj_model)
print('Sole:',sole_length_x,sole_width_y,'Hip Range:',hip_range,'Knee Range:',knee_range,'Reach:',max_reach,'Length:',leg_length)

#I used the raibert heuristic from the pdf
def generate_footstep(hip_pos:np.ndarray,v_current:np.ndarray,v_desired:np.ndarray,k:np.ndarray,s:int,T:float, 
    W:float,desired_angle_delta:float,angle:float=0.0,max_turn:float=0.35)->tuple[np.ndarray, float]:
   # s=1 or s=-1 for left or right

    x_foot=hip_pos[0]+T/2*v_current[0]+k[0]*(v_current[0]-v_desired[0])
    y_foot=hip_pos[1]+s*W/2+T/2.0*v_current[1]+k[1]*(v_current[1]-v_desired[1])
    target_pos=np.array([x_foot,y_foot])

    #Angle
    angle_delta=np.clip(desired_angle_delta,-max_turn,max_turn)
    target_angle=angle+angle_delta
    
    return target_pos, target_angle
#Todo discuss
def project_kinematics(target_pos:np.ndarray,hip_pos:np.ndarray,s:int,max_step:float,min_width:float)->np.ndarray:

    #Limit maximum reach circular clamping
    if np.linalg.norm(target_pos-hip_pos)>max_step:
        target_pos=hip_pos+((target_pos-hip_pos)/np.linalg.norm(target_pos-hip_pos))* max_step
        
    # Prevent lateral clamping
    # If s=1 y must be positive else negative
    dy_relative=target_pos[1]-hip_pos[1]
    if s==1 and dy_relative<min_width/2.0:
        target_pos[1]=hip_pos[1]+min_width/2.0
    elif dy_relative>-min_width/2.0:
        target_pos[1]=hip_pos[1]-min_width/2.0

    return target_pos
#fix hitting robot
def get_terrain_height(mj_model,mj_data,x:float,y:float,ray_start_z:float=1)->float:
    #A downward ray for Z-height
    pnt = np.array([x, y,ray_start_z])
    vec = np.array([0.0, 0.0, -1.0])
    geomid = np.array([-1], dtype=np.int32)
    #distance to intersection
    dist = mujoco.mj_ray(mj_model, mj_data, pnt, vec, None, 1, -1, geomid)
    return max(ray_start_z-dist,0)



MAX_STEP =max_reach
MIN_WIDTH =sole_width_y*1.2

z_origin = get_terrain_height(mj_model, mj_data, x=0.0, y=0.0)
print(f"Terrain height at (0.0, 0.0): {z_origin:.4f} m")
# raw_target_pos, target_yaw = generate_footstep(

# valid_target_pos = project_kinematics(

# target_z = get_terrain_height(


# mpc_command 


if __name__ == "__main__":
    
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    viewer_thread.start()
    sim_thread.start()
