import numpy as np
import mujoco

class StateEstimator:
    def __init__(self,mj_model,mj_data):
        self.model=mj_model
        self.data=mj_data
        
        self.imu_quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self.imu_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.frame_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "frame_pos")
        self.frame_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "frame_vel")

    def get_sensor_data(self,sensor_id:int,dim:int) -> np.ndarray:
        if sensor_id==-1:return np.zeros(dim)
        adress=self.model.sensor_adr[sensor_id]
        return self.data.sensordata[adress:adress+dim].copy()

    def update(self) -> dict:
        base_pos=self.get_sensor_data(self.frame_pos_id, 3)
        base_quat=self.get_sensor_data(self.imu_quat_id, 4)
        base_vel=self.get_sensor_data(self.frame_vel_id, 3)
        base_omega=self.get_sensor_data(self.imu_gyro_id, 3)

        # Removes floating base wich isnt used in calculations
        joint_pos = self.data.qpos[7:].copy()
        joint_vel = self.data.qvel[6:].copy()

        return {"base_pos": base_pos,"base_quat": base_quat,"base_vel": base_vel,
        "base_omega": base_omega,"joint_pos": joint_pos,"joint_vel": joint_vel}