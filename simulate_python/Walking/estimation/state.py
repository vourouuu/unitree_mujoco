import numpy as np
import mujoco

class StateEstimator:
    def __init__(self,mj_model,mj_data):
        self.model=mj_model
        self.data=mj_data
        self.frame_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "frame_pos")
        self.imu_quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self.frame_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "frame_vel")
        self.imu_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")


    def update(self) -> dict:
        adr=self.model.sensor_adr[self.frame_pos_id]
        base_pos=self.data.sensordata[adr:adr+3].copy()

        adr=self.model.sensor_adr[self.imu_quat_id]
        base_quat=self.data.sensordata[adr:adr+4].copy()

        adr=self.model.sensor_adr[self.frame_vel_id]
        base_vel=self.data.sensordata[adr:adr+3].copy()

        adr=self.model.sensor_adr[self.imu_gyro_id]
        base_omega=self.data.sensordata[adr:adr+3].copy()


        return {
            "base_pos": base_pos,"base_quat": base_quat,"base_vel": base_vel,"base_omega": base_omega,
            "joint_pos": self.data.qpos[7:].copy(),"joint_vel": self.data.qvel[6:].copy(),}