import numpy as np
import mujoco
class FootstepGenerator:
    def __init__(self,step_duration:float,step_width:float=0.18,max_turn:float=0.35,max_reach:float=0.5,min_width:float=0.09,
    sole_length_x:float=0.067,sole_width_y:float=0.075,hip_range=[-2.5307,2.8798],knee_range=[-0.087267,2.8798]):
        self.T=step_duration
        self.w=step_width
        self.max_reach=max_reach
        self.min_width=min_width
        self.max_turn=max_turn
        self.sole_length_x=sole_length_x
        self.sole_width_y=sole_width_y
        self.hip_range=hip_range
        self.knee_range=knee_range

    def get_measurements(self,mj_model,mj_data):
        left_foot_id=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        hip_body_id=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
        hip_pitch_id=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
        knee_id=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")

        foot_geom_id=[g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] == left_foot_id][0]
        hip_range=mj_model.jnt_range[hip_pitch_id]
        knee_range=mj_model.jnt_range[knee_id]
        sole_length_x=float(mj_model.geom_aabb[foot_geom_id][3] * 2.0)
        sole_width_y=float(mj_model.geom_aabb[foot_geom_id][4] * 2.0)
        leg_length=float(np.linalg.norm(mj_data.xpos[hip_body_id] - mj_data.xpos[left_foot_id]))
        #assumption
        max_reach=leg_length*0.50
        min_width=sole_width_y*1.2

        return leg_length,sole_length_x,sole_width_y,hip_range,knee_range,max_reach,min_width

    def generate_footstep(self,hip_pos:np.ndarray,v_current:np.ndarray,v_desired:np.ndarray,k:np.ndarray,s:int,
                        desired_angle_delta:float,angle:float,max_turn:float)->tuple[np.ndarray, float]:
        # s=1 or s=-1 for left or right
        x_foot=self.T/2*v_current[0]+k[0]*(v_current[0]-v_desired[0])
        y_foot=s*self.w/2+self.T/2*v_current[1]+k[1]*(v_current[1]-v_desired[1])
        target_pos=np.array([x_foot,y_foot])
        #Angle
        target_angle=angle+np.clip(desired_angle_delta,-max_turn,max_turn)
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        target_pos=hip_pos+R@target_pos
        return target_pos, target_angle

    def project_kinematics(self,target_pos:np.ndarray,hip_pos:np.ndarray,s:int,max_step:float,min_width:float)->np.ndarray:
        #Limit maximum reach circular clamping
        if np.linalg.norm(target_pos-hip_pos)>max_step:
            target_pos=hip_pos+((target_pos-hip_pos)/np.linalg.norm(target_pos-hip_pos))* max_step

        # Prevent lateral clamping
        # If s=1 y must be positive else negative
        dy_relative=target_pos[1]-hip_pos[1]
        if s==1 and dy_relative<min_width/2:
            target_pos[1]=hip_pos[1]+min_width/2
        elif dy_relative>-min_width/2:
            target_pos[1]=hip_pos[1]-min_width/2

        return target_pos

    def get_terrain_height(self,mj_model,mj_data,pos:np.ndarray,z_start:float,robot_root_id:int=1)->float:
        pnt=np.array([pos[0],pos[1],float(z_start)])
        vec=np.array([0,0,-1])
        geomid=np.array([-1],dtype=np.int32)
        while True:
            dist=mujoco.mj_ray(mj_model,mj_data,pnt,vec,None,1,-1,geomid)
            if dist>0:
                hit_z=pnt[2]-dist
                #check that robot wasnt hit
                current_id=mj_model.geom_bodyid[geomid[0]]
                robot=False
                temp_id=current_id
                while temp_id != 0:
                    if temp_id==robot_root_id:
                        robot=True
                        break
                    temp_id=mj_model.body_parentid[temp_id]
                if robot:pnt[2]=hit_z-0.001
                else:return hit_z
            else:
                return 0
    def process_step(self,mj_model,mj_data,hip_pos:np.ndarray,v_current:np.ndarray,v_desired:np.ndarray, k:np.ndarray,
                    s:int,desired_angle_delta:float,angle:float,z_start:float)->tuple[np.ndarray,float]:
                    
        target_pos,target_angle=self.generate_footstep(hip_pos,v_current,v_desired,k,s,desired_angle_delta,angle,self.max_turn)
        target_pos=self.project_kinematics(target_pos,hip_pos,s,self.max_reach,self.min_width)
        #I assume flat for now
        target_z=self.get_terrain_height(mj_model,mj_data,target_pos,z_start)
        target_pos=np.array([target_pos[0],target_pos[1],target_z])
        return target_pos,target_angle

    def nominal_footstep(self,hip_pos:np.ndarray,v_desired:np.ndarray,s:int,
                        desired_angle_delta:float,angle:float,max_turn:float)->tuple[np.ndarray, float]:
        # s=1 or s=-1 for left or right
        x_foot=self.T/2*v_desired[0]
        y_foot=s*self.w/2+self.T/2*v_desired[1]
        target_pos=np.array([x_foot,y_foot])
        #Angle
        target_angle=angle+np.clip(desired_angle_delta,-max_turn,max_turn)
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        target_pos=hip_pos+R@target_pos
        return target_pos,target_angle


    def nominal_footstep_polygons(self,mj_model,mj_data,hip_pos:np.ndarray,v_current:np.ndarray,v_desired:np.ndarray,k:np.ndarray,
                                    s:int,N:int,z_start:float,angle:float,desired_angle_delta:float,max_turn:float)->list:

            polygons=[]
            target_pos,target_angle=self.process_step(mj_model,mj_data,hip_pos,v_current,v_desired,k,s,desired_angle_delta,angle,z_start)
            polygons.append([target_pos[0]-self.sole_length_x/2,target_pos[0]+self.sole_length_x/2,
            target_pos[1]-self.sole_width_y/2,target_pos[1]+self.sole_width_y/2])
            pos=target_pos[:2]
            for i in range(1,N):
                s=-s  
                pos,angle=self.nominal_footstep(pos,v_desired,s,0,angle,max_turn)
                polygons.append([pos[0]-self.sole_length_x/2,pos[0]+self.sole_length_x/2,
                    pos[1]-self.sole_width_y/2,pos[1]+self.sole_width_y/2])
                
            return polygons