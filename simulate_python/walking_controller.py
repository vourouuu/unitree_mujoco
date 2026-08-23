import numpy as np
import mujoco

import numpy as np

class PDControllerBrain:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.num_dof = 29
        
        # testing tuning table
        # Format: [q_target, Kp, Kd]
        tuning_parameters = [
            # left leg(0-5)
            [ -0.3,  100.0,  5.0 ],  # 0: left_hip_pitch
            [  0.0,  100.0,  5.0 ],  # 1: left_hip_roll
            [  0.0,  100.0,  5.0 ],  # 2: left_hip_yaw
            [  0.6,  100.0,  5.0 ],  # 3: left_knee
            [ -0.3,   40.0,  1.0 ],  # 4: left_ankle_pitch
            [  0.0,   40.0,  1.0 ],  # 5: left_ankle_roll
            
            # right leg(6-11)
            [ -0.3,  100.0,  5.0 ],  # 6: right_hip_pitch
            [  0.0,  100.0,  5.0 ],  # 7: right_hip_roll
            [  0.0,  100.0,  5.0 ],  # 8: right_hip_yaw
            [  0.6,  100.0,  5.0 ],  # 9: right_knee
            [ -0.3,   40.0,  1.0 ],  # 10: right_ankle_pitch
            [  0.0,   40.0,  1.0 ],  # 11: right_ankle_roll
            
            # waist(12-14)
            [  0.0,   5.0,  0.5 ],  # 12: waist_yaw
            [  0.0,   5.0,  0.5 ],  # 13: waist_roll
            [  0.0,   5.0,  0.5 ],  # 14: waist_pitch
            
            # left arm(15-21)
            [  0.2,   2.0,  0.1 ],  # 15: left_shoulder_pitch
            [  0.0,   2.0,  0.1 ],  # 16: left_shoulder_roll
            [  0.0,   2.0,  0.1 ],  # 17: left_shoulder_yaw
            [  0.0,   2.0,  0.1 ],  # 18: left_elbow
            [  0.0,   1.0,  0.05],  # 19: left_wrist_roll
            [  0.0,   1.0,  0.05],  # 20: left_wrist_pitch
            [  0.0,   1.0,  0.05],  # 21: left_wrist_yaw
            
            # right arm(22-28)
            [  0.2,   2.0,  0.1 ],  # 22: right_shoulder_pitch
            [  0.0,   2.0,  0.1 ],  # 23: right_shoulder_roll
            [  0.0,   2.0,  0.1 ],  # 24: right_shoulder_yaw
            [  0.0,   2.0,  0.1 ],  # 25: right_elbow
            [  0.0,   1.0,  0.05],  # 26: right_wrist_roll
            [  0.0,   1.0,  0.05],  # 27: right_wrist_pitch
            [  0.0,   1.0,  0.05]   # 28: right_wrist_yaw
        ]
        
        # slice the table into vectorized NumPy arrays
        tuning_array = np.array(tuning_parameters)
        
        self.q_target = tuning_array[:, 0]  # Column 0: Targets
        self.kp       = tuning_array[:, 1]  # Column 1: Kp gains
        self.kd       = tuning_array[:, 2]  # Column 2: Kd gains

    def think(self):
        q_current = self.data.sensordata[0:29]
        qdot_current = self.data.sensordata[29:58]
        
        tau = self.kp * (self.q_target - q_current) - self.kd * qdot_current
        self.data.ctrl[:] = tau


class FootstepPlanner:
    def generate_footsteps(self,distance, step_length, foot_spread=0.1185):
        contacts = []

        contacts.append([0., +foot_spread])
        contacts.append([0., -foot_spread])
        x = 0
        y = foot_spread
        while x < distance:
            if distance - x <= step_length:
                x += min(distance - x, 0.5 * step_length)
            else:  # arent close to the end
                x += step_length
            y = -y

            contacts.append([x, y])
        contacts.append([x, -y])  # now x == distance
        return contacts

    def __init__(self, distance=1.0, step_length=0.25, foot_spread=0.1185):
        self.footsteps = self.generate_footsteps(distance, step_length, foot_spread)



# ai helped with this.
    def draw(self, model):
        for i, step in enumerate(self.footsteps):
            site_name = f"step_{i}"
            
            # Find the site ID by its name
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            
            # If the site exists in the XML, move it to the floor
            if site_id != -1:
                model.site_pos[site_id] = [step[0], step[1], 0.001] 


class WalkingFSM:

    def __init__(self, footsteps, ssp_duration=1.6, dsp_duration=0.4, dt= 0.005):
        self.footsteps = footsteps
        self.next_footstep = 0 # determines if left (even) or right (odd) foot is swinging


        self.ssp_duration = ssp_duration
        self.dsp_duration = dsp_duration
        self.rem_time = 0.0
        

        #
        # self.start_standing()
        self.start_walking = True
        self.state = "Standing"

        self.planner = DCMPlanner(dt=dt)
    
        # State variables to track references
        self.ref_dcm = np.zeros(2)
        self.ref_com = np.zeros(2)

    def tick(self, dt=0.005):

        if self.rem_time > 0:
            self.rem_time -= dt

        if self.state == "Standing":
            return self.run_standing()
        elif self.state == "DoubleSupport":
            return self.run_double_support()
        elif self.state == "SingleSupport":
            return self.run_single_support()
        raise Exception("Unknown state: " + self.state)

# standing code
    def start_standing(self):
        self.start_walking = False
        self.state = "Standing"
        return self.run_standing()

    def run_standing(self):
        if self.start_walking:
            self.start_walking = False
            if self.next_footstep < len(self.footsteps):
                return self.start_double_support()

# double support code
    def start_double_support(self):
        print(f"entered DSP. Next step: {self.next_footstep}")
        
        #Double support is longer for the first and very last steps to give time for the robot to land and stay
        if self.next_footstep == 0 or self.next_footstep == len(self.footsteps) - 1:
            self.rem_time = 4 * self.dsp_duration
        else:
            self.rem_time = self.dsp_duration
            # find the dsp boundaries and trajectories
        
        

        self.state = "DoubleSupport"
        #TODO: initialize Com shift trajectory NOT NEEDED APPARENTLY
        # self.swing_target = self.footsteps[self.next_footstep]
        # self.start_com_mpc_dsp()
        # return self.run_double_support()

    def run_double_support(self):
        if self.rem_time <= 0.:
            return self.start_single_support()
        else:
            pass
            # go from start dcm to end dcm. see equations 25-28 dcm paper


#single support code
    def start_single_support(self):
        print("entered ssp.")
        is_left_swing = (self.next_footstep % 2 == 1)

        self.next_footstep += 1
        self.rem_time = self.ssp_duration
        self.state = "SingleSupport"

        # initialize swing foot trajectory
        # at least thats what happened in scaron's code. i dont think i need it with dcm

    def run_single_support(self):
        if self.rem_time <= 0.:
            if self.next_footstep < len(self.footsteps):
                return self.start_double_support()
            else:  # footstep sequence is over
                print("Footstep sequence complete. Returning to Standing.")
                self.state = "Standing"
        else:
            stance_idx = self.next_footstep - 1
            stance_pos = np.array(self.footsteps[stance_idx])
            
            self.ref_dcm = self.planner.compute_next_dcm(self.ref_dcm, stance_pos) # might need to find where it landed exactly somehow through mujoco
            self.ref_com = self.planner.compute_next_com(self.ref_com, self.ref_dcm)

            #send info to conttroller to inverse the kinematics :)

import numpy as np

class DCMPlanner:
    def __init__(self, height=0.793, dt=0.005):
        # pelvis height from slx, might be raized up bu 0.1 because robot starts in the air
        self.height = height
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.height)
        self.dt = dt
        
        # constants so why not precompute them
        self.alpha = np.exp(self.omega * self.dt)
        self.beta = np.exp(-self.omega * self.dt) 
        
    def compute_next_dcm(self, current_dcm, stable_foot_pos):

        # set the stable foot
        p_zmp = np.array(stable_foot_pos)
        
        #xatz kef 10 sel 36
        next_dcm = current_dcm * self.alpha + (1-self.alpha) * p_zmp
        return next_dcm

    def compute_next_com(self, current_com, current_dcm):
        next_com = self.beta * current_com + (1.0 - self.beta) * current_dcm
        return next_com