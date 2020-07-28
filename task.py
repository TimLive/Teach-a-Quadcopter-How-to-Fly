import numpy as np
from physics_sim import PhysicsSim
import numpy as np
import copy
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=15., target_pos=None, mode = 0):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 6

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.mode = mode
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
                 
        self.env_bounds = self.sim.upper_bounds - self.sim.lower_bounds
        self.distance = 0
        self.last_pose = copy.copy(self.sim.pose[:3])
        self.distance_min = (((self.sim.pose[0] - self.target_pos[0])**2 +
                         (self.sim.pose[1] - self.target_pos[1])**2 +
                         (self.sim.pose[2] - self.target_pos[2])**2)** 0.5)
        
        
    #mode=0 
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        self.distance = (((self.sim.pose[0] - self.target_pos[0])**2 +
                         (self.sim.pose[1] - self.target_pos[1])**2 +
                         (self.sim.pose[2] - self.target_pos[2])**2)** 0.5)
#         print("distance", self.distance, " from ", self.sim.pose[:3], " to ", self.target_pos)
#         print("distance min", self.distance_min)
#         print("dis", self.distance)
        
        if self.mode==0:
            reward = (self.env_bounds / 2.0 - self.distance).sum() / self.action_repeat
        elif self.mode==1:
            reward = (self.env_bounds / 2 - .3*(abs(self.sim.pose[:3] - self.target_pos))).sum() / self.action_repeat
        elif self.mode == 2:
            reward = self.distance_min - self.distance
        elif self.mode == 3:
            reward = 1.0 - (np.mean(self.distance / self.env_bounds)) ** 0.4  #https://www.youtube.com/watch?v=0R3PnJEisqk
        else:
            reward = (self.env_bounds / 2.0 - self.distance).sum() / self.action_repeat
            
                    
        if self.distance < self.distance_min:
            self.distance_min = self.distance
            
        distance_last = (((self.last_pose[0] - self.target_pos[0])**2 +
                         (self.last_pose[1] - self.target_pos[1])**2 +
                         (self.last_pose[2] - self.target_pos[2])**2)** 0.5)
#         print("last dis", distance_last)
        
#         reward = distance_last - self.distance
        self.last_pose = copy.copy(self.sim.pose[:3])

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
#             pose_all.append(np.hstack((self.sim.pose, self.sim.v, self.sim.angular_v)))
        next_state = np.concatenate(pose_all)        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
#         state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v] * self.action_repeat) 
        return state