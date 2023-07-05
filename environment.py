
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import gym
from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
import numpy as np
import pygame as pg
import math

class FlockingEnv(FlockAviary):
    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 2, neighbourhood_radius: float = np.inf, initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, freq: int = 240, aggregate_phy_steps: int = 1, gui=False, record=False, obs: ObservationType = ObservationType.KIN, act: ActionType = ActionType.RPM):
        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, freq, aggregate_phy_steps, gui, record, obs, act)

    def step(self, actions):
        # actions[0][-1]*=3
        
        action = actions[0]
        # print(actions[0][-1])
        # action = np.random.rand(4)
        # actions[0][2] = 0
        vx = action[0] - action[1]
        vy = action[2] - action[3]
        
        action = {0:np.array([vx, vy, 0, 2])}
            
        return super().step(action)
         

    def _computeReward(self):
        rewards = {}
        # obs = (self._getDroneStateVector(i))
        states = np.array([self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES)])
        des_state = self.get_desired_state(0)
        # print(des_state)
        rewards[0] = -2 * np.linalg.norm(des_state[10:13] - states[0, 10:13])/2
        # rewards[0] = -2 * np.linalg.norm(np.array([0.1,0.05,0]) - states[0, 0:3])**2
        # print(rewards[0])
        rewards[0] += -1 * sum(np.abs(states[0, 7:9]))
        # rewards[1] = -2 * np.linalg.norm(np.array([0.5,0.5,0]) - states[1, 0:3])**2
        # rewards[1]=0
        # for i in range(1, self.NUM_DRONES):
        #     rewards[i] = -1 * np.linalg.norm(states[i-1, 2] - states[i, 2])**2
        # print(f"reward 0: {rewards[0]:0.3f} | reward 1: {rewards[1]:0.3f} ")
        return rewards
    
    def _clipAndNormalizeStateWarning(self, state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        pass

    def get_desired_state(self, i):
        state = np.zeros(20)
        state[10:13] = self.SPEED_LIMIT*2*np.array([0,1,0])
        return self._clipAndNormalizeState(state)
    


def v_cap(v):
    return v/np.linalg.norm(v)

class DebugEnv(gym.Env):
    def __init__(self, gui=False):
        super().__init__()

        self.screen_width = 400
        self.screen_height = 400

        self.v_max = 8 # pixels

        # Define observation space
        self.observation_space = [gym.spaces.Box(low=np.array([0,0,-1,1]), high=np.array([self.screen_width,self.screen_height,2,2]), shape=(4,), dtype=float)]

        # Define action space
        # self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)]
        self.action_space = [gym.spaces.Discrete(4)]

        # Initialize agent's state
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.pg_scale = 20
        self.gui = gui
        if self.gui:
            pg.init()
            self.screen = pg.display.set_mode((self.screen_width, self.screen_height))  

    def observation(self):
        return {0:np.array([self.agent_x, self.agent_y, self.agent_vx, self.agent_vy])}
    
    def step(self, action_dict):
        # Update agent's state based on the action
        done = False
        # print(action_dict)
        action = action_dict[0]
        accel = 5

        acc_x = (action[0] - action[1])*accel
        acc_y = (action[2] - action[3])*accel
        
        # step the newton euler thing
        dt = 0.3
        # acc_x = des_vx - self.agent_vx
        # acc_y = des_vy - self.agent_vy
        self.agent_vx += acc_x*dt
        self.agent_vy += acc_y*dt

        mag = math.sqrt(self.agent_vx**2 + self.agent_vy**2)
        if mag >self.v_max:
            self.agent_vx *= self.v_max/mag
            self.agent_vy *= self.v_max/mag

        self.agent_x += self.agent_vx*dt + 0.5*acc_x*dt**2
        self.agent_y += self.agent_vy*dt + 0.5*acc_y*dt**2
        # Clip agent's position within the screen bounds
        # self.agent_x = max(min(self.agent_x, self.screen_width), 0)
        # self.agent_y = max(min(self.agent_y, self.screen_height), 0)

        if self.agent_x < 0 or self.agent_x>self.screen_width or self.agent_y<0 or self.agent_y >self.screen_height:
            done = True

        # Compute reward, done, and info
        reward = 0.0
        # done = False
        info = {}
        return self.observation(), self.reward(), done, info

    def reward(self):
        # des_vx = 
        des_v = v_cap(np.array([1,0.5]))
        return {0: -np.linalg.norm(des_v - v_cap(np.array(self.agent_vx, self.agent_vy)))}

    def reset(self):
        # Reset agent's state to initial values
        self.agent_x = self.screen_width/2
        self.agent_y = self.screen_height/2
        self.agent_vx = 0.0
        self.agent_vy = 0.0

        return self.observation()

    def render(self, mode='human'):
        if mode == 'human':
            for event in pg.event.get():
                if event.type==pg.QUIT:
                    exit(0)
            pg.display.set_caption('My Environment')
            self.screen.fill((255, 255, 255))

            # Draw agent
            pg.draw.circle(self.screen, (0, 0, 255), (int(self.agent_x), int(self.agent_y)), 10)

            pg.display.flip()
        else:
            super().render(mode)

    def close(self):
        pg.quit()

    def sample_action(self):
        return {0:env.action_space[0].sample()}

if __name__=="__main__":
    gui = True
    env = DebugEnv(gui=gui)
    print(env.observation_space[0].shape[0])
    for episode in range(40):
        _ = env.reset()
        for step in range(100):
            action = {0:np.array([0,1])}
            # action = env.sample_action()
            # print(action)
            obs, reward, done, info = env.step(action)
            # print(reward[0])
            if gui:
                env.render()
            if done:
                break
        # time.sleep()