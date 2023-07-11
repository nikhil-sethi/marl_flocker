
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import gym
from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
import numpy as np
import pygame as pg
import math
import pybullet as p

def v_cap(v):
    return v/np.linalg.norm(v)

def norm2d(v):
    return math.sqrt(v[0]**2+v[1]**2)

class FlockingEnv(FlockAviary):
    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 2, neighbourhood_radius: float = np.inf, initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, freq: int = 240, aggregate_phy_steps: int = 1, gui=False, record=False, obs: ObservationType = ObservationType.KIN, act: ActionType = ActionType.RPM):
        self.obs_idx=[]
        self.closest_obs_dist = {i:0 for i in range(num_drones)} # 0 us just an initial placeholder

        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, freq, aggregate_phy_steps, gui, record, obs, act)
        self.v_old= np.zeros((self.NUM_DRONES, 2))
        
        
        # self._addObstacles()

    def _observationSpace(self):
        # add closest distance from obstacle to observation
        return gym.spaces.Dict({i: gym.spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0]),
                                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 0.5]),
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})
    

    def _addObstacles(self):
        obs_id = p.loadURDF("/home/nikhil/Nikhil/Masters/Courses/ae4350_bil/obstacles/wall.urdf",
                    [0, 0.75, .15],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                    )
        self.obs_idx.append(obs_id)

        obs_id = p.loadURDF("/home/nikhil/Nikhil/Masters/Courses/ae4350_bil/obstacles/wall.urdf",
                    [-0.75, 0, .15],
                    p.getQuaternionFromEuler([0, 0, 1.57]),
                    physicsClientId=self.CLIENT
                    )
        self.obs_idx.append(obs_id)

        obs_id = p.loadURDF("/home/nikhil/Nikhil/Masters/Courses/ae4350_bil/obstacles/wall.urdf",
                    [0, -0.75, .15],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                    )
        self.obs_idx.append(obs_id)

        obs_id = p.loadURDF("/home/nikhil/Nikhil/Masters/Courses/ae4350_bil/obstacles/wall.urdf",
                    [0.75, 0, .15],
                    p.getQuaternionFromEuler([0, 0, 1.57]),
                    physicsClientId=self.CLIENT
                    )
        self.obs_idx.append(obs_id)
        
        # return super()._addObstacles()


    def reset(self):
        # self.INIT_RPYS
        # self.INIT_XYZS = -0.3 + 0.6*np.random.rand(self.NUM_DRONES,3)
        self.INIT_XYZS = np.array([
            [0,0,0.1],
            [0.5,0.5,0.1]
        ])
        
        # self.INIT_XYZS[:,-1]=0.1
        # self.vel = np.array([[1,1,0],[-1,-1,0]])
        obs = super().reset()
        return obs
    
    def step(self, actions):
        # actions[0][-1]*=3
        action_dict = {}
        for i, action in actions.items():
            # action = actions[0]
            # print(actions[0][-1])
            # action = np.random.rand(4)
            # actions[0][2] = 0
            vx = action[0] - action[1]
            vy = action[2] - action[3]
            # v_mag = action[4]
            # if abs(v_mag)>3:
            #     v_mag=3
            action_dict[i] = np.array([vx, vy, 0, 1])
            # if i ==0:
            #     action_dict[i] += np.array([1, 1, 0, 1])
            # elif i ==1:
            #     action_dict[i] += np.array([-1, -1, 0, 1])
            # print(action_dict[i])
        # print(actions)



        return super().step(action_dict)
         

    def _computeReward(self):
        rewards = {}
        # obs = (self._getDroneStateVector(i))
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        for i in range(self.NUM_DRONES):
            r_sep = 0
            r_align = 0
            r_cohere = 0
            r_v_mag = 0
            r_acc = 0
            r_wall = 0
            rewards[i] = 0

            # Motion reward: smooth motions
            
            # don't stay at rest
            v_curr = states[i, 10:12]

            v_mag = norm2d(v_curr)
            if v_mag<0.5:
                r_v_mag += v_mag - 0.5
            
            r_acc = -(norm2d(self.v_old[i]) - v_mag)

            r_motion = 2 * r_v_mag + r_acc
            
            # Stability: minimize roll and pitch angle magnitudes
            r_stab = -1 * sum(np.abs(states[i, 7:9]))

            pos_i = states[i, 0:2]
            if i==0:
                r_target = -np.linalg.norm(np.array([1,1])- pos_i)**2
            elif i==1:
                r_target = -np.linalg.norm(np.array([0,0])- pos_i)**2

            for j in range(self.NUM_DRONES):
                if j==i:
                    continue
                pos_j = states[j, 0:2]
                dist = np.linalg.norm(pos_j - pos_i)
                # print(dist)
                if dist<0.2:
                    r_sep += -5
                # elif 0.2< dist < 0.4:
                r_align += -4 * np.linalg.norm(states[j,10:12] - states[i, 10:12])
                # elif dist > 0.4:
                #     r_cohere += -dist
                r_cohere += -2*dist

                # target waypoint
                
            for k in self.obs_idx:
                closest_dist = 100
                closest_obs_pt = p.getClosestPoints(bodyA=i+1,
                    bodyB=k,
                    distance = 10, # so that we always return a distance but rewards are shaped accordingly
                    physicsClientId=self.CLIENT
                    )
                dist = closest_obs_pt[0][8]
                if dist <closest_dist:
                    closest_dist = dist
            if closest_dist <=0:
                r_wall += -2
            if 0<closest_dist < 0.3:
                r_wall+= 2*(closest_dist**0.3 - 0.3**0.3) # closer you are to the wall, more negative it is

            self.closest_obs_dist[i] = closest_dist
                # if len(obs_k_closest)>0:
                #     print(i, )
            # print(coll_pts)

            # if self.collision:
            #     rewards[i]+= -10
            rewards[i] += r_align 
            rewards[i] += r_sep
            # rewards[i] += r_target
            rewards[i] += r_wall
            # rewards[i] += r_cohere
            # rewards[i] += r_motion
            rewards[i] += r_stab
            self.v_old[i] = v_curr
            # print(f"Rewards {i}: r_align: {r_align} | r_sep: {r_sep} | r_cohere: {r_cohere} r_motion: {r_motion} r_stab: {r_stab} | r_target: {r_target} | r_wall: {r_wall}")

        return rewards
    

    def _computeObs(self):
        orig_obs = super()._computeObs()

        # add distance to closest wall
        for i in range(self.NUM_DRONES):
            orig_obs[i] = np.append(orig_obs[i], self.closest_obs_dist[i])
        
        return orig_obs

    def _clipAndNormalizeStateWarning(self, state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        pass

    def get_desired_state(self, i):
        state = np.zeros(20)
        state[0:3] =  np.array([0.1,0.5,0.1])
        state[10:13] = self.SPEED_LIMIT*2*v_cap(np.array([2,1,0]))
        return self._clipAndNormalizeState(state)
    

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