
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
import numpy as np

class FlockingEnv(FlockAviary):
    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 2, neighbourhood_radius: float = np.inf, initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, freq: int = 240, aggregate_phy_steps: int = 1, gui=False, record=False, obs: ObservationType = ObservationType.KIN, act: ActionType = ActionType.RPM):
        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, freq, aggregate_phy_steps, gui, record, obs, act)

    def _computeReward(self):
        rewards = {}
        # obs = (self._getDroneStateVector(i))
        states = np.array([self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES)])
        des_state = self.get_desired_state(0)
        rewards[0] = -2 * np.linalg.norm(des_state[10:13] - states[0, 10:13])**2
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