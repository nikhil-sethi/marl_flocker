from maddpg import MADDPG
from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
import numpy as np

if __name__=="__main__":
    num_agents = 2
    
    env = FlockAviary(num_drones=num_agents, gui=True, act=ActionType.VEL)
    num_obs = env.observation_space[0].shape[0]

    # Hyperparameters
    episodes = 10000  # number of training expi
    T = 1000 # maximum steps in episode
    minibatch_size = 1000
    episodes_before_train = 10 # on average 10 episodes to fill the replay buffer

    algo = MADDPG(num_agents=num_agents, num_acts=env.action_space[0].shape[0], num_obs=env.observation_space[0].shape[0], minibatch_size=minibatch_size)
    
    # ====== main process =====
    # Reference: https://arxiv.org/pdf/1509.02971.pdf
    for episode in range(episodes):
        obs_dict = env.reset()

        for t in range(1, T):
            # use the actor to get action for each agent
            
            action_dict = algo.get_action_dict(obs_dict)
            # print(action_dict)
            # Take one step in the world
            obs_dict_next, reward_dict, done_dict, info_dict = env.step(action_dict)
            
            # if episode ends, manually set observations to None
            if t == T-1:
                for i in range(num_agents):
                    for __ in range(num_obs):
                        obs_dict_next[i] = np.full(fill_value = None, shape=(num_obs,))

            # update replay buffer
            algo.experiences.push((obs_dict, action_dict, reward_dict, obs_dict_next))
            

            obs_dict = obs_dict_next

            # update actor + critic + targets
            if episode>episodes_before_train:
                algo.update()

        print("episode done")


