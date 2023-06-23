from maddpg import MADDPG
from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

if __name__=="__main__":
    num_agents = 2
    env = FlockAviary(num_drones=num_agents, gui=True, act=ActionType.VEL)
    algo = MADDPG(num_agents=num_agents, num_acts=env.action_space[0].shape[0], num_obs=env.observation_space[0].shape[0])


    # ====== main process =====
    # Reference: https://arxiv.org/pdf/1509.02971.pdf


    # Hyperparameters
    episodes = 10000  # number of training expi
    T = 2000 # maximum steps in episode
    for _ in range(episodes):
        obs_dict = env.reset()

        for t in range(1, T):
            # use the actor to get action for each agent
            
            action_dict = algo.get_action_dict(obs_dict)
            print(action_dict)
            # Take one step in the world
            obs_dict_next, reward_dict, done_dict, info_dict = env.step(action_dict)


            # update replay buffer
            algo.experiences.push((obs_dict, action_dict, reward_dict, obs_dict_next))

            obs_dict = obs_dict_next

            # update actor + critic + targets
            algo.update()


