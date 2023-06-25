from maddpg import MADDPG
from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
from environment import FlockingEnv
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
import numpy as np
import os
from subprocess import Popen, PIPE
import time
import torch

TRAIN = False
DEBUG = True

np.set_printoptions(precision=3, suppress=True)


if DEBUG:
    PIPE_PATH0 = "/tmp/my_pipe0"
    PIPE_PATH1 = "/tmp/my_pipe1"

    if not os.path.exists(PIPE_PATH0):
        os.mkfifo(PIPE_PATH0)

    if not os.path.exists(PIPE_PATH1):
        os.mkfifo(PIPE_PATH1)

    term0 = Popen(['xterm', '-e', 'tail -f %s' % PIPE_PATH0])
    term1 = Popen(['xterm', '-e', 'tail -f %s' % PIPE_PATH1])




if __name__=="__main__":
    try:
        num_agents = 1
        
        env = FlockingEnv(num_drones=num_agents, gui=DEBUG, act=ActionType.VEL)
        num_obs = env.observation_space[0].shape[0]

        # Hyperparameters
        episodes = 10000  # number of training expi
        T = 2000 # maximum steps in episode
        minibatch_size = 1000
        episodes_before_train = 60 # on average 10 episodes to fill the replay buffer
        history = {"reward":[], 0:{"reward":0}, 1:{"reward":0}}
        
        algo = MADDPG(TRAIN, num_agents=num_agents, num_acts=env.action_space[0].shape[0], num_obs=env.observation_space[0].shape[0], minibatch_size=minibatch_size, history=history, ep_before_train = episodes_before_train)
        total_reward = 0
        
        # ====== main process =====
        # Reference: https://arxiv.org/pdf/1509.02971.pdf
        for episode in range(episodes):
            obs_dict = env.reset()

            for t in range(1, T):
                # use the actor to get action for each agent
                
                action_dict = algo.get_action_dict(obs_dict)
                # print(obs_dict[0][7:10])
                action_dict[0][2]=0
                # action_dict[1][2]=0
                # action_dict = {0: np.array([0, 1, 0., 2.])}
                # print(action_dict)
                # Take one step in the world
                obs_dict_next, reward_dict, done_dict, info_dict = env.step(action_dict)

                if t%1==0 and DEBUG:
                    with open(PIPE_PATH0, "w") as p:
                        p.write(f"pos: {obs_dict[0][0:3]} | vel: {obs_dict[0][6:9]} | act: {action_dict[0]} \n")
                        # time.sleep(1)\
                    # with open(PIPE_PATH1, "w") as p:
                    #     p.write(f"pos: {obs_dict[1][0:3]} | vel: {obs_dict[1][6:9]} | act: {action_dict[1]} \n")

                history[0]["reward"] += reward_dict[0]
                # history[1]["reward"] += reward_dict[1]
                history["reward"].append(sum(reward_dict.values())/num_agents)
                # if episode ends, manually set observations to None
                if t == T-1:
                    for i in range(num_agents):
                        for __ in range(num_obs):
                            obs_dict_next[i] = np.full(fill_value = None, shape=(num_obs,))

                # update replay buffer
                algo.experiences.push((obs_dict, action_dict, reward_dict, obs_dict_next))
                

                obs_dict = obs_dict_next

                # update actor + critic + targets
                if episode>episodes_before_train and TRAIN:
                    act_loss, q_loss = algo.update()
                    # history[0]["act_loss"] += act_loss[0]
                    if act_loss and DEBUG:
                        print(act_loss[0].data, q_loss[0].data)
                    if episode % 10 ==0:
                        for i,model in enumerate(algo.actors):
                            torch.save(model, f'actor_{i}')
                algo.episode = episode
            print(f"Episode {episode} done | reward: {history['reward'][-1]:0.5f} | Avg reward: {sum(history['reward'])/len(history['reward']):0.5f}")
    
    
    except:
        if DEBUG:
            term0.kill()
            term1.kill()
        if TRAIN:
            for i,model in enumerate(algo.actors):
                torch.save(model, f'actor_{i}')
        raise
