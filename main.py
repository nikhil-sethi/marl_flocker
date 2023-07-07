from maddpg import MADDPG
from gym_pybullet_drones.envs.multi_agent_rl import FlockAviary
from environment import FlockingEnv, DebugEnv
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
import numpy as np
import os
from subprocess import Popen, PIPE
import time
import torch
import argparse
from replay_buffer import ReplayBuffer
import pybullet as p


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # cpu is apparently faster in this case

np.set_printoptions(precision=3, suppress=True)
seed = 0
# np.random.seed(seed)
# torch.manual_seed(seed)

if __name__=="__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    args = parser.parse_args()

    TRAIN = args.train
    try:
        num_agents = 2
        # initial_xyz = np.array([[1,1,1]])
        env = FlockingEnv(num_drones=num_agents, gui=not TRAIN, act=ActionType.VEL, freq=80)
        # env = DebugEnv(gui= not TRAIN)
        num_obs = env.observation_space[0].shape[0]
        num_acts = env.action_space[0].shape[0]
        # Hyperparameters
        episodes = 10000  # number of training expi
        max_steps = 200 # maximum steps in episode
        minibatch_size = 1024
        episodes_before_train = 50 # on average 10 episodes to fill the replay buffer
        history = {"reward":[], 0:{"reward":0, "act_loss":[], "q_loss":[]}, 1:{"reward":0}}
        
        algo = MADDPG(TRAIN, num_agents=num_agents, num_acts=num_acts, num_obs=num_obs, minibatch_size=minibatch_size, history=history, ep_before_train = episodes_before_train, device = device)
        
        

        experiences = ReplayBuffer(1e6, batch_size=minibatch_size, num_acts=num_acts, num_obs=num_obs, num_agents=num_agents)
        
        total_reward = 0
        best_reward = -np.inf
        total_steps = 0
        # ====== main process =====
        # Reference: https://arxiv.org/pdf/1509.02971.pdf
        for episode in range(episodes):
            obs_dict = env.reset()
            # id = p.addUserDebugPoints(pointPositions = [[0.1,0.5,0.1]], pointColorsRGB = [[0,0,0]], pointSize=5)
            reward_ep = 0
            act_loss_ep = 0
            q_loss_ep = 0
            start = time.perf_counter()
            dones = np.zeros((num_agents,1), dtype=bool)
            episode_steps = 0
            
            while not any(dones):
                # use the actor to get action for each agent
                action_dict = algo.get_action_dict(obs_dict)
                # print(obs_dict[0][7:10])
                # action_dict[0][2]=0
                # action_dict[1][2]=0
                # action_dict = {0: np.array([0, 1])}
                # print(obs_dict[0][-2:])
                # Take one step in the world
                # t1 = time.perf_counter()
                # print(episode, action_dict, obs_dict)
                # time.sleep(1)
                obs_dict_next, reward_dict, done_dict, info_dict = env.step(action_dict)
                if not TRAIN:
                    time.sleep(0.05)
                dones = np.array(list(done_dict.values())[:-1])[:,np.newaxis]

                history[0]["reward"] += reward_dict[0]
                # history[1]["reward"] += reward_dict[1]
                reward_ep += sum(reward_dict.values())/num_agents
                
                # if episode ends, manually set observations to None
                if episode_steps >= max_steps:
                    dones = np.ones((num_agents,1), dtype=bool)

                # update replay buffer           
                experiences.push((obs_dict, action_dict, reward_dict, obs_dict_next, dones))
                

                obs_dict = obs_dict_next

                # update actor + critic + targets
                if total_steps % 100==0:
                    if experiences.ready() and TRAIN and episode>episodes_before_train:
                    
                        act_loss, q_loss = algo.update(experiences)
                        # history[0]["act_loss"] += act_loss[0]
                        if act_loss:
                            act_loss_ep += act_loss[0]
                            q_loss_ep += q_loss[0]

                        if avg_reward > best_reward:
                            for i,model in enumerate(algo.actors):
                                print("Saving model...")
                                torch.save(model, f'actor_{i}')
                            best_reward = avg_reward
                
                total_steps +=1
                episode_steps += 1
                algo.episode = episode
            history["reward"].append(reward_ep)
            
            history[0]["act_loss"].append(act_loss_ep)
            history[0]["q_loss"].append(q_loss_ep)

            avg_reward = np.mean(history['reward'][-100:])
            if episode%20 ==0:
                print(f"Episode {episode} done | reward: {reward_ep:0.5f} | Avg reward: {avg_reward:0.5f} | act loss: {sum(history[0]['act_loss'])/len(history['reward']):0.5f} | q loss: {sum(history[0]['q_loss'])/len(history['reward']):0.5f} ")
            
    
    
    except:
        if not TRAIN:
            term0.kill()
            term1.kill()
        else:
            for i,model in enumerate(algo.actors):
                torch.save(model, f'actor_{i}')
        raise
