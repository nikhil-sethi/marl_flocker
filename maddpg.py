
from models import Actor, Critic
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch
from distributions import OrnsteinUhlenbeckProcess
import numpy as np


class MADDPG:
    def __init__(self, num_obs, num_acts, num_agents, buffer_size =1e6, minibatch_size=1e3) -> None:
        # understanding from https://www.youtube.com/watch?v=LaIrP-MsPSU

        self.num_agents = num_agents
        self.num_obs = num_obs
        self.num_acts = num_acts

        self.actors = [Actor(num_obs, num_acts) for _ in range(num_agents)]
        
        self.critic_input_size = num_agents*num_acts*num_obs
        self.critics = [Critic(self.critic_input_size, 1) for _ in range(num_agents)]

        self.actor_targets = deepcopy(self.actors)
        self.critic_targets = deepcopy(self.critics)

        # initialise replay buffer
        self.experiences = ReplayBuffer(buffer_size, num_acts, num_obs, num_agents)
        self.minibatch_size = minibatch_size

        self.exploration_process = OrnsteinUhlenbeckProcess(num_acts)

    def update(self):
        for i in range(self.num_agents):
            # Sample minibatch of experiences
            batch = self.experiences.sample(self.minibatch_size)

            # === calc optimal q val ===
            # prepare next states for actor

            
            obs_next_batch = torch.Tensor(batch[:, :, -self.num_obs:].astype(np.double)).double() # shape = minibatch_size x num_agents x num_obs
            # next state actions from the target actors
            a_next_batch = torch.hstack([self.actor_targets[k](obs_next_batch[:,k,:]) for k in range(self.num_agents)]) # shape = minibatch_size x num_acts
            # print(obs_next_batch.flatten(start_dim=1))
            # 
            # q_next_ = self.critics[i]()

            # update critic

            # update actor

            # update targets


    def get_action_dict(self, obs_dict):
        assert len(obs_dict) == len(self.actors), "csdfg"
        
        # feedforward observation to actor
        with torch.no_grad():
            return {i:self.get_action(i, obs_dict[i]) for i in range(self.num_agents)}
    

    def get_action(self, i, obs:np.ndarray) -> np.ndarray:
        
        return self.actors[i](torch.from_numpy(obs)).numpy()
    

if __name__=="__main__":

    # unit test maddpg update
    minibatch_size = 5
    num_agents = 2
    num_acts = 3
    num_obs = 4
    size = 10
    algo = MADDPG(num_agents=num_agents, num_acts=num_acts, num_obs=num_obs,buffer_size=size, minibatch_size=minibatch_size)

    for _ in range(size):
        obs_dict = {i:np.random.rand(num_obs) for i in range(num_agents)}
        act_dict = {i:np.random.rand(num_acts) for i in range(num_agents)}
        obs_dict_next = {i:np.random.rand(num_obs) for i in range(num_agents)}
        reward_dict = {i:np.random.rand(1)[0] for i in range(num_agents)}

        algo.experiences.push((obs_dict, act_dict, reward_dict, obs_dict_next))

    algo.update()

    
