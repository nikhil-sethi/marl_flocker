
from models import Actor, Critic
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch
from distributions import OrnsteinUhlenbeckProcess
import numpy as np


class MADDPG:
    def __init__(self, num_obs, num_acts, num_agents, minibatch_size) -> None:
        # understanding from https://www.youtube.com/watch?v=LaIrP-MsPSU

        self.num_agents = num_agents

        self.actors = [Actor(num_obs, num_acts) for _ in range(num_agents)]
        self.critics = [Critic(num_agents*num_acts*num_obs, 1) for _ in range(num_agents)]

        self.actor_targets = deepcopy(self.actors)
        self.critic_targets = deepcopy(self.critics)

        # initialise replay buffer
        self.experiences = ReplayBuffer(1e6, num_acts, num_obs, num_agents)
        self.minibatch_size = minibatch_size

        self.exploration_process = OrnsteinUhlenbeckProcess(num_acts)

    def update(self):
        # for each agent
            # Sample minibatch of experiences
            batch = self.experiences.sample(self.minibatch_size)
            # calc optimal q val

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
