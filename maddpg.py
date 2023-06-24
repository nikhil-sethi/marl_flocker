
from models import Actor, Critic
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch
from distributions import OrnsteinUhlenbeckProcess
import numpy as np


class MADDPG:
    def __init__(self, num_obs, num_acts, num_agents, buffer_size =1e6, minibatch_size=1e3) -> None:
        

        self.num_agents = num_agents
        self.num_obs = num_obs
        self.num_acts = num_acts

        self.actors = [Actor(num_obs, num_acts) for _ in range(num_agents)]
        
        self.critic_input_size = num_agents*(num_acts + num_obs)
        self.critics = [Critic(self.critic_input_size, 1) for _ in range(num_agents)]
        self.critic_loss_fn = torch.nn.MSELoss()
        
        self.actor_targets = deepcopy(self.actors)
        self.critic_targets = deepcopy(self.critics)

        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=0.001) for critic in self.critics]

        # initialise replay buffer
        self.experiences = ReplayBuffer(buffer_size, num_acts, num_obs, num_agents)
        self.minibatch_size = minibatch_size

        self.exploration_process = OrnsteinUhlenbeckProcess(num_acts)
        self.gamma = 0.99
        

    def update(self):
        # understanding from https://www.youtube.com/watch?v=LaIrP-MsPSU + https://arxiv.org/pdf/1706.02275.pdf
        for i in range(self.num_agents):
            # Sample minibatch of experiences
            batch = self.experiences.sample(self.minibatch_size).astype(np.double)
            if np.all(batch==None):
                print("batch empty")
                return
            # mask for episode end
            done_mask = (batch[:,i,-self.num_obs+1] == None) # HACK: did only for one agent and one future state observation because the done mask will be same for all anyways.
            
            # batch = batch[~done_mask, ...] # only keep episodes which are not terminal

            # ====== preprocess batch for networks ========
            
            obs_batch = torch.Tensor(batch[..., :self.num_obs]) # shape = minibatch_size x num_agents x num_obs
            act_batch = torch.Tensor(batch[..., self.num_obs:self.num_obs+self.num_acts]).double() # shape = minibatch_size x num_agents x num_acts
            obs_next_batch = torch.Tensor(batch[~done_mask, :, -self.num_obs:]).double() # shape = minibatch_size x num_agents x num_obs
            reward_batch = torch.Tensor(batch[~done_mask, i, self.num_obs + self.num_acts])

            # ======= calc optimal q val ========
                # next state actions from the target actors
            act_next_batch = torch.hstack([self.actor_targets[k](obs_next_batch[:,k,:]) for k in range(self.num_agents)]) # shape = minibatch_size x num_acts
            
                # The future q value depends on all next states and actions
            q_next = self.critic_targets[i](torch.hstack([obs_next_batch.flatten(start_dim=1), act_next_batch]))

                # for the last step in the episode, the target q is 0
            q_target = torch.zeros(self.minibatch_size, 1).double()
            
            q_target[~done_mask] = reward_batch.reshape(-1,1) + self.gamma*q_next # shape= batch_size x 1
            
            # === calulate critic loss ===
                # just the current observation forwarded through the critic
            q_current = self.critics[i](torch.hstack([obs_batch.flatten(start_dim=1), act_batch.flatten(start_dim=1)]))
            q_current[torch.isnan(q_current)] = 0    # this is sketchy. might cause troubles in the future

            q_loss = torch.sqrt(self.critic_loss_fn(q_target, q_current)) # scalar

            # ====== critic backpropagate and optimize =========
            self.critic_optimizers[i].zero_grad()
            q_loss.backward()
            self.critic_optimizers[i].step()
            
            # ===== calculate policy gradient =====

                # get current action from current state 
            act_i = self.actors[i](obs_batch[:, i, :]) 

                # intersperse this action into the batch (i.e. assuming all other actors are constant => you choose only your own destiny)
            act_batch[:, i, :] = act_i    

                # let's see how good this idea is i.e. ask the omniscient all powerful oracle: the centralised critic
            q_act = self.critics[i](torch.hstack([obs_batch.flatten(start_dim=1), act_batch.flatten(start_dim=1)]))
            
            act_loss = -torch.mean(q_act) # the negative sign is just to make sure it's gradient ascent

            # ===== actor backpropagate and optimize =====
            self.actor_optimizers[i].zero_grad()
            act_loss.backward()
            self.actor_optimizers[i].step()

            # soft update targets
            
            self.soft_update_targets()

    def soft_update(target, source, t):
        # courtesy of github.com/xuehy/pytorch-maddpg.git
    
        for target_param, source_param in zip(target.parameters(),
                                            source.parameters()):
            target_param.data.copy_(
                (1 - t) * target_param.data + t * source_param.data)


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

    for j in range(size):
        obs_dict = {i:np.random.rand(num_obs) for i in range(num_agents)}
        act_dict = {i:np.random.rand(num_acts) for i in range(num_agents)}
        obs_dict_next = {i:np.random.rand(num_obs) for i in range(num_agents)}
        reward_dict = {i:np.random.rand(1)[0] for i in range(num_agents)}

        if j == 2:
            obs_dict_next = {i:np.full(fill_value=None, shape=(num_obs,)) for i in range(num_agents)}

        algo.experiences.push((obs_dict, act_dict, reward_dict, obs_dict_next))

    algo.update()

    
