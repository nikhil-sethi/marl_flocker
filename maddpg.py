
from models import Actor, Critic
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch
from distributions import OrnsteinUhlenbeckProcess
import numpy as np


class MADDPG:
    def __init__(self, train, num_obs, num_acts, num_agents, buffer_size =1e6, minibatch_size=1e3, history={}, ep_before_train = 100, device = torch.device('cpu')) -> None:
        
        self.episodes_before_train = ep_before_train
        self.history = history
        self.num_agents = num_agents
        self.num_obs = num_obs
        self.num_acts = num_acts
        self.train = train
        self.actors = []
        self.device = device

        for i in range(num_agents):
            if train:
                self.actors.append(Actor(num_obs, num_acts, device=device))
            else:
                model = torch.load(f'actor_{i}')
                model.to(device)
                model.eval()
                self.actors.append(model)
        # self.actors = [ if tfor _ in range(num_agents)]
        
        self.critic_input_size = num_agents*(num_acts + num_obs)
        self.critics = [Critic(self.critic_input_size, 1, device=device) for _ in range(num_agents)]
        self.critic_loss_fn = torch.nn.functional.mse_loss
        
        self.actor_targets = deepcopy(self.actors)
        self.critic_targets = deepcopy(self.critics)

        # self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]
        # self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=0.001) for critic in self.critics]

        # initialise replay buffer
        self.experiences = ReplayBuffer(buffer_size, batch_size=minibatch_size, num_acts=num_acts, num_obs=num_obs, num_agents=num_agents)
        self.minibatch_size = minibatch_size

        self.exploration_process = OrnsteinUhlenbeckProcess(num_acts)
        self.gamma = 0.95
        self.step = 0
        self.tau = 0.01      
        self.var = [1 for i in range(num_agents)]
        self.episode = 0

    def update(self):
        # understanding from https://www.youtube.com/watch?v=LaIrP-MsPSU + https://arxiv.org/pdf/1706.02275.pdf
        
        act_losses = []
        q_losses = []

        # Sample minibatch of experiences
        batch = self.experiences.sample()
        
        # if np.all(batch==None):
        #     # print("batch empty")
        #     return act_losses, q_losses
        for i in range(self.num_agents):
            
            # mask for episode end
            done_mask = (batch[:,i,-self.num_obs+1] == 0) # HACK: did only for one agent and one future state observation because the done mask will be same for all anyways.
            # print(done_mask.shape)
            none_mask = (batch[:,i,0] == None)

            # print(sum(~none_mask))
            # batch = batch[~done_mask, ...] # only keep episodes which are not terminal

            # ====== preprocess batch for networks ========
            
            obs_batch = torch.from_numpy(batch[:, :, :self.num_obs]).to(self.device)# shape = minibatch_size x num_agents x num_obs
            act_batch = torch.from_numpy(batch[:,:, self.num_obs:self.num_obs+self.num_acts]).to(self.device) # shape = minibatch_size x num_agents x num_acts
            obs_next_batch = torch.from_numpy(batch[:, :, -self.num_obs-1:-1]).to(self.device) # shape = minibatch_size x num_agents x num_obs
            reward_batch = torch.from_numpy(batch[:, i, self.num_obs + self.num_acts, np.newaxis]).to(self.device)

            done_batch = torch.from_numpy(batch[:, i, -1,np.newaxis]).to(self.device)

            # ======= calc optimal q val ========
                # next state actions from all the target actors
            act_next_batch = torch.hstack([self.actor_targets[k].forward(obs_next_batch[:,k,:]) for k in range(self.num_agents)]) # shape = minibatch_size x num_acts
            
                # The future q value depends on all next states and actions
            with torch.no_grad():
                q_next = self.critic_targets[i].forward(torch.hstack([obs_next_batch.flatten(start_dim=1), act_next_batch]))

                    # for the last step in the episode, the target q is 0
                # q_target = torch.zeros(self.minibatch_size, 1, device=self.device).double()
                
                # q_target[~done_mask] = reward_batch.reshape(-1,1) + self.gamma*q_next # shape= batch_size x 1
                
                q_target = reward_batch + (1-done_batch).int()*self.gamma*q_next # shape= batch_size x 1


            # === calulate critic loss ===

            # q_current = torch.zeros(self.minibatch_size, 1, device=self.device).double()
                # just the current observation forwarded through the critic
            q_current = self.critics[i](torch.hstack([obs_batch.flatten(start_dim=1), act_batch.flatten(start_dim=1)]))

            q_loss = self.critic_loss_fn(q_target, q_current) # scalar

            # ====== critic backpropagate and optimize =========
            self.critics[i].optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            self.critics[i].optimizer.step()
            
            # ===== calculate policy gradient =====
                # get current action from current state 
            

            act_i = self.actors[i].forward(obs_batch[:, i, :]) 

                # intersperse this action into the batch (i.e. assuming all other actors are constant => you choose only your own destiny)
            ac = act_batch.clone()
            ac[:, i, :] = act_i    
            # print(act_batch)
                # let's see how good this idea is i.e. ask the omniscient all powerful oracle: the centralised critic
            q_act = self.critics[i].forward(torch.hstack([obs_batch.flatten(start_dim=1), ac.flatten(start_dim=1)]))
            
            
            act_loss = -q_act.mean() # the negative sign is just to make sure it's gradient ascent
            # print(act_loss)

            # ===== actor backpropagate and optimize =====
            self.actors[i].optimizer.zero_grad()
            # for name, param in self.actors[i].named_parameters():
            #     print(name, torch.isfinite(param).all())
            act_loss.backward(retain_graph=True)
            # for name, param in self.actors[i].named_parameters():
            #     print("param.data",torch.isfinite(param.data).all())
            #     print(name, torch.isfinite(param.grad).all())
            self.actors[i].optimizer.step()
            
            # don't append the loss directly! it has the graph attached to it and will increase memory a lot
            act_losses.append(act_loss.data)
            q_losses.append(q_loss.data) 

        # soft update targets
        # if self.step % 100 ==0:
        for j in range(self.num_agents):
            self.soft_update2(self.critic_targets[j], self.critics[j])
            self.soft_update2(self.actor_targets[j], self.actors[j])


        return act_losses, q_losses

    def soft_update2(self, target , source):

        target_params = target.named_parameters()
        source_params = source.named_parameters()

        target_critic_state_dict = dict(target_params)
        critic_state_dict = dict(source_params)
        for name in critic_state_dict:
            critic_state_dict[name] = self.tau*critic_state_dict[name].clone() + \
                    (1-self.tau)*target_critic_state_dict[name].clone()

        target.load_state_dict(critic_state_dict)

    def soft_update(self, target, source):
        # courtesy of github.com/xuehy/pytorch-maddpg.git
    
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            # print(source_param.data)
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)
            # print(target_param.data)

    def get_action_dict(self, obs_dict):
        assert len(obs_dict) == len(self.actors), "csdfg"
        
        # feedforward observation to actor
        with torch.no_grad():
            return {i:self.get_action(i, obs_dict[i]) for i in range(self.num_agents)}
    

    def get_action(self, i, obs:np.ndarray) -> np.ndarray:
        self.step += 1
        # vel = np.zeros(self.num_acts)

        # if self.train:
        #     if not self.experiences.ready() or self.episode<self.episodes_before_train: 
        #         vel += np.random.rand(self.num_acts)


        # if self.episode>self.episodes_before_train and self.train:
        vel = self.actors[i].forward(torch.from_numpy(obs)).detach().numpy()
        if self.train:
            noise = np.random.rand(self.num_acts)
            vel += noise
        # print(vel)
            # print(vel)
            # if self.train:
            #     vel += np.random.rand(self.num_acts) #* self.var[i]

        if self.episode>self.episodes_before_train and self.var[i] > 0.05:
            self.var[i] *= 0.999998
        # if not self.train:
        #     vel = np.clip(vel, 0.0, 1.0)
        # vel[3] *= 3 # max speed
        return vel
    

if __name__=="__main__":

    # unit test maddpg update
    minibatch_size = 5
    num_agents = 2
    num_acts = 3
    num_obs = 4
    size = 10
    algo = MADDPG(train=True, num_agents=num_agents, num_acts=num_acts, num_obs=num_obs,buffer_size=size, minibatch_size=minibatch_size)

    for j in range(size):
        obs_dict = {i:np.random.rand(num_obs) for i in range(num_agents)}
        act_dict = {i:np.random.rand(num_acts) for i in range(num_agents)}
        obs_dict_next = {i:np.random.rand(num_obs) for i in range(num_agents)}
        reward_dict = {i:np.random.rand(1)[0] for i in range(num_agents)}

        if j %2==0:
            # obs_dict = {i:np.full(fill_value=None, shape=(num_obs,)) for i in range(num_agents)}
            obs_dict_next = {i:np.full(fill_value=None, shape=(num_obs,)) for i in range(num_agents)}

        algo.experiences.push((obs_dict, act_dict, reward_dict, obs_dict_next))

    algo.update()

    
