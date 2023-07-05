import numpy as np

class ReplayBuffer():
    def __init__(self, size, batch_size, num_acts, num_obs, num_agents) -> None:
        self._size = int(size)
        self._batch_size = batch_size
        self._counter = 0
        self.num_obs = num_obs
        self.num_acts = num_acts
        experince_size = int(2 + num_acts + 2*num_obs) # done, reward, acts, obs, obs_next
        self._buffer = np.zeros(shape=(int(size),num_agents, experince_size), dtype=np.float)
        # self._sampler = np.random.default_rng()

    def push(self, experience: tuple):

        obs = np.array(list(experience[0].values()))
        # print(experience[1])
        act = np.array(list(experience[1].values()))
        rewards = np.array(list(experience[2].values()))[:,np.newaxis]
        obs_next = np.array(list(experience[3].values()))
        dones = experience[4]

        index = self._counter % self._size # the % operator allows to cycle and replace old expereiences
        self._buffer[index, :, :] = np.hstack([obs, act, rewards, obs_next, dones])
        self._counter +=1
    

    def sample(self):
        """Returns a (size x num_agents x experience_size) sample from the buffer"""
        choice = np.random.choice(min(self._counter, self._size), self._batch_size, replace=False)

        obs = self._buffer[choice, :, :self.num_obs]# shape = minibatch_size x num_agents x num_obs
        act = self._buffer[choice, :, self.num_obs:self.num_obs + self.num_acts] # shape = minibatch_size x num_agents x num_acts
        # print(act)
        rewards = self._buffer[choice, :, self.num_obs + self.num_acts]
        obs_next = self._buffer[choice, :, -self.num_obs-1:-1] # shape = minibatch_size x num_agents x num_obs
        dones = self._buffer[choice, :, -1]

        return obs, act, rewards, obs_next, dones 

    def ready(self):
        return self._counter >= self._batch_size

if __name__=="__main__":
    # unit testing the replay buffer

    num_agents = 2
    num_acts = 3
    num_obs = 4
    size = 10
    experiences = ReplayBuffer(size, num_acts, num_obs, num_agents)
    
    for _ in range(size):
        obs_dict = {i:np.random.rand(num_obs) for i in range(num_agents)}
        act_dict = {i:np.random.rand(num_acts) for i in range(num_agents)}
        obs_dict_next = {i:np.random.rand(num_obs) for i in range(num_agents)}
        reward_dict = {i:np.random.rand(1)[0] for i in range(num_agents)}

        experiences.push((obs_dict, act_dict, reward_dict, obs_dict_next))

    print(experiences.sample(6))