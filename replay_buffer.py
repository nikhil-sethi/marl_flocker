import numpy as np

class ReplayBuffer():
    def __init__(self, size, num_acts, num_obs, num_agents) -> None:
        self._size = int(size)
        self._index = 0
        experince_size = int(1 + num_acts + 2*num_obs) # extra one is for the reward 
        self._buffer = np.full(fill_value=None, shape=(num_agents, int(size), experince_size))
        self._sampler = np.random.default_rng()

    def push(self, experience: tuple):

        obs = np.array(list(experience[0].values()))
        act = np.array(list(experience[1].values()))
        rewards = np.array(list(experience[2].values()))[:,np.newaxis]
        
        obs_next = np.array(list(experience[3].values()))

        self._buffer[:, self._index, :] = np.hstack([obs, act, rewards, obs_next])
        self._index = (self._index +1)% self._size # the % operator allows to cycle and replace old expereiences

    def sample(self, size):
        """Returns a (size x num_agents x experience_size) sample from the buffer"""
        return self._sampler.choice(self._buffer, size, axis=1, replace=False)
    

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