import numpy as np

class ReplayBuffer():
    def __init__(self, size, num_acts, num_obs, num_agents) -> None:
        self._size = int(size)
        self._index = 0
        self._buffer = np.full(fill_value=None, shape=(num_agents, int(size), int(1 + num_acts + 2*num_obs)))


    def push(self, experience: tuple):

        obs = np.array(list(experience[0].values()))
        act = np.array(list(experience[1].values()))
        rewards = np.array(list(experience[2].values()))[:,np.newaxis]
        
        obs_next = np.array(list(experience[3].values()))

        self._buffer[:, self._index, :] = np.hstack([obs, act, rewards, obs_next])
        self._index = (self._index +1)% self._size