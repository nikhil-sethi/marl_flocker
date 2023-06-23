import torch
import copy

class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mean=0, theta=0.15, sigma=0.2) -> None:
        self.theta = theta
        self.mean = mean*torch.ones(size)
        self.sigma = sigma
        self.size = size
        self.reset

    def reset(self):
        self.state = copy.copy(self.mean)

    def sample(self):
        dx = self.theta*(self.mean - self.state) + self.sigma*torch.normal(self.size)
        self.state += dx

        return self.state
    




# class DiagGaussianProcess():
#     def __init__(self, flat):
#         self.flat = flat
#         # mean, logstd = tf.split(axis=1, num_or_size_splits=2, value=flat)
#         self.mean = mean
#         self.logstd = logstd
#         self.std = tf.exp(logstd)

#     def sample(self):
#         return self.mean +  torch.normal(self.mean.shape)
    
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)