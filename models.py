from torch import nn
import torch
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, num_units=64, alpha=0.001, device = torch.device('cpu')) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        self._model = nn.Sequential(
            nn.Linear(input_size, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, output_size)
        ).double()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
 
        self.to(self.device)
        
    def forward(self, input):
        # input_flat = self.flatten(input)

        return self._model(input)


class Actor(BaseModel):
    """Policy network"""
    def __init__(self, input_shape, output_size, device= torch.device('cpu')) -> None:
        super().__init__(input_shape, output_size, device=device)
        num_units = 64
        # self._model.add_module("last", nn.Tanh())
        # self.to(self.device)

    # def forward(self, inp):
    #     return nn.Tanh


class Critic(BaseModel):
    """Q-value network"""
    def __init__(self, input_shape, output_size=1, device= torch.device('cpu')) -> None:
        super().__init__(input_shape, output_size, device=device)

    # def forward(self):
    #     pass


if __name__=="__main__":
    actor = Actor(12,4)

    action = actor(torch.from_numpy(np.random.rand(12)))

    print(action)