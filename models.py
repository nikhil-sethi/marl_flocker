from torch import nn
import torch
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, num_units=64) -> None:
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
        
    def forward(self, input):
        # input_flat = self.flatten(input)

        return self._model(input)


class Actor(BaseModel):
    """Policy network"""
    def __init__(self, input_shape, output_size) -> None:
        super().__init__(input_shape, output_size)

    # def forward(self, inp):
    #     pass


class Critic(BaseModel):
    """Q-value network"""
    def __init__(self, input_shape, output_size=1) -> None:
        super().__init__(input_shape, output_size)

    # def forward(self):
    #     pass


if __name__=="__main__":
    actor = Actor(12,4)

    action = actor(torch.from_numpy(np.random.rand(12)))

    print(action)