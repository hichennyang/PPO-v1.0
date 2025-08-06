import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal


def orthogonal_init(layer: nn.Linear, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int, 
        action_dim: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,  
        hidden_sizes: list[int] = (),
    ):
        super().__init__()
        assert action_min.shape == action_max.shape
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min

        model = []
        hidden_sizes = [state_dim, *list(hidden_sizes)]
        
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:], strict=True):
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            layers += [nn.Tanh()]

            model += layers

        model += [nn.Linear(hidden_sizes[-1], action_dim)]
        self.model = nn.Sequential(*model)

        for module in self.model:
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
        
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) 
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = self.action_min + self.action_range*((1+torch.tanh(self.model(state))) / 2)
        return mean

    def get_dist(self, state: torch.Tensor) -> Normal:
        mean = self.forward(state)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self, 
        state_dim: int,
        hidden_sizes: list[int] = (),
    ):
        super().__init__()

        model = []
        hidden_sizes = [state_dim, *list(hidden_sizes)]
        
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:], strict=True):
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            layers += [nn.Tanh()]

            model += layers

        model += [nn.Linear(hidden_sizes[-1], 1)]
        self.model = nn.Sequential(*model)

        for module in self.model:
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        v_s: torch.Tensor = self.model(state)
        return v_s

