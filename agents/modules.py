import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal


# Trick 8: orthogonal initialization
def orthogonal_init(layer: nn.Linear, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(
        self, 
        args,
        state_dim: int,
        action_dim: int,
        max_action,
        use_orthogonal_init: bool = True,
        use_tanh: bool = True,
    ):
        super(Actor_Gaussian, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]  # Trick10: use tanh

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


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

# class Critic(nn.Module):
#     def __init__(self, args):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
#         self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
#         self.fc3 = nn.Linear(args.hidden_width, 1)
#         self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

#         if args.use_orthogonal_init:
#             print("------use_orthogonal_init------")
#             orthogonal_init(self.fc1)
#             orthogonal_init(self.fc2)
#             orthogonal_init(self.fc3)

#     def forward(self, s):
#         s = self.activate_func(self.fc1(s))
#         s = self.activate_func(self.fc2(s))
#         v_s = self.fc3(s)
#         return v_s
