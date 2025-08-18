import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.distributions import Beta, Normal
from gymnasium import spaces

from utils.converters import get_converter, Converter, BoxConverter, DiscreteConverter

# https://github.com/adik993/ppo-pytorch/blob/master/curiosity/icm.py

def orthogonal_init(layer: nn.Linear, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class GaussianActor(nn.Module):
    def __init__(
        self,
        state_converter: Converter,
        action_converter: Converter, 
        hidden_sizes: list[int] = (),
    ):
        super().__init__()
        self.state_converter = state_converter
        self.action_converter = action_converter

        model = []
        hidden_sizes = [self.state_converter.shape[0], *list(hidden_sizes)]
        
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:], strict=True):
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            layers += [nn.Tanh()]

            model += layers

        # model += [nn.Linear(hidden_sizes[-1], action_dim)]
        model.append(self.action_converter.policy_out_model(hidden_sizes[-1]))
        self.model = nn.Sequential(*model)

        for module in self.model:
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
        
        # self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # mean: torch.Tensor = self.action_min + self.action_range*((1+torch.tanh(self.model(state))) / 2)
        # return mean
        return self.model(state)

    def get_dist(self, state: torch.Tensor) -> Normal:
        # mean = self.forward(state)
        # log_std = self.log_std.expand_as(mean)
        # std = torch.exp(log_std)
        # dist = Normal(mean, std)
        logits = self.forward(state)
        dist = self.action_converter.distribution(logits)
        return dist


class Critic(nn.Module):
    def __init__(
        self, 
        state_converter: Converter,
        hidden_sizes: list[int] = (),
    ):
        super().__init__()
        self.state_converter = state_converter
        
        model = []
        hidden_sizes = [self.state_converter.shape[0], *list(hidden_sizes)]
        
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
    

class ForwardModel(nn.Module):
    def __init__(self, action_converter: Converter, state_feature_dims: int):
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 128
        if action_converter.is_discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)
        
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + state_feature_dims, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_feature_dims)
        )

    def forward(self, state_latent: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.is_discrete else action)
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, action_converter: Converter, state_latent_features: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            action_converter.policy_out_model(128)
        )

    def forward(self, state_latent: Tensor, next_state_latent: Tensor):
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))
    
class ICM(nn.Module):
    def __init__(self, state_converter: Converter, action_converter: Converter, intrinsic_reward_integration=0.01):
        super().__init__()
        self.action_converter = action_converter
        self.tau = intrinsic_reward_integration

        self.encoder = nn.Sequential(
            nn.Linear(state_converter.shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.forward_model = ForwardModel(action_converter, 128)
        self.inverse_model = InverseModel(action_converter, 128)
    
    def forward(self, state: Tensor, next_state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    def reward(self, reward: Tensor, state: Tensor, next_state: Tensor, action: Tensor):
        with torch.no_grad():
            next_state, next_state_hat, _ = self.forward(state, next_state, action)
            intrinsic_reward = (next_state - next_state_hat).norm(2, dim=-1, keepdim=True).pow(2)

        return (1.-self.tau)*reward + self.tau*intrinsic_reward

    # def loss(self, state: Tensor, next_state: Tensor, action: Tensor):
    #     next_state, next_state_hat, action_hat = self.forward(state, next_state, action)
    #     forward_loss = 0.5 * (next_state_hat - next_state.detach()).norm(2, dim=-1).pow(2).mean()
    #     inverse_loss = self.action_converter.distance(action_hat, action)
    #     curiosity_loss = forward_loss + inverse_loss
    #     return curiosity_loss