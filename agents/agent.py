import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces


class Agent(nn.Module):
    def __init__(
        self, 
        agent_name: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int = 1,
        writer: SummaryWriter | None = None,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.agent_name = agent_name
        self.observation_space = observation_space
        self.action_space = action_space
        self.writer = writer
        self.num_envs = num_envs
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        self.device = device

    @abstractmethod
    def on_observe(self, **kwargs):
        ...

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
    
    @abstractmethod
    def on_act(self, **kwargs):
        ...

    @abstractmethod
    def update(self, global_step: int) -> dict[str, float]:
        ...
