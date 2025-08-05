import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

class Agent(nn.Module):
    def __init__(
        self, 
        batch_size: int,
        mini_batch_size: int,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.writer = writer
        self.device = device

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def update(
        self,
        replay_buffer, 
        total_steps
    ):
        ...
