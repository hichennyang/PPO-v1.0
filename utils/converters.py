from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Categorical, Normal
from torch.nn import CrossEntropyLoss, MSELoss
from gymnasium import spaces



# 参考：https://github.com/adik993/ppo-pytorch/blob/master/envs/converters.py#L15
# 下一步工作内容，先按照这个converter改进现有PPO算法的接口
# 下下步工作内容，实现ICM模块
# 下下下步工作内容，改进ICM模块，使其适应含有冗余信息的输入（其实就是在特征提取模块上做文章），先用一个神经网络提取状态特征，然后再用原始特征提取得到的结果减去该特征
class Converter(ABC):
    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """
        Whether underlying space is discrete or not
        :return: ``True`` if space is discrete aka. ``gym.spaces.Discrete``, ``False`` otherwise
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Returns a tuple of integers representing the shape of the observation to be passed as input to the
        model

        :return: tuple of integers representing the shape of the observation/
        """
        raise NotImplementedError()

    @abstractmethod
    def distribution(self, logits: Tensor) -> Distribution:
        """
        Returns a distribution appropriate for a ``gym.Space`` parametrized using provided ``logits``

        :return: logits returned by the model
        """
        raise NotImplementedError()
    
    @abstractmethod
    def action(self, tensor: Tensor) -> Tensor:
        """
        Converts logits to action

        :param tensor: logits(output from the model before calling activation function) parametrizing action space
                       distribution
        :return: a tensor containing the action
        """
        raise NotImplementedError('Implement me')
    
    @abstractmethod
    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        """
        Returns the distance between two tensors of an underlying space
        :param policy_logits: predictions
        :param y: actual values
        :return: distance/loss
        """
        raise NotImplementedError()
    
    @abstractmethod
    def policy_out_model(self, in_features: int) -> nn.Module:
        """
        Returns the output layer for the policy that is appropriate for a given action space
        :return: torch module that accepts ``in_features`` and outputs values for policy
        """
        raise NotImplementedError()
    
    @staticmethod
    def for_space(space: spaces.Space):
        if isinstance(space, spaces.Discrete):
            return DiscreteConverter(space)
        elif isinstance(space, spaces.Box):
            return BoxConverter(space)


class DiscreteConverter(Converter):
    def __init__(self, space: spaces.Discrete) -> None:
        self.space = space
        self.loss = CrossEntropyLoss()

    @property
    def is_discrete(self) -> bool:
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        return self.space.n,

    def distribution(self, logits: Tensor) -> Distribution:
        return Categorical(logits=logits)

    def action(self, tensor: Tensor) -> Tensor:
        return self.distribution(tensor).sample()
    
    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        return self.loss(policy_logits, y.long())
    
    def policy_out_model(self, in_features: int) -> nn.Module:
        return nn.Linear(in_features, self.shape[0])

class BoxConverter(Converter):
    def __init__(self, space: spaces.Box) -> None:
        self.space = space
        self.loss = MSELoss()

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def shape(self) -> tuple[int, ...]:
        return self.space.shape

    def distribution(self, logits: Tensor) -> Distribution:
        assert logits.size(-1) % 2 == 0
        mid = logits.size(-1) // 2
        loc = logits[..., :mid]
        scale = logits[..., mid:]
        return Normal(loc, scale)

    def action(self, logits: Tensor) -> Tensor:
        min = torch.tensor(self.space.low, device=logits.device)
        max = torch.tensor(self.space.high, device=logits.device)
        return torch.max(torch.min(self.distribution(logits=logits).sample(), max), min)
    
    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        return self.loss(self.action(policy_logits), y)

    def policy_out_model(self, in_features: int) -> nn.Module:
        return NormalDistributionModule(in_features, self.shape[0])

class NormalDistributionModule(nn.Module):
    def __init__(self, in_features: int, n_action_values: int):
        super().__init__()
        self.policy_mean = nn.Linear(in_features, n_action_values)
        self.policy_std = nn.Parameter(torch.zeros(1, n_action_values))

    def forward(self, x):
        mean = self.policy_mean(x)
        std = self.policy_std.expand_as(mean).exp()
        return torch.cat((mean, std), dim=-1)

def get_converter(space: spaces.Space) -> Converter:
    if isinstance(space, spaces.Box):
        converter = BoxConverter(space)
    else:
        raise NotImplementedError()

    return converter