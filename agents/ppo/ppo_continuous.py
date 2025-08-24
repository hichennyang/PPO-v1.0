import torch
import torch.nn.functional as F
import numpy as np
from typing import SupportsFloat
from functools import singledispatchmethod
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from agents.agent import Agent
from agents.modules import GaussianActor, Critic, ICM
from utils import ReplayBuffer
from utils.converters import get_converter, Converter

class PPOContinuous(Agent):
    def __init__(
        self, 
        agent_name: str,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        batch_size: int,
        replaybuffer_size: int,
        lr_a: float,
        actor_hidden_sizes: list[int],
        lr_c: float,
        critic_hidden_sizes: list[int],
        gamma: float,
        gae_lambda: float,
        epsilon: float,
        policy_entropy_coef: float,
        adam_eps: float = 1e-8,
        writer: SummaryWriter | None = None,
        num_envs: int = 1,
        seed: int | None = None,
        device: torch.device = torch.device("cpu")
    ):
        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.Box)
        super().__init__(agent_name, observation_space, action_space, num_envs, writer, seed, device)

        self.batch_size = batch_size
        self.lr_a = lr_a                            # Learning rate of actor
        self.lr_c = lr_c                            # Learning rate of critic
        self.gamma = gamma                          # Discount factor
        self.gae_lambda = gae_lambda                # GAE parameter
        self.epsilon = epsilon                      # PPO clip parameter
        self.entropy_coef = policy_entropy_coef     # Entropy coefficient
        self.adam_eps = adam_eps

        self.action_min = torch.from_numpy(action_space.low).to(device=self.device)
        self.action_max = torch.from_numpy(action_space.high).to(device=self.device)

        state_converter = get_converter(observation_space)
        self.action_converter = get_converter(action_space)

        self.actor = GaussianActor(
            state_converter = state_converter,
            action_converter = self.action_converter,
            # state_dim = observation_space.shape[-1],
            # action_dim = action_space.shape[-1],
            # action_min = self.action_min,
            # action_max = self.action_max,
            hidden_sizes = actor_hidden_sizes
        ).to(device=self.device)
        self.critic = Critic(
            state_converter = state_converter,
            # state_dim = observation_space.shape[-1],
            hidden_sizes = critic_hidden_sizes
        ).to(device=self.device)
        self.icm = ICM(
            state_converter = state_converter,
            action_converter = self.action_converter,
            intrinsic_reward_integration = 0.1
        ).to(device=self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=self.adam_eps)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=self.adam_eps)
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=self.lr_a)

        self.replay_buffer = ReplayBuffer(
            state_dim = observation_space.shape[-1],
            action_dim = action_space.shape[-1],
            size = replaybuffer_size,
            num_envs = num_envs,
            ignore_obs_next = True
        )
        self.returns = np.zeros(shape=(self.num_envs, 1), dtype=np.float32)

    def on_observe(self, **kwargs):
        pass

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(dim=0)
        obs = obs.to(device=self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(obs)
            act = dist.sample()  # Sample the action according to the probability distribution
            act = torch.clamp(act, min=self.action_min, max=self.action_max)  # [-max,max]
            act_log_prob: torch.Tensor = dist.log_prob(act)  # The log probability density of the action
        return act.cpu().squeeze(dim=0).numpy(), act_log_prob.cpu().squeeze(dim=0).numpy()

    @singledispatchmethod
    def on_act(
        self, 
        rew: SupportsFloat | np.ndarray, 
        terminated: bool | np.ndarray,
        truncated: bool | np.ndarray,
        obs: np.ndarray,
        act: np.ndarray, 
        act_log_prob: np.ndarray, 
        global_step: int,
        env_indices: np.ndarray | None = None
    ):
        ...
    
    @on_act.register
    def _(
        self, 
        rew: SupportsFloat, 
        terminated: bool,
        truncated: bool,
        obs: np.ndarray,
        act: np.ndarray, 
        act_log_prob: np.ndarray, 
        global_step: int,
        env_indices: np.ndarray | None = None
    ):
        self.replay_buffer.add(rew, terminated, truncated, obs, act, act_log_prob, env_indices)
        if env_indices is None:
            env_indices = np.arange(self.num_envs)
        self.returns[env_indices] += rew
        done = terminated | truncated
        if done:
            if self.writer:
                self.writer.add_scalar(f"return/{self.agent_name}_mean", self.returns[0].item(), global_step)
            
            self.returns[0] = 0.0

    @on_act.register
    def _(
        self, 
        rew: np.ndarray, 
        terminated: np.ndarray,
        truncated: np.ndarray,
        obs: np.ndarray,
        act: np.ndarray, 
        act_log_prob: np.ndarray, 
        global_step: int,
        env_indices: np.ndarray | None = None
    ):
        self.replay_buffer.add(rew, terminated, truncated, obs, act, act_log_prob, env_indices)
        if env_indices is None:
            env_indices = np.arange(self.num_envs)
        self.returns[env_indices] += np.expand_dims(rew, axis=1)
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            indices = np.where(done)[0]                 #TODO: 仿真环境并行下，当每一时刻不是所有的仿真环境都能执行动作的情况下具有严重BUG
            # indices = env_indices[done]               # 解决上述问题的代码
            if self.writer:
                self.writer.add_scalar(f"return/{self.agent_name}_mean", self.returns[indices].mean(axis=0), global_step)
                if len(indices) > 1:
                    self.writer.add_scalar(f"return/{self.agent_name}_std", self.returns[indices].std(axis=0), global_step)
            self.returns[indices] = 0.0


    def update(self, global_step: int):
        obs, act, act_log_prob, obs_next, rew, done = self.replay_buffer.sample()

        obs = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)
        act = torch.from_numpy(act).to(dtype=torch.float32, device=self.device)
        act_log_prob = torch.from_numpy(act_log_prob).to(dtype=torch.float32, device=self.device)
        obs_next = torch.from_numpy(obs_next).to(dtype=torch.float32, device=self.device)
        raw_rew = torch.from_numpy(rew).to(dtype=torch.float32, device=self.device)
        
        rew = self.icm.reward(raw_rew, obs, obs_next, act)
        
        done = torch.from_numpy(done).to(dtype=torch.float32, device=self.device)
        assert len(obs) == len(act) == len(act_log_prob) == len(obs_next) == len(rew) == len(done)
        
        # Calculate the advantage using GAE
        with torch.no_grad():  # adv and v_target have no gradient
            v_s: torch.Tensor = self.critic(obs)
            v_s = v_s.detach()
            v_s_prime: torch.Tensor = self.critic(obs_next)
            v_s_prime = v_s_prime.detach()

        adv = torch.zeros(rew.shape).to(device=self.device)
        delta = rew + self.gamma*(1.0-done)*v_s_prime - v_s
        discount = (1-done)*self.gamma*self.gae_lambda
        _gae = 0.0
        for i in range(len(adv)-1, -1, -1):
            _gae = delta[i] + discount[i]*_gae
            adv[i] = _gae 

        v_target = adv + v_s

        adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        actor_loss_list: list[SupportsFloat] = []
        critic_loss_list: list[SupportsFloat] = []
        for _ in range(10):
            for index in BatchSampler(SubsetRandomSampler(range(len(obs))), self.batch_size, False):
                dist = self.actor.get_dist(obs[index])
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                a_logprob_now: torch.Tensor = dist.log_prob(act[index])
                ratio = torch.exp(a_logprob_now.sum(1, keepdim=True) - act_log_prob[index].sum(1, keepdim=True))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef*dist_entropy                          # calculate policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # Update critic
                v_s = self.critic(obs[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                
                # Update icm
                next_state, next_state_hat, action_hat = self.icm(obs[index], obs_next[index], act[index])
                forward_loss = 0.5 * (next_state_hat - next_state.detach()).norm(2, dim=-1).pow(2).mean()
                inverse_loss = self.action_converter.distance(action_hat, act[index])
                curiosity_loss = forward_loss + inverse_loss
                self.optimizer_icm.zero_grad()
                curiosity_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
                self.optimizer_icm.step()

                actor_loss_list.append(actor_loss.mean().item())
                critic_loss_list.append(critic_loss.item())
            
        actor_loss_mean = np.asarray(actor_loss_list, dtype=np.float32).mean()
        actor_loss_std = np.asarray(actor_loss_list, dtype=np.float32).std()
        critic_loss_mean = np.asarray(critic_loss_list, dtype=np.float32).mean()
        critic_loss_std = np.asarray(critic_loss_list, dtype=np.float32).std()

        self.replay_buffer.reset()

        if self.writer:
            self.writer.add_scalar(f"loss/{self.agent_name}_actor_mean", actor_loss_mean, global_step)
            self.writer.add_scalar(f"loss/{self.agent_name}_actor_std", actor_loss_std, global_step)
            self.writer.add_scalar(f"loss/{self.agent_name}_critic_mean", critic_loss_mean, global_step)
            self.writer.add_scalar(f"loss/{self.agent_name}_critic_std", critic_loss_std, global_step)

        return {
            f"loss/{self.agent_name}_actor_mean": actor_loss_mean, 
            f"loss/{self.agent_name}_actor_std": actor_loss_std,
            f"loss/{self.agent_name}_critic_mean": critic_loss_mean,
            f"loss/{self.agent_name}_critic_std": critic_loss_std
        }
