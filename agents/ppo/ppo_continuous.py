import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from agents.modules import GaussianActor, Critic
from agents.agent import Agent


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
        device: torch.device = torch.device("cpu")
    ):
        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.Box)
        super().__init__(agent_name, observation_space, action_space, writer, num_envs, device)

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

        self.actor = GaussianActor(
            state_dim = observation_space.shape[-1],
            action_dim = action_space.shape[-1],
            action_min = self.action_min,
            action_max = self.action_max,
            hidden_sizes = actor_hidden_sizes
        ).to(device=self.device)
        self.critic = Critic(
            state_dim = observation_space.shape[-1],
            hidden_sizes = critic_hidden_sizes
        ).to(device=self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=self.adam_eps)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=self.adam_eps)

    def on_observe(
        self,
        obs: np.ndarray
    ):
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

    def on_act(self, **kwargs):
        pass
    
    def update(self, replay_buffer, total_steps: int):
        obs, act, act_log_prob, obs_next, rew, done = replay_buffer.sample()

        obs = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)
        act = torch.from_numpy(act).to(dtype=torch.float32, device=self.device)
        act_log_prob = torch.from_numpy(act_log_prob).to(dtype=torch.float32, device=self.device)
        obs_next = torch.from_numpy(obs_next).to(dtype=torch.float32, device=self.device)
        rew = torch.from_numpy(rew).to(dtype=torch.float32, device=self.device)
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

                v_s = self.critic(obs[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()


        return {"actor_loss": actor_loss.mean().item(), "critic_loss": critic_loss.item()}
