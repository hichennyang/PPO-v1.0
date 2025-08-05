import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from agents.modules import Actor_Gaussian, GaussianActor, Critic
from agents.agent import Agent


class PPOContinuous(Agent):
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        action_min: np.ndarray,
        action_max: np.ndarray,
        batch_size: int,
        mini_batch_size: int,
        max_train_steps: int,
        lr_a: float,
        actor_hidden_sizes: list[int],
        lr_c: float,
        critic_hidden_sizes: list[int],
        gamma: float,
        gae_lambda: float,
        epsilon: float,
        policy_entropy_coef: float,
        use_grad_clip: bool,
        use_lr_decay: bool,
        use_adv_norm: bool,
        repeat: int,
        adam_eps: float = 1e-8,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(batch_size, mini_batch_size, writer, device)
        self.action_min = torch.from_numpy(action_min).to(device=self.device)
        self.action_max = torch.from_numpy(action_max).to(device=self.device)
        # self.batch_size = args.batch_size
        # self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = max_train_steps
        self.lr_a = lr_a  # Learning rate of actor
        self.lr_c = lr_c  # Learning rate of critic
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE parameter
        self.epsilon = epsilon  # PPO clip parameter
        self.repeat = repeat  # PPO parameter
        self.entropy_coef = policy_entropy_coef  # Entropy coefficient
        self.adam_eps = adam_eps
        self.use_grad_clip = use_grad_clip
        self.use_lr_decay = use_lr_decay
        self.use_adv_norm = use_adv_norm

        self.actor = GaussianActor(
            state_dim = state_dim,
            action_dim = action_dim,
            action_min = self.action_min,
            action_max = self.action_max,
            hidden_sizes = actor_hidden_sizes
        ).to(device=self.device)
        self.critic = Critic(
            state_dim = state_dim,
            hidden_sizes = critic_hidden_sizes
        ).to(device=self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=self.adam_eps)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=self.adam_eps)

    def evaluate(self, state: np.ndarray):  # When evaluating the policy, we only use the mean
        state = torch.tensor(state, dtype=torch.float).unsqueeze(dim=0)
        state = state.to(device=self.device)
        action: torch.Tensor = self.actor(state)
        return action.detach().cpu().squeeze(dim=0).numpy()

    def choose_action(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(dim=0)
        state = state.to(device=self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(state)
            act = dist.sample()  # Sample the action according to the probability distribution
            act = torch.clamp(act, min=self.action_min, max=self.action_max)  # [-max,max]
            act_log_prob: torch.Tensor = dist.log_prob(act)  # The log probability density of the action
        return act.cpu().squeeze(dim=0).numpy(), act_log_prob.cpu().squeeze(dim=0).numpy()

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
        if self.use_adv_norm:
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.repeat):
            for index in BatchSampler(SubsetRandomSampler(range(len(obs))), self.mini_batch_size, False):
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
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(obs[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

        return {"actor_loss": actor_loss.mean().item(), "critic_loss": critic_loss.item()}
    
    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p["lr"] = lr_c_now
