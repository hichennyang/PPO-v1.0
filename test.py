import subprocess
import random
import gymnasium
import torch
import numpy as np
from typing import Any, SupportsFloat
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.ppo.ppo_continuous import PPOContinuous


def test(env_id: str, agent: PPOContinuous, episodes: int, seed: int | None = None, **env_kwargs: Any) -> dict[str, SupportsFloat]:
    env = gymnasium.make_vec(env_id, num_envs=episodes, **env_kwargs)
    observations, infos = env.reset()
    returns = np.zeros(shape=(env.num_envs,), dtype=np.float32)
    returns_mean = []
    while episodes > 0:
        action, _ = agent(observations)
        observations_next, rewards, terminations, truncations, infos = env.step(action)

        returns += rewards
        done = np.logical_or(terminations, truncations)
        if np.any(done):
            indices = np.where(done)[0]
            for index in indices:
                returns_mean.append(returns[index])
            returns[indices] = 0.0
            episodes -= len(indices)

        observations = observations_next
    
    return {f"test/{agent.agent_name}_return_mean": np.asarray(returns_mean).mean()}
        
def train(env_id: str, run_dir: Path, steps:int, seed: int | None = None):
    # 设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建仿真环境（训练）
    env = gymnasium.make_vec(env_id, num_envs=1)
    
    # 创建智能体
    writer = SummaryWriter(log_dir=run_dir)
    agent = PPOContinuous(
        agent_name = "agent",
        observation_space = env.single_observation_space,
        action_space = env.single_action_space,
        batch_size = 50,
        replaybuffer_size = 2000,
        lr_a = 3e-4,
        actor_hidden_sizes = [128, 128],
        lr_c = 3e-4,
        critic_hidden_sizes = [128, 128],
        gamma = 0.99,
        gae_lambda = 0.95,
        epsilon = 0.2,
        policy_entropy_coef = 0.01,
        adam_eps = 1e-5,
        writer = writer,
        num_envs = env.num_envs,
        seed = seed,
        device = torch.device("cuda")
    )

    observations, infos = env.reset(seed=seed)
    progress_bar = tqdm(total=steps, desc="Training Progress")
    while progress_bar.n < progress_bar.total:
        agent.on_observe()
        actions, actions_log_prob = agent(observations)
        observations_next, rewards, terminations, truncations, infos = env.step(actions)
        agent.on_act(
            rewards, terminations, truncations, observations, actions, actions_log_prob, progress_bar.n 
        )
        progress_bar.update(env.num_envs)
        
        # 更新智能体
        if progress_bar.n % (env.num_envs*agent.replay_buffer.max_size) == 0:
            agent.update(progress_bar.n)
        
        # 测试智能体
        if progress_bar.n % int(2e4) == 0:
            performance_dict = test(env_id, agent, episodes=10)
            for k, v in performance_dict.items():
                writer.add_scalar(k, v, global_step=progress_bar.n)
            
        # 保存智能体
        if progress_bar.n % 100000 == 0:
            torch.save(agent.actor, run_dir/"actor.pt")
            torch.save(agent.actor.state_dict(), run_dir/"actor.pth")

        observations = observations_next

def launch_tensorboard(log_dir):
    """启动TensorBoard"""
    cmd = ["tensorboard", "--logdir", str(log_dir), "--port", "6006"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    tensorboard_url = "http://localhost:6006"
    print(f"🚀 TensorBoard 启动: {tensorboard_url}")
    
    return process

if __name__ == "__main__":
    now = datetime.now()
    run_dir = Path.cwd() / "runs" / now.strftime("%b%d_%H-%M-%S")
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    tensorboard_process = launch_tensorboard(run_dir.parent)

    train("BipedalWalker-v3", run_dir, steps=int(5e6), seed=1)
