import gymnasium
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.ppo.ppo_continuous import PPOContinuous
from utils import ReplayBuffer

def evaluate_policy(env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, truncated, terminated, _ = env.step(action)
            done = truncated | terminated

            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times

def main(env_id: str, seed: int):
    train_env = gymnasium.make(env_id)
    evalu_env = gymnasium.make(env_id)

    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    replay_buffer = ReplayBuffer(
        state_dim = train_env.observation_space.shape[-1],
        action_dim = train_env.action_space.shape[-1],
        size = 2000,
        ignore_obs_next = True
    )
    agent = PPOContinuous(
        state_dim = train_env.observation_space.shape[-1],
        action_dim = train_env.action_space.shape[-1],
        action_min = train_env.action_space.low,
        action_max = train_env.action_space.high,
        batch_size = 2000,
        mini_batch_size = 50,
        max_train_steps = int(3e6),
        lr_a = 3e-4,
        actor_hidden_sizes = [128, 128],
        lr_c = 3e-4,
        critic_hidden_sizes = [128, 128],
        gamma = 0.99,
        gae_lambda = 0.95,
        epsilon = 0.2,
        policy_entropy_coef = 0.01,
        use_grad_clip = True,
        use_lr_decay = True,
        use_adv_norm = True,
        repeat = 10,
        adam_eps = 1e-5,
        device = torch.device("cuda")
    )

    # Build a tensorboard
    writer = SummaryWriter()
    done = True
    total_steps = 0
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating

    while total_steps < int(3e6):
        if done:
            obs, _ = train_env.reset()

        act, act_log_prob = agent.choose_action(obs)  # Action and the corresponding log probability
        obs_next, rew, truncated, terminated, _ = train_env.step(act)
        done = truncated | terminated

        replay_buffer.add(
            rew,
            terminated,
            truncated,
            obs,
            act,
            act_log_prob,
        )
        obs = obs_next
        total_steps += 1
        
        # When the number of transitions in buffer reaches batch_size,then update
        if replay_buffer.size[0].item() >= 2000:
            agent.update(replay_buffer, total_steps)
            replay_buffer.reset()
        
        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % int(5e3) == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(evalu_env, agent)
            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            writer.add_scalar('step_rewards_{}'.format(env_id), evaluate_rewards[-1], global_step=total_steps)

if __name__ == "__main__":
    main("BipedalWalker-v3", seed=10)
