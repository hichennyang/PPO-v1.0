import gymnasium
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.ppo.ppo_continuous import PPOContinuous


def evaluate_policy(env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = agent(s)  # We use the deterministic policy during the evaluating
            s_, r, truncated, terminated, _ = env.step(action)
            done = truncated | terminated

            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times

def main(env_id: str, seed: int):
    # train_env = gymnasium.make(env_id)
    train_env = gymnasium.make_vec(env_id, num_envs=10)
    evalu_env = gymnasium.make(env_id)

    # # set random seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # Build a tensorboard
    writer = SummaryWriter()
  
    agent = PPOContinuous(
        agent_name = "agent",
        observation_space = train_env.single_observation_space,
        action_space = train_env.single_action_space,
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
        num_envs = train_env.num_envs,
        device = torch.device("cuda")
    )

    done = True
    total_steps = 0
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating

    obs, _ = train_env.reset()
    while total_steps < int(3e6):
        # if done:
        #     obs, _ = train_env.reset()

        act, act_log_prob = agent(obs)  # Action and the corresponding log probability
        obs_next, rew, truncated, terminated, _ = train_env.step(act)
        done = truncated | terminated
        agent.on_act(
            rew, terminated, truncated, obs, act, act_log_prob, total_steps 
        )

        obs = obs_next
        total_steps += 1
        
        # When the number of transitions in buffer reaches batch_size,then update
        if agent.replay_buffer.size[0].item() >= 2000:
            agent.update(total_steps)
        
        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % int(5e3) == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(evalu_env, agent)
            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            writer.add_scalar('step_rewards_{}'.format(env_id), evaluate_rewards[-1], global_step=total_steps)

if __name__ == "__main__":
    for i in range(100):
        main("BipedalWalker-v3", seed=10)

# 重写main.py，重点是evaluate部分，以及seed部分，实现实验可复现
# 编写RND模块，然后进行实验