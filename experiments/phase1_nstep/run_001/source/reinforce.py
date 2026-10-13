"""REINFORCE：完整轨迹采样、reward-to-go 与策略更新。"""

# TODO：采集完整 episode，保存 observation、action、reward 和结束标记。
# TODO：逐条轨迹倒序计算 reward-to-go，并用手算例子验证。
# TODO：实现一次策略更新，再连接训练循环与日志。

import argparse

import gymnasium as gym
import numpy as np
import torch

from networks import PolicyNetwork
from evaluate import evaluate_policy
from experiment_io import RunRecorder


def collect_episode(env, policy):
    observations = []
    actions = []
    rewards = []

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        observations.append(obs.copy())

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

        obs = next_obs

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
    }


def compute_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def update_policy(policy, optimizer, episodes, gamma):
    episode_losses = []

    for episode in episodes:
        returns = compute_returns(episode["rewards"], gamma)

        observations = torch.as_tensor(
            episode["observations"], dtype=torch.float32
        )
        actions = torch.as_tensor(
            episode["actions"], dtype=torch.long
        )
        returns = torch.as_tensor(
            returns, dtype=torch.float32
        )

        logits = policy(observations)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        timesteps = torch.arange(
            len(returns), dtype=torch.float32
        )
        discounts = gamma ** timesteps

        episode_loss = -(discounts * log_probs * returns).sum()
        episode_losses.append(episode_loss)

    loss = torch.stack(episode_losses).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save REINFORCE on CartPole")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if min(args.updates, args.eval_every, args.eval_episodes) < 1:
        parser.error("updates, eval-every and eval-episodes must be positive")
    torch.manual_seed(0)

    gamma = 0.99
    episodes_per_update = 8
    num_updates = args.updates

    recorder = RunRecorder({
        "env": "CartPole-v1", "seed": 0, "gamma": gamma,
        "episodes_per_update": episodes_per_update, "num_updates": num_updates,
        "learning_rate": 3e-4, "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes, "eval_seed_start": 10000,
        "network": "4-64-64-2, Tanh", "tail_return": "zero at termination or time limit",
    }, args.output_root)
    print(f"Output directory: {recorder.path}")

    env = gym.make("CartPole-v1")
    env.reset(seed=0)

    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=3e-4
    )

    total_steps = 0
    status = "interrupted_or_failed"
    try:
        mean_return, std_return = evaluate_policy(policy, n_episodes=args.eval_episodes)
        print(f"初始评估: {mean_return:.1f} ± {std_return:.1f}")
        recorder.record(policy, optimizer, 0, 0, evaluation=(mean_return, std_return))
        for update_index in range(num_updates):
            episodes = []
            episode_returns = []

            # 采样整个批次期间，策略参数保持不变。
            for _ in range(episodes_per_update):
                episode = collect_episode(env, policy)
                episodes.append(episode)

                episode_returns.append(
                    float(episode["rewards"].sum())
                )
                total_steps += len(episode["rewards"])

            # 收集完 8 局之后，只更新一次。
            loss = update_policy(
                policy, optimizer, episodes, gamma
            )

            assert np.isfinite(loss), "loss 出现 NaN 或 Inf"

            print(
                f"update={update_index + 1:3d}, "
                f"steps={total_steps:6d}, "
                f"mean_return={np.mean(episode_returns):7.2f}, "
                f"loss={loss:10.3f}"
            )

            evaluation = None
            if (update_index + 1) % args.eval_every == 0 or update_index + 1 == num_updates:
                mean_return, std_return = evaluate_policy(policy, n_episodes=args.eval_episodes)
                evaluation = (mean_return, std_return)
                print(
                    f"独立评估: {mean_return:.1f} ± {std_return:.1f}"
                )

            recorder.record(policy, optimizer, update_index + 1, total_steps,
                            train_mean=float(np.mean(episode_returns)), loss=loss,
                            evaluation=evaluation)
        status = "completed"

    finally:
        env.close()
        recorder.finish(status)
