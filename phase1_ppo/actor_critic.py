"""Actor-critic：从学习到的 baseline 过渡到 n-step 更新与 GAE。"""

# TODO：先使用完整轨迹回报训练 critic，并构造固定的 advantage。
# TODO：加入固定长度 rollout，区分终止、截断和 rollout 尾部。
# TODO：实现 GAE 与 return target，分别更新独立的 actor 和 critic。

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from networks import PolicyNetwork, ValueNetwork
from reinforce import collect_episode, compute_returns
from evaluate import evaluate_policy
from experiment_io import RunRecorder


def update_critic(value_net, optimizer, observations, returns):
    values = value_net(observations)

    assert values.shape == returns.shape

    loss = ((values - returns.detach()) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def update_actor(policy, value_net, optimizer, episodes, gamma):
    episode_losses = []

    for episode in episodes:
        observations = torch.as_tensor(
            episode["observations"], dtype=torch.float32
        )
        actions = torch.as_tensor(
            episode["actions"], dtype=torch.long
        )
        returns = torch.as_tensor(
            compute_returns(episode["rewards"], gamma),
            dtype=torch.float32,
        )

        # 用更新前的 critic 给这批动作评分。
        # advantage 是固定权重，不参与反向传播。
        with torch.no_grad():
            values = value_net(observations)
            advantages = returns - values

        logits = policy(observations)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        timesteps = torch.arange(
            len(returns), dtype=torch.float32
        )
        discounts = gamma ** timesteps

        episode_loss = -(
            discounts * log_probs * advantages
        ).sum()

        episode_losses.append(episode_loss)

    loss = torch.stack(episode_losses).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE with a learned MC baseline")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if min(args.updates, args.eval_every, args.eval_episodes) < 1:
        parser.error("updates, eval-every and eval-episodes must be positive")
    torch.manual_seed(args.seed)

    gamma = 0.99
    episodes_per_update = 8
    num_updates = args.updates

    output_root = args.output_root or Path(__file__).resolve().parents[1] / "experiments" / "phase1_baseline"
    recorder = RunRecorder({
        "algorithm": "REINFORCE + learned Monte Carlo baseline (no bootstrap)",
        "env": "CartPole-v1", "seed": args.seed, "gamma": gamma,
        "episodes_per_update": episodes_per_update, "num_updates": num_updates,
        "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
        "actor_updates_per_batch": 1, "critic_updates_per_batch": 1,
        "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
        "eval_seed_start": 10000, "actor_network": "4-64-64-2 Tanh",
        "critic_network": "4-64-64-1 Tanh", "tail_return": "zero at termination or time limit",
    }, output_root)
    print(f"Output directory: {recorder.path}")

    env = gym.make("CartPole-v1")
    env.reset(seed=args.seed)

    policy = PolicyNetwork()
    value_net = ValueNetwork()

    actor_optimizer = torch.optim.Adam(
        policy.parameters(), lr=3e-4
    )
    critic_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=1e-3
    )

    total_steps = 0
    status = "interrupted_or_failed"
    try:
        mean_return, std_return = evaluate_policy(policy, n_episodes=args.eval_episodes)
        print(f"初始评估: {mean_return:.1f} ± {std_return:.1f}")
        recorder.record(policy, actor_optimizer, 0, 0, evaluation=(mean_return, std_return),
                        value_net=value_net, critic_optimizer=critic_optimizer)

        for update_index in range(num_updates):
            episodes = [
                collect_episode(env, policy)
                for _ in range(episodes_per_update)
            ]

            episode_returns = [
                float(episode["rewards"].sum())
                for episode in episodes
            ]

            total_steps += sum(
                len(episode["rewards"])
                for episode in episodes
            )

            observations = torch.as_tensor(
                np.concatenate(
                    [episode["observations"] for episode in episodes],
                    axis=0,
                ),
                dtype=torch.float32,
            )

            returns = torch.as_tensor(
                np.concatenate(
                    [
                        compute_returns(episode["rewards"], gamma)
                        for episode in episodes
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
            )

            # 清除上一轮留下的 critic 梯度，检查 actor 的隔离。
            critic_optimizer.zero_grad(set_to_none=True)

            actor_loss = update_actor(
                policy,
                value_net,
                actor_optimizer,
                episodes,
                gamma,
            )

            assert all(
                parameter.grad is None
                for parameter in value_net.parameters()
            )

            # 清除 actor 梯度，检查 critic 的隔离。
            actor_optimizer.zero_grad(set_to_none=True)

            critic_loss = update_critic(
                value_net,
                critic_optimizer,
                observations,
                returns,
            )

            assert all(
                parameter.grad is None
                for parameter in policy.parameters()
            )

            assert np.isfinite(actor_loss)
            assert np.isfinite(critic_loss)

            print(
                f"update={update_index + 1:3d}, "
                f"steps={total_steps:6d}, "
                f"mean_return={np.mean(episode_returns):7.2f}, "
                f"actor_loss={actor_loss:10.3f}, "
                f"critic_loss={critic_loss:10.3f}"
            )

            evaluation = None
            if (update_index + 1) % args.eval_every == 0 or update_index + 1 == num_updates:
                mean_return, std_return = evaluate_policy(policy, n_episodes=args.eval_episodes)
                evaluation = (mean_return, std_return)
                print(
                    f"独立评估: {mean_return:.1f} ± {std_return:.1f}"
                )

            recorder.record(policy, actor_optimizer, update_index + 1, total_steps,
                            train_mean=float(np.mean(episode_returns)), loss=actor_loss,
                            evaluation=evaluation, value_net=value_net,
                            critic_optimizer=critic_optimizer, critic_loss=critic_loss)
        status = "completed"

    finally:
        env.close()
        recorder.finish(status)
