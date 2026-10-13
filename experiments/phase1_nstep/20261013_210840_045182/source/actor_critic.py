"""Actor-critic：从学习到的 baseline 过渡到 n-step 更新与 GAE。"""

# Fixed rollouts with n-step bootstrap; GAE is the next learning step.

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from networks import PolicyNetwork, ValueNetwork
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


def update_actor(policy, optimizer, observations, actions, advantages):
    """One on-policy update, averaged over rollout transitions."""
    logits = policy(observations)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    assert log_probs.shape == advantages.shape
    loss = -(log_probs * advantages.detach()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def prepare_rollout_targets(value_net, rollout, gamma):
    """Freeze both targets before either network is updated (CPU version)."""
    observations = torch.as_tensor(rollout["observations"], dtype=torch.float32)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long)
    next_observations = torch.as_tensor(rollout["next_observations"], dtype=torch.float32)
    old_values = value_net(observations)
    next_values = value_net(next_observations)
    returns = torch.as_tensor(compute_bootstrapped_returns(
        rollout["rewards"], next_values.numpy(), rollout["terminated"],
        rollout["truncated"], gamma,
    ), dtype=torch.float32)
    advantages = returns - old_values
    assert returns.shape == advantages.shape == actions.shape
    assert torch.isfinite(returns).all() and torch.isfinite(advantages).all()
    return observations, actions, returns, advantages


def collect_rollout(env, policy, obs, n_steps):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminated_flags = []
    truncated_flags = []

    for _ in range(n_steps):
        observations.append(obs.copy())

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

        # 必须保存 reset 前的真正下一状态。
        next_observations.append(next_obs.copy())
        terminated_flags.append(terminated)
        truncated_flags.append(truncated)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    rollout = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(
            next_observations, dtype=np.float32
        ),
        "terminated": np.asarray(terminated_flags, dtype=np.bool_),
        "truncated": np.asarray(truncated_flags, dtype=np.bool_),
    }

    return rollout, obs


def compute_bootstrapped_returns(
    rewards,
    next_values,
    terminated,
    truncated,
    gamma,
):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        if terminated[t]:
            # 真正终止：没有后续回报。
            future_return = 0.0

        elif truncated[t] or t == len(rewards) - 1:
            # 截断边界或 rollout 尾部：使用真正末状态的预测。
            future_return = next_values[t]

        else:
            # 同一 episode 内：接上刚算好的下一步回报。
            future_return = running_return

        running_return = rewards[t] + gamma * future_return
        returns[t] = running_return

    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-rollout n-step actor-critic on CartPole")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if min(args.updates, args.eval_every, args.eval_episodes, args.rollout_steps) < 1:
        parser.error("updates, rollout-steps, eval-every and eval-episodes must be positive")
    torch.manual_seed(args.seed)

    gamma = 0.99
    num_updates = args.updates

    output_root = args.output_root or Path(__file__).resolve().parents[1] / "experiments" / "phase1_nstep"
    recorder = RunRecorder({
        "algorithm": "Fixed-rollout n-step actor-critic (no GAE)",
        "env": "CartPole-v1", "seed": args.seed, "gamma": gamma,
        "rollout_steps": args.rollout_steps, "num_updates": num_updates,
        "actor_loss_weighting": "uniform transition mean, no outer gamma^t",
        "train_mean_definition": "full episodes completed within this rollout; blank if none",
        "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
        "actor_updates_per_batch": 1, "critic_updates_per_batch": 1,
        "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
        "eval_seed_start": 10000, "actor_network": "4-64-64-2 Tanh",
        "critic_network": "4-64-64-1 Tanh", "tail_return": "zero at termination; bootstrap at truncation or rollout end",
    }, output_root)
    print(f"Output directory: {recorder.path}")

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    policy = PolicyNetwork()
    value_net = ValueNetwork()

    actor_optimizer = torch.optim.Adam(
        policy.parameters(), lr=3e-4
    )
    critic_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=1e-3
    )

    total_steps = 0
    running_episode_return = 0.0
    status = "interrupted_or_failed"
    try:
        mean_return, std_return = evaluate_policy(policy, n_episodes=args.eval_episodes)
        print(f"初始评估: {mean_return:.1f} ± {std_return:.1f}")
        recorder.record(policy, actor_optimizer, 0, 0, evaluation=(mean_return, std_return),
                        value_net=value_net, critic_optimizer=critic_optimizer)

        for update_index in range(num_updates):
            rollout, obs = collect_rollout(env, policy, obs, args.rollout_steps)
            total_steps += len(rollout["rewards"])

            # Episode returns must span rollout boundaries; do not log a partial sum as an episode.
            episode_returns = []
            for reward, terminated, truncated in zip(
                rollout["rewards"], rollout["terminated"], rollout["truncated"]
            ):
                running_episode_return += float(reward)
                if terminated or truncated:
                    episode_returns.append(running_episode_return)
                    running_episode_return = 0.0
            train_mean = float(np.mean(episode_returns)) if episode_returns else None

            observations, actions, returns, advantages = prepare_rollout_targets(
                value_net, rollout, gamma
            )

            # 清除上一轮留下的 critic 梯度，检查 actor 的隔离。
            critic_optimizer.zero_grad(set_to_none=True)

            actor_loss = update_actor(
                policy, actor_optimizer, observations, actions, advantages,
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

            train_text = f"{train_mean:.2f}" if train_mean is not None else "n/a"
            print(
                f"update={update_index + 1:3d}, "
                f"steps={total_steps:6d}, "
                f"completed_episodes={len(episode_returns)}, mean_return={train_text}, "
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
                            train_mean=train_mean, loss=actor_loss,
                            evaluation=evaluation, value_net=value_net,
                            critic_optimizer=critic_optimizer, critic_loss=critic_loss)
        status = "completed"

    finally:
        env.close()
        recorder.finish(status)
