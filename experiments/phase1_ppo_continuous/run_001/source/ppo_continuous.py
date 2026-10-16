"""单环境连续动作 PPO：原始高斯动作用于更新，裁剪动作交给环境。"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actor_critic import compute_gae
from experiment_io import RunRecorder
from networks import GaussianPolicyNetwork, ValueNetwork
from ppo import update_ppo, compute_ppo_diagnostics

class ObservationNormalizer:
    """逐维累计总体方差；normalize 只读取统计量，update 显式更新。"""

    def __init__(self, obs_dim, epsilon=1e-8, clip=10.0):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon
        self.clip = clip

    def update(self, observations):
        batch = np.asarray(observations, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1:] != self.mean.shape or len(batch) == 0:
            raise ValueError("observations must be a nonempty [N, obs_dim] batch")
        if not np.isfinite(batch).all():
            raise ValueError("observations contain non-finite values")
        batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)
        if self.count == 0:
            self.mean, self.var = batch_mean, batch_var
            self.count = len(batch)
            return
        total = self.count + len(batch)
        delta = batch_mean - self.mean
        m2 = (self.var * self.count + batch_var * len(batch)
              + delta ** 2 * self.count * len(batch) / total)
        self.mean = self.mean + delta * len(batch) / total
        self.var = np.maximum(m2 / total, 0.0)
        self.count = total

    def normalize(self, observations):
        normalized = (np.asarray(observations) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        # Tensor / Python 标量兼容 weights_only=True 的 checkpoint 加载。
        return {"mean": torch.from_numpy(self.mean.copy()),
                "var": torch.from_numpy(self.var.copy()), "count": self.count,
                "epsilon": self.epsilon, "clip": self.clip}

    def load_state_dict(self, state):
        mean = state["mean"].cpu().numpy().copy()
        var = state["var"].cpu().numpy().copy()
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("normalizer observation dimensions do not match")
        self.mean, self.var = mean, var
        self.count = int(state["count"])
        self.epsilon, self.clip = float(state["epsilon"]), float(state["clip"])


def collect_continuous_rollout(env, policy, obs, n_steps, obs_normalizer=None):
    observations = []
    raw_observations = []
    actions = []
    old_log_probs = []
    rewards = []
    next_observations = []
    terminated_flags = []
    truncated_flags = []

    for _ in range(n_steps):
        raw_observations.append(obs.copy())
        network_obs = obs_normalizer.normalize(obs) if obs_normalizer else obs.copy()
        observations.append(network_obs)

        obs_tensor = torch.as_tensor(
            network_obs, dtype=torch.float32
        )

        with torch.no_grad():
            dist = policy.get_distribution(obs_tensor)

            # 原始动作和 log-prob 必须对应。
            action_tensor = dist.sample()
            old_log_prob = dist.log_prob(action_tensor).item()

        raw_action = action_tensor.cpu().numpy().copy()

        # 裁剪只影响环境执行，不覆盖原始动作。
        env_action = np.clip(
            raw_action,
            env.action_space.low,
            env.action_space.high,
        )

        next_obs, reward, terminated, truncated, _ = env.step(
            env_action
        )

        actions.append(raw_action)
        old_log_probs.append(old_log_prob)
        rewards.append(reward)

        # 保存 reset 前真正的下一状态，供 GAE bootstrap 使用。
        next_observations.append(
            obs_normalizer.normalize(next_obs) if obs_normalizer else next_obs.copy()
        )
        terminated_flags.append(terminated)
        truncated_flags.append(truncated)

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    rollout = {
        "raw_observations": np.asarray(raw_observations, dtype=np.float32),
        "observations": np.asarray(
            observations, dtype=np.float32
        ),
        "actions": np.asarray(
            actions, dtype=np.float32
        ),
        "old_log_probs": np.asarray(
            old_log_probs, dtype=np.float32
        ),
        "rewards": np.asarray(
            rewards, dtype=np.float32
        ),
        "next_observations": np.asarray(
            next_observations, dtype=np.float32
        ),
        "terminated": np.asarray(
            terminated_flags, dtype=np.bool_
        ),
        "truncated": np.asarray(
            truncated_flags, dtype=np.bool_
        ),
    }

    return rollout, obs


@torch.no_grad()
def prepare_continuous_targets(
    value_net,
    rollout,
    gamma=0.99,
    gae_lambda=0.95,
):
    observations = torch.as_tensor(
        rollout["observations"], dtype=torch.float32
    )
    next_observations = torch.as_tensor(
        rollout["next_observations"], dtype=torch.float32
    )

    # 连续动作是浮点向量。
    actions = torch.as_tensor(
        rollout["actions"], dtype=torch.float32
    )
    old_log_probs = torch.as_tensor(
        rollout["old_log_probs"], dtype=torch.float32
    )

    # 使用更新前的 critic 生成固定标签。
    values = value_net(observations)
    next_values = value_net(next_observations)

    advantages_np, returns_np = compute_gae(
        rewards=rollout["rewards"],
        values=values.numpy(),
        next_values=next_values.numpy(),
        terminated=rollout["terminated"],
        truncated=rollout["truncated"],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    advantages = torch.as_tensor(
        advantages_np, dtype=torch.float32
    )
    returns = torch.as_tensor(
        returns_np, dtype=torch.float32
    )

    batch_size = observations.shape[0]

    assert actions.ndim == 2
    assert actions.shape[0] == batch_size
    assert (
        old_log_probs.shape
        == advantages.shape
        == returns.shape
        == (batch_size,)
    )

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()

    return (
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
    )


@torch.no_grad()
def evaluate_continuous_policy(
    policy,
    env_id="HalfCheetah-v5",
    n_episodes=5,
    seed_start=10000,
    obs_normalizer=None,
):
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    env = gym.make(env_id)
    episode_returns = []

    was_training = policy.training
    policy.eval()

    try:
        for episode_index in range(n_episodes):
            obs, _ = env.reset(
                seed=seed_start + episode_index
            )
            episode_return = 0.0

            while True:
                network_obs = obs_normalizer.normalize(obs) if obs_normalizer else obs
                obs_tensor = torch.as_tensor(
                    network_obs, dtype=torch.float32
                )

                # 确定性评估：直接使用高斯分布的均值。
                mean, _ = policy(obs_tensor)
                raw_action = mean.cpu().numpy()

                env_action = np.clip(
                    raw_action,
                    env.action_space.low,
                    env.action_space.high,
                )

                obs, reward, terminated, truncated, _ = (
                    env.step(env_action)
                )
                episode_return += float(reward)

                if terminated or truncated:
                    break

            episode_returns.append(episode_return)

    finally:
        env.close()
        policy.train(was_training)

    return (
        float(np.mean(episode_returns)),
        float(np.std(episode_returns)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous-action PPO-Clip")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    args = parser.parse_args()

    if min(
        args.updates,
        args.rollout_steps,
        args.update_epochs,
        args.minibatch_size,
        args.eval_every,
        args.eval_episodes,
    ) < 1:
        parser.error("更新次数、采样步数、batch 大小和评估参数必须为正")

    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("gae-lambda 必须在 [0, 1] 内")

    if not 0.0 < args.clip_coef < 1.0:
        parser.error("clip-coef 必须在 (0, 1) 内")

    torch.manual_seed(args.seed)
    gamma = 0.99

    env = gym.make(args.env_id)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicyNetwork(obs_dim, action_dim)
    value_net = ValueNetwork(obs_dim)
    obs_normalizer = ObservationNormalizer(obs_dim)

    actor_optimizer = torch.optim.Adam(
        policy.parameters(), lr=3e-4
    )
    critic_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=1e-3
    )

    output_root = (
        args.output_root
        or Path(__file__).resolve().parents[1]
        / "experiments"
        / "phase1_ppo_continuous"
    )

    recorder = RunRecorder(
        {
            "algorithm": "PPO-Clip continuous",
            "env": args.env_id,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "action_handling": "store raw Gaussian action; clip only for env.step",
            "evaluation_action": "clipped Gaussian mean",
            "observation_normalization": True,
            "observation_normalization_scope": "frozen during rollout and PPO; update from raw rollout after optimization; frozen evaluation",
            "observation_normalization_epsilon": obs_normalizer.epsilon,
            "observation_normalization_clip": obs_normalizer.clip,
            "seed": args.seed,
            "gamma": gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "rollout_steps": args.rollout_steps,
            "num_updates": args.updates,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 1e-3,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "eval_seed_start": 10000,
            "advantage_normalization": True,
            "advantage_normalization_scope": "once per rollout, population std, epsilon=1e-8",
            "actor_network": f"{obs_dim}-64-64-{action_dim} Tanh, Gaussian",
            "critic_network": f"{obs_dim}-64-64-1 Tanh",
            "train_mean_definition": (
                "full episodes completed within this rollout; blank if none"
            ),
        },
        output_root,
    )

    print(f"Output directory: {recorder.path}")

    obs, _ = env.reset(seed=args.seed)

    total_steps = 0
    running_episode_return = 0.0
    status = "interrupted_or_failed"

    try:
        # 更新前先评估一次，作为起点。
        evaluation = evaluate_continuous_policy(
            policy, env_id=args.env_id, n_episodes=args.eval_episodes,
            obs_normalizer=obs_normalizer,
        )
        print(
            f"初始评估: {evaluation[0]:.1f} ± {evaluation[1]:.1f}"
        )

        recorder.record(
            policy,
            actor_optimizer,
            0,
            0,
            observation_normalizer_state=obs_normalizer.state_dict(),
            evaluation=evaluation,
            value_net=value_net,
            critic_optimizer=critic_optimizer,
        )

        for update in range(1, args.updates + 1):
            # A. 用当前策略采集新 rollout。
            rollout, obs = collect_continuous_rollout(
                env, policy, obs, args.rollout_steps, obs_normalizer=obs_normalizer
            )
            total_steps += len(rollout["rewards"])

            # B. 更新网络前，计算并冻结 GAE 与回报标签。
            observations, actions, old_log_probs, advantages, returns = (
                prepare_continuous_targets(
                    value_net,
                    rollout,
                    gamma,
                    gae_lambda=args.gae_lambda,
                )
            )

            # 只标准化 actor 的权重；critic 仍使用原始 GAE 回报标签。
            # 整批 rollout 计算一次，所有 epoch / minibatch 共用。
            actor_advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

            # C. 对同一批数据执行多个 epoch 的 PPO 更新。
            actor_loss, critic_loss = update_ppo(
                policy,
                value_net,
                actor_optimizer,
                critic_optimizer,
                observations,
                actions,
                old_log_probs,
                actor_advantages,
                returns,
                clip_coef=args.clip_coef,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
            )

            diagnostics = compute_ppo_diagnostics(
                policy,
                observations,
                actions,
                old_log_probs,
                clip_coef=args.clip_coef,
            )

            print(
                f"approx_kl={diagnostics['approx_kl']:.6f}, "
                f"clip_fraction={diagnostics['clip_fraction']:.3f}, "
                f"entropy={diagnostics['entropy']:.3f}"
            )

            if not (
                np.isfinite(actor_loss)
                and np.isfinite(critic_loss)
                and all(np.isfinite(value) for value in diagnostics.values())
            ):
                raise FloatingPointError("训练 loss 出现非有限值")

            # D. 统计完整回合的原始奖励。
            # 一个回合可能跨越多批 rollout，所以累计量不能每批清零。
            episode_returns = []

            for reward, terminated, truncated in zip(
                rollout["rewards"],
                rollout["terminated"],
                rollout["truncated"],
            ):
                running_episode_return += float(reward)

                if terminated or truncated:
                    episode_returns.append(running_episode_return)
                    running_episode_return = 0.0

            train_mean = (
                float(np.mean(episode_returns))
                if episode_returns
                else None
            )

            train_text = (
                f"{train_mean:.2f}"
                if train_mean is not None
                else "n/a"
            )

            print(
                f"update={update:3d}, "
                f"steps={total_steps:6d}, "
                f"mean_return={train_text}, "
                f"actor_loss={actor_loss:.3f}, "
                f"critic_loss={critic_loss:.3f}"
            )

            # 此批采样和所有 PPO epoch 都使用同一套输入尺度。
            # 更新统计量供下一批及评估使用；上面的 KL 只度量固定输入上的网络更新。
            obs_normalizer.update(rollout["raw_observations"])

            # E. 定期独立评估；不从评估观测更新统计量。
            evaluation = None

            if (
                update % args.eval_every == 0
                or update == args.updates
            ):
                evaluation = evaluate_continuous_policy(
                    policy,
                    env_id=args.env_id,
                    n_episodes=args.eval_episodes,
                    obs_normalizer=obs_normalizer,
                )
                print(
                    f"独立评估: "
                    f"{evaluation[0]:.1f} ± {evaluation[1]:.1f}"
                )

            # F. 保存模型、训练指标及评估结果。
            recorder.record(
                policy,
                actor_optimizer,
                update,
                total_steps,
                train_mean=train_mean,
                loss=actor_loss,
                critic_loss=critic_loss,
                diagnostics=diagnostics,
                observation_normalizer_state=obs_normalizer.state_dict(),
                evaluation=evaluation,
                value_net=value_net,
                critic_optimizer=critic_optimizer,
            )

        status = "completed"

    finally:
        env.close()
        recorder.finish(status)
