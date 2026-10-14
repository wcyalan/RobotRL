"""PPO-Clip：在 actor-critic 与 GAE 的基础上加入近端策略更新。"""

# TODO：采样时保存 old log-prob，更新前验证 ratio 约等于 1。
# TODO：实现 clipped surrogate 与多 epoch minibatch 更新。
# TODO：记录 KL、clip fraction、entropy 和评估回报。

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actor_critic import collect_rollout, prepare_rollout_targets
from evaluate import evaluate_policy
from experiment_io import RunRecorder
from networks import PolicyNetwork, ValueNetwork


def compute_ppo_actor_loss(
    policy,
    observations,
    actions,
    old_log_probs,
    advantages,
    clip_coef=0.2,
):
    # 当前策略对这批历史动作的 log-prob。
    logits = policy(observations)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)

    # 旧概率和优势在本批数据的所有更新中保持固定。
    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()

    assert (
        new_log_probs.shape
        == old_log_probs.shape
        == advantages.shape
    )

    ratio = torch.exp(new_log_probs - old_log_probs)

    objective = ratio * advantages
    clipped_objective = torch.clamp(
        ratio,
        1.0 - clip_coef,
        1.0 + clip_coef,
    ) * advantages

    loss = -torch.minimum(
        objective, clipped_objective
    ).mean()

    return loss


def update_ppo(
    policy,
    value_net,
    actor_optimizer,
    critic_optimizer,
    observations,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_coef=0.2,
    update_epochs=4,
    minibatch_size=32,
):
    # 整个更新过程中，旧概率、优势和 critic 标签保持固定。
    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()
    returns = returns.detach()

    batch_size = observations.shape[0]

    if batch_size == 0 or update_epochs < 1 or minibatch_size < 1:
        raise ValueError("batch、epoch 和 minibatch 大小必须为正")

    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    sample_count = 0

    for _ in range(update_epochs):
        # 每轮重新打乱索引；所有数据用同一组索引取样。
        indices = torch.randperm(
            batch_size, device=observations.device
        )

        for start in range(0, batch_size, minibatch_size):
            mb_indices = indices[start:start + minibatch_size]

            mb_observations = observations[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # 更新 actor。
            actor_loss = compute_ppo_actor_loss(
                policy,
                mb_observations,
                mb_actions,
                mb_old_log_probs,
                mb_advantages,
                clip_coef=clip_coef,
            )

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 更新 critic。
            values = value_net(mb_observations)
            assert values.shape == mb_returns.shape

            critic_loss = ((values - mb_returns) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # 按样本数汇总，兼容最后一个 minibatch 不满的情况。
            n = mb_indices.numel()
            actor_loss_sum += actor_loss.item() * n
            critic_loss_sum += critic_loss.item() * n
            sample_count += n

    return (
        actor_loss_sum / sample_count,
        critic_loss_sum / sample_count,
    )


@torch.no_grad()
def compute_ppo_diagnostics(
    policy,
    observations,
    actions,
    old_log_probs,
    clip_coef=0.2,
):
    logits = policy(observations)
    dist = torch.distributions.Categorical(logits=logits)

    new_log_probs = dist.log_prob(actions)
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()

    approx_kl = ((ratio - 1.0) - log_ratio).mean()

    clip_fraction = (
        (ratio - 1.0).abs() > clip_coef
    ).float().mean()

    entropy = dist.entropy().mean()

    return {
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "entropy": entropy.item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-Clip on CartPole")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-root", default=None)
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

    policy = PolicyNetwork()
    value_net = ValueNetwork()

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
        / "phase1_ppo"
    )

    recorder = RunRecorder(
        {
            "algorithm": "PPO-Clip",
            "env": "CartPole-v1",
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
            "advantage_normalization": False,
            "actor_network": "4-64-64-2 Tanh",
            "critic_network": "4-64-64-1 Tanh",
            "train_mean_definition": (
                "full episodes completed within this rollout; blank if none"
            ),
        },
        output_root,
    )

    print(f"Output directory: {recorder.path}")

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    total_steps = 0
    running_episode_return = 0.0
    status = "interrupted_or_failed"

    try:
        # 更新前先评估一次，作为起点。
        evaluation = evaluate_policy(
            policy, n_episodes=args.eval_episodes
        )
        print(
            f"初始评估: {evaluation[0]:.1f} ± {evaluation[1]:.1f}"
        )

        recorder.record(
            policy,
            actor_optimizer,
            0,
            0,
            evaluation=evaluation,
            value_net=value_net,
            critic_optimizer=critic_optimizer,
        )

        for update in range(1, args.updates + 1):
            # A. 用当前策略采集新 rollout。
            rollout, obs = collect_rollout(
                env, policy, obs, args.rollout_steps
            )
            total_steps += len(rollout["rewards"])

            # B. 更新网络前，计算并冻结 GAE 与回报标签。
            observations, actions, returns, advantages = (
                prepare_rollout_targets(
                    value_net,
                    rollout,
                    gamma,
                    gae_lambda=args.gae_lambda,
                )
            )

            old_log_probs = torch.as_tensor(
                rollout["old_log_probs"],
                dtype=torch.float32,
            )

            # C. 对同一批数据执行多个 epoch 的 PPO 更新。
            actor_loss, critic_loss = update_ppo(
                policy,
                value_net,
                actor_optimizer,
                critic_optimizer,
                observations,
                actions,
                old_log_probs,
                advantages,
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

            # E. 定期独立评估。
            evaluation = None

            if (
                update % args.eval_every == 0
                or update == args.updates
            ):
                evaluation = evaluate_policy(
                    policy,
                    n_episodes=args.eval_episodes,
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
                evaluation=evaluation,
                value_net=value_net,
                critic_optimizer=critic_optimizer,
            )

        status = "completed"

    finally:
        env.close()
        recorder.finish(status)
