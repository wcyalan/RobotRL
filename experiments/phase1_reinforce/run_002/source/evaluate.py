"""独立评估：在独立环境中评估策略，记录回报并支持回放。"""

# TODO：固定评估 seeds，使用确定性动作，不更新网络。
# TODO：汇总每局回报、均值、标准差与达到 500 分的局数。
# TODO：支持加载已保存的模型进行评估和渲染。

import gymnasium as gym
import numpy as np
import torch


def evaluate_policy(policy, n_episodes=20, seed_start=10000, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    episode_returns = []

    was_training = policy.training
    policy.eval()

    try:
        with torch.no_grad():
            for episode_index in range(n_episodes):
                obs, info = env.reset(seed=seed_start + episode_index)
                terminated = False
                truncated = False
                total_reward = 0.0

                while not (terminated or truncated):
                    obs_tensor = torch.as_tensor(
                        obs, dtype=torch.float32
                    )

                    logits = policy(obs_tensor)
                    action = logits.argmax(dim=-1).item()

                    obs, reward, terminated, truncated, info = (
                        env.step(action)
                    )
                    total_reward += reward

                episode_returns.append(total_reward)

    finally:
        env.close()
        policy.train(was_training)

    return (
        float(np.mean(episode_returns)),
        float(np.std(episode_returns)),
    )


if __name__ == "__main__":
    import argparse
    from networks import PolicyNetwork

    parser = argparse.ArgumentParser(description="Evaluate a saved REINFORCE policy")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy = PolicyNetwork()
    policy.load_state_dict(checkpoint["policy_state_dict"])
    mean, std = evaluate_policy(policy, args.episodes, args.seed_start, args.render)
    print(f"Checkpoint update={checkpoint['update']}, steps={checkpoint['steps']}")
    print(f"Evaluation ({args.episodes} episodes, seeds starting at {args.seed_start}): {mean:.1f} +/- {std:.1f}")
