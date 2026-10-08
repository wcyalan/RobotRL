"""Phase 0 - 任务 1：CartPole 随机策略基线。

每步等概率随机选左/右，看平均能撑多少步（期望 20~30 步，满分 500）。
目的是验证工具链 + 建立"没学过的策略长什么样"的直觉基线。

用法:
    python 01_random_cartpole.py             # 无渲染，跑 10 个 episode
    python 01_random_cartpole.py --render    # 弹窗实时观看
    python 01_random_cartpole.py --episodes 50
"""
import argparse

import gymnasium as gym
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="弹窗实时渲染")
    args = parser.parse_args()

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)

    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        total, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)

    env.close()
    print(f"\n随机策略: {args.episodes} 局平均 return = "
          f"{np.mean(returns):.1f} ± {np.std(returns):.1f}（满分 500）")


if __name__ == "__main__":
    main()
