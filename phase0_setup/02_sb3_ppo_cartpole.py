"""Phase 0 - 任务 2：SB3 PPO 训练 CartPole（对照组）。

用现成的 stable-baselines3 PPO 把 CartPole-v1 训到满分，作为阶段 1 手写
PPO 的性能基准。达到 500 分自动停止。产物：
    experiments/phase0_sb3_ppo/ppo_cartpole.zip   训好的模型（本地保留，不进 git）
    experiments/phase0_sb3_ppo/curve.png          训练曲线（进 git）

用法:
    python 02_sb3_ppo_cartpole.py                # 训练 + 评估 + 存曲线
    python 02_sb3_ppo_cartpole.py --steps 50000  # 自定义步数上限
    python 02_sb3_ppo_cartpole.py --watch        # 加载模型弹窗观看
"""
import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "experiments" / "phase0_sb3_ppo"
TOTAL_TIMESTEPS_DEFAULT = 200_000
SUCCESS_WINDOW = 20


class RewardRecordAndEarlyStop(BaseCallback):
    """记录每局 return；最近 20 局全部 500 分时提前停止训练。"""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:  # Monitor wrapper 提供的字段
                self.episode_rewards.append(info["episode"]["r"])
        if len(self.episode_rewards) >= SUCCESS_WINDOW:
            recent = self.episode_rewards[-SUCCESS_WINDOW:]
            if np.mean(recent) >= 500.0:
                print(f"最近 {SUCCESS_WINDOW} 局均 500 分，提前停止 "
                      f"(timestep={self.num_timesteps})")
                return False
        return True


def evaluate(model, n_episodes=20):
    env = Monitor(gym.make("CartPole-v1"))
    returns = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=20260101 + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return returns


def plot_curve(rewards, steps_used):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rewards, lw=0.8, alpha=0.4, label="episode return")
    if len(rewards) >= 20:
        k = min(50, len(rewards) // 2)
        smooth = np.convolve(rewards, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k - 1, len(rewards)), smooth, lw=2,
                label=f"moving avg ({k})")
    ax.axhline(500, color="gray", ls="--", lw=1, label="max = 500")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(f"SB3 PPO on CartPole-v1 ({steps_used} steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curve.png", dpi=150)
    print(f"训练曲线: {OUT_DIR / 'curve.png'}")


def train(args):
    env = Monitor(gym.make("CartPole-v1"))
    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    callback = RewardRecordAndEarlyStop()
    model.learn(total_timesteps=args.steps, callback=callback)
    env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "ppo_cartpole")

    returns = evaluate(model)
    print(f"\n评估（确定性策略, 20 局）: {np.mean(returns):.1f} ± "
          f"{np.std(returns):.1f}（满分 500）")
    plot_curve(callback.episode_rewards, model.num_timesteps)
    print(f"模型: {OUT_DIR / 'ppo_cartpole.zip'}（本地保留，不进 git）")


def watch(args):
    model = PPO.load(OUT_DIR / "ppo_cartpole")
    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(5):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS_DEFAULT)
    parser.add_argument("--watch", action="store_true", help="加载模型观看（先训练过）")
    args = parser.parse_args()
    watch(args) if args.watch else train(args)


if __name__ == "__main__":
    main()
