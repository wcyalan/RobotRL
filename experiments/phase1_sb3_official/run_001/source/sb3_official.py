"""直接使用安装版本 SB3 PPO/MlpPolicy 默认算法参数，无观测/奖励归一化。

python phase1_ppo/sb3_official.py --steps 1024000 --seed 0
只设置 seed、CPU；callback 负责更新后的评估和记录，不重写 PPO/GAE。
这不是 RL Zoo 的环境调参配置。
"""

import argparse
import csv
import hashlib
import io
import json
import time
import warnings
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from experiment_io import FIELDS, plot_metrics
from ppo_continuous import evaluate_continuous_policy
from sb3_continuous import SB3PolicyAdapter, atomic_write


class RecordAfterUpdate(BaseCallback):
    def __init__(self, path, eval_every, eval_episodes):
        super().__init__()
        self.path, self.eval_every, self.eval_episodes = path, eval_every, eval_episodes
        self.rows, self.completed = [], []
        self.best, self.best_steps = -float("inf"), 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.completed.append(float(info["episode"]["r"]))
        return True

    def _on_rollout_start(self):
        # 此时上一批 train() 已完成；rollout_end 则发生在优化之前。
        self.record()

    def _on_training_end(self):
        self.record(final=True)

    def save(self, name):
        buffer = io.BytesIO()
        self.model.save(buffer)
        atomic_write(self.path / f"{name}.zip", buffer.getvalue())

    def write_metrics(self):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(self.rows)
        atomic_write(self.path / "metrics.csv", stream.getvalue().encode("utf-8"))

    def record(self, final=False):
        steps = self.model.num_timesteps
        if self.rows and self.rows[-1]["steps"] == steps:
            return
        row = dict.fromkeys(FIELDS, "")
        row.update(update=steps // self.model.n_steps, steps=steps)
        if self.completed:
            row["train_mean"] = float(np.mean(self.completed))
            self.completed.clear()
        if steps:
            for source, target in [("policy_gradient_loss", "loss"), ("value_loss", "critic_loss"),
                                   ("approx_kl", "approx_kl"), ("clip_fraction", "clip_fraction")]:
                row[target] = float(self.model.logger.name_to_value[f"train/{source}"])
            row["entropy"] = -float(self.model.logger.name_to_value["train/entropy_loss"])
        if steps % self.eval_every == 0 or final:
            row["eval_mean"], row["eval_std"] = evaluate_continuous_policy(
                SB3PolicyAdapter(self.model.policy), n_episodes=self.eval_episodes, seed_start=10000)
            if row["eval_mean"] > self.best:
                self.best, self.best_steps = row["eval_mean"], steps
                self.save("best")
        if not all(np.isfinite(v) for v in row.values() if v != ""):
            raise FloatingPointError("Non-finite training/evaluation metrics")
        self.rows.append(row)
        self.save("last")
        try:
            self.write_metrics()
        except OSError as error:
            warnings.warn(f"CSV write failed; checkpoint saved; retry next update: {error}")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1024000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every-steps", type=int, default=51200)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/phase1_sb3_official")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every_steps < 1 or args.eval_episodes < 1:
        parser.error("steps and evaluation parameters must be positive")
    torch.set_num_threads(1)
    env = Monitor(gym.make("HalfCheetah-v5"))
    # 所有算法和网络超参数均使用 SB3 本身默认值。
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu")
    if args.steps % model.n_steps or args.eval_every_steps % model.n_steps:
        env.close()
        parser.error(f"steps and eval-every-steps must be multiples of {model.n_steps}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = args.output_root / f"run_{index:03d}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            index += 1
    config = dict(algorithm="SB3 PPO official constructor defaults", env="HalfCheetah-v5",
                  seed=args.seed, steps=args.steps, rollout_steps=model.n_steps,
                  observation_normalization=False, reward_normalization=False,
                  evaluation_action="clipped Gaussian mean", eval_seed_start=10000,
                  eval_every_steps=args.eval_every_steps, eval_episodes=args.eval_episodes,
                  torch_threads=1, device="cpu", n_envs=1,
                  versions=dict(sb3=sb3.__version__, gymnasium=gym.__version__,
                                mujoco=mujoco.__version__, torch=str(torch.__version__), numpy=np.__version__),
                  hyperparameters={key:getattr(model,key) for key in [
                      "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
                      "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "use_sde"]},
                  clip_range=model.clip_range(1.0), clip_range_vf=None,
                  net_arch=model.policy.net_arch, activation=str(model.policy.activation_fn),
                  ortho_init=model.policy.ortho_init, optimizer=str(type(model.policy.optimizer)),
                  optimizer_eps=model.policy.optimizer.defaults["eps"])
    (path / "source").mkdir()
    config["source_sha256"] = {}
    for source in Path(__file__).parent.glob("*.py"):
        data = source.read_bytes()
        (path / "source" / source.name).write_bytes(data)
        config["source_sha256"][source.name] = hashlib.sha256(data).hexdigest()
    atomic_write(path / "config.json", json.dumps(config, indent=2).encode("utf-8"))
    callback = RecordAfterUpdate(path, args.eval_every_steps, args.eval_episodes)
    started, status = time.perf_counter(), "interrupted_or_failed"
    print(f"Output directory: {path}", flush=True)
    try:
        model.learn(args.steps, callback=callback)
        status = "completed"
    finally:
        env.close()
        atomic_write(path / "summary.json", json.dumps(dict(
            status=status, steps=model.num_timesteps, best_eval_mean=callback.best,
            best_steps=callback.best_steps, elapsed_seconds=time.perf_counter()-started,
            checkpoint_note="Standard SB3 zip; no normalization state required."), indent=2).encode("utf-8"))
        if callback.rows:
            callback.write_metrics()
            plot_metrics(callback.rows, path / "curve.png")


if __name__ == "__main__":
    main()
