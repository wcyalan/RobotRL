"""SB3 PPO 对照：复用手写实现的 observation 统计规则与评估函数。

短验证：python phase1_ppo/sb3_continuous.py --updates 10 --seed 0
约 1M 步：--updates 1000 --eval-every 50 --seed 0（另一组使用 seed 1）
回放：python phase1_ppo/replay_continuous.py <run目录>/last.zip
新模型 zip 内含归一化统计量；旧模型还需同名 _normalizer.pt。保留 config.json。
恢复：--resume <run目录>/last.zip --updates 1000（累计目标，环境/RNG 重启）
"""

import argparse
import csv
import hashlib
import json
import io
import os
import tempfile
import time
import warnings
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from experiment_io import FIELDS, plot_metrics
from ppo_continuous import ObservationNormalizer, evaluate_continuous_policy


def atomic_write(path, data):
    """同目录临时文件写完再替换；短暂写盘失败重试，保留旧文件。"""
    path = Path(path)
    for attempt in range(4):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(0.2 * (attempt + 1))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def read_normalizer(model, checkpoint):
    state = getattr(model, "saved_normalizer_state", None)
    if state is None:  # 兼容之前保存的 zip + pt。
        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint.with_name(checkpoint.stem + "_normalizer.pt"),
                           weights_only=True, map_location="cpu")
    if int(state["count"]) != model.num_timesteps:
        raise ValueError("模型步数与归一化统计量不匹配，拒绝加载")
    return state


class FrozenRolloutObservation(gym.Wrapper):
    """采样期间只读取统计量；输入原始观测留到 PPO 更新后累计。"""

    def __init__(self, env):
        super().__init__(env)
        self.normalizer = ObservationNormalizer(env.observation_space.shape[0])
        self.raw_obs = None
        self.pending = []
        self.completed_returns = []
        self.episode_return = 0.0
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=env.observation_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        self.raw_obs, info = self.env.reset(**kwargs)
        self.episode_return = 0.0
        return self.normalizer.normalize(self.raw_obs), info

    def step(self, action):
        self.pending.append(self.raw_obs.astype(np.float32).copy())
        self.raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        if terminated or truncated:
            self.completed_returns.append(self.episode_return)
        # 返回 reset 前的标准化终止观测，让 SB3 正确处理时间截断 bootstrap。
        return self.normalizer.normalize(self.raw_obs), reward, terminated, truncated, info

    def finish_update(self):
        self.normalizer.update(np.asarray(self.pending))
        self.pending.clear()
        return self.normalizer.normalize(self.raw_obs)


class AlignedPPO(PPO):
    """不改动 SB3 的 PPO loss/GAE，只在优化完成后更新输入统计量。"""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["normalizing_env", "after_update"]

    def train(self):
        super().train()
        # SB3 缓存着上一批末尾的观测；统计量变化后必须重新标准化。
        self._last_obs = self.normalizing_env.finish_update()[None, :]
        if getattr(self, "after_update", None) is not None:
            self.after_update()


class SB3PolicyAdapter(torch.nn.Module):
    """让共用评估/回放函数取得 SB3 高斯策略的均值和标准差。"""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        single = observations.ndim == 1
        batch = observations.unsqueeze(0) if single else observations
        dist = self.policy.get_distribution(batch).distribution
        mean, std = dist.mean, dist.stddev
        return (mean[0], std[0]) if single else (mean, std)


def load_bundle(checkpoint):
    checkpoint = Path(checkpoint)
    model = PPO.load(checkpoint, device="cpu")
    state = read_normalizer(model, checkpoint)
    normalizer = ObservationNormalizer(model.observation_space.shape[0])
    normalizer.load_state_dict(state)
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    return SB3PolicyAdapter(model.policy).eval(), normalizer, config, model.num_timesteps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--resume", type=Path, help="从 checkpoint 继续；updates 是累计目标，在新目录保存")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/phase1_sb3_continuous")
    args = parser.parse_args()
    previous_config = None
    if args.resume:
        previous_config = json.loads((args.resume.parent / "config.json").read_text(encoding="utf-8"))
        args.seed = previous_config["seed"]
        args.rollout_steps = previous_config["rollout_steps"]
        args.eval_every = previous_config["eval_every"]
        args.eval_episodes = previous_config["eval_episodes"]
    if min(args.updates, args.rollout_steps, args.eval_every, args.eval_episodes) < 1:
        parser.error("所有步数和评估参数必须为正")
    if args.rollout_steps % 64:
        parser.error("rollout-steps 必须是 64 的倍数")
    torch.set_num_threads(1)
    args.output_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = args.output_root / f"run_{index:03d}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            index += 1
    wrapped = FrozenRolloutObservation(Monitor(gym.make("HalfCheetah-v5")))
    env = DummyVecEnv([lambda: wrapped])
    model = AlignedPPO(
        "MlpPolicy", env, seed=args.seed, device="cpu", n_steps=args.rollout_steps,
        batch_size=64, n_epochs=4, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh),
    )
    if args.resume:
        model = AlignedPPO.load(args.resume, env=env, device="cpu", force_reset=True)
        wrapped.normalizer.load_state_dict(read_normalizer(model, args.resume))
        if args.updates * args.rollout_steps <= model.num_timesteps:
            env.close()
            parser.error("updates 必须大于 checkpoint 已完成的更新次数")
    model.normalizing_env = wrapped
    adapter = SB3PolicyAdapter(model.policy)
    config = dict(algorithm="SB3 PPO-Clip", env="HalfCheetah-v5", seed=args.seed,
                  obs_dim=17, action_dim=6, num_updates=args.updates,
                  rollout_steps=args.rollout_steps, minibatch_size=64, update_epochs=4,
                  gamma=0.99, gae_lambda=0.95, clip_coef=0.2, learning_rate=3e-4,
                  n_envs=1, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                  target_kl=None, clip_range_vf=None, use_sde=False,
                  net_arch=dict(pi=[64, 64], vf=[64, 64]), activation="Tanh",
                  eval_every=args.eval_every, eval_episodes=args.eval_episodes, eval_seed_start=10000,
                  observation_normalization=True, reward_normalization=False,
                  observation_normalization_scope="frozen rollout/update; accumulate raw input observations after optimization",
                  evaluation_action="clipped Gaussian mean", torch_threads=1,
                  differences_from_handwritten=dict(
                      initialization="SB3 orthogonal; actor output gain 0.01",
                      optimizer="one Adam, lr=3e-4, eps=1e-5; handwritten separate Adam, critic lr=1e-3, eps=1e-8",
                      gradient_clipping="global norm 0.5; handwritten none",
                      value_loss_weight="0.5; handwritten critic MSE weight 1",
                      advantage_normalization="each minibatch, sample std; handwritten once per rollout, population std",
                      kl_diagnostics="SB3 averages minibatch pre-update estimates; handwritten recomputes full batch after update"),
                  versions=dict(sb3=sb3.__version__, gymnasium=gym.__version__,
                                mujoco=mujoco.__version__, torch=str(torch.__version__), numpy=np.__version__))
    (path / "source").mkdir()
    if previous_config is not None:
        config.update(previous_config)
        config.update(num_updates=args.updates, resume_from=str(args.resume.resolve()),
                      resumed_at_steps=model.num_timesteps,
                      resume_note="Restores weights, optimizer and normalization; environment and RNG restart. Not an uninterrupted seed replicate.")
    config["source_sha256"] = {}
    for name in ["sb3_continuous.py", "ppo_continuous.py", "replay_continuous.py", "experiment_io.py",
                 "actor_critic.py", "networks.py", "ppo.py", "reinforce.py", "evaluate.py"]:
        data = Path(__file__).with_name(name).read_bytes()
        (path / "source" / name).write_bytes(data)
        config["source_sha256"][name] = hashlib.sha256(data).hexdigest()
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, best, best_update = [], -float("inf"), 0
    if args.resume:
        with (args.resume.parent / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if int(float(row["steps"])) <= model.num_timesteps:
                    rows.append({key: float(value) if value else "" for key, value in row.items()})
        # 保留原运行的最佳模型，仅使用不晚于恢复点的评估。
        evaluated = [row for row in rows if row["eval_mean"] != ""]
        if evaluated:
            best_row = max(evaluated, key=lambda row: row["eval_mean"])
            original_best = args.resume.parent / "best.zip"
            prior_model = PPO.load(original_best, device="cpu")
            if prior_model.num_timesteps == int(best_row["steps"]):
                state = read_normalizer(prior_model, original_best)
                prior_model.saved_normalizer_state = state
                buffer = io.BytesIO()
                prior_model.save(buffer)
                atomic_write(path / "best.zip", buffer.getvalue())
                buffer = io.BytesIO()
                torch.save(state, buffer)
                atomic_write(path / "best_normalizer.pt", buffer.getvalue())
                best, best_update = best_row["eval_mean"], int(best_row["update"])

    def save(name):
        # 模型与统计量嵌入同一个原子替换的 zip，避免中断时两者错配。
        model.saved_normalizer_state = wrapped.normalizer.state_dict()
        buffer = io.BytesIO()
        model.save(buffer)
        atomic_write(path / f"{name}.zip", buffer.getvalue())
        buffer = io.BytesIO()
        torch.save(model.saved_normalizer_state, buffer)
        try:
            atomic_write(path / f"{name}_normalizer.pt", buffer.getvalue())
        except OSError as error:
            warnings.warn(f"统计量副本保存失败；zip 内已有完整统计量：{error}")

    def write_metrics():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(path / "metrics.csv", buffer.getvalue().encode("utf-8"))

    def record():
        nonlocal best, best_update
        update = model.num_timesteps // args.rollout_steps
        row = dict.fromkeys(FIELDS, "")
        row.update(update=update, steps=model.num_timesteps)
        completed = wrapped.completed_returns
        if completed:
            row["train_mean"] = float(np.mean(completed))
            completed.clear()
        for source, target in [("policy_gradient_loss", "loss"), ("value_loss", "critic_loss"),
                               ("approx_kl", "approx_kl"), ("clip_fraction", "clip_fraction")]:
            if update:
                row[target] = float(model.logger.name_to_value[f"train/{source}"])
        if update:
            row["entropy"] = -float(model.logger.name_to_value["train/entropy_loss"])
        if update % args.eval_every == 0 or update == args.updates:
            row["eval_mean"], row["eval_std"] = evaluate_continuous_policy(
                adapter, n_episodes=args.eval_episodes, seed_start=10000,
                obs_normalizer=wrapped.normalizer)
            if row["eval_mean"] > best:
                best, best_update = row["eval_mean"], update
                save("best")
        if not all(np.isfinite(v) for v in row.values() if v != ""):
            raise FloatingPointError("SB3 metrics contain non-finite values")
        rows.append(row)
        save("last")
        try:
            write_metrics()
        except OSError as error:
            # 下一轮重写全部内存记录；日志失败不丢弃已完成的模型更新。
            warnings.warn(f"metrics.csv 写入失败，模型已保存，下一轮重试：{error}")
        print(json.dumps(row), flush=True)

    model.after_update = record
    status = "interrupted_or_failed"
    try:
        print(f"Output directory: {path}", flush=True)
        if args.resume:
            save("last")
            write_metrics()
        else:
            record()
        remaining = args.updates * args.rollout_steps - model.num_timesteps
        model.learn(remaining, reset_num_timesteps=not bool(args.resume))
        status = "completed"
    finally:
        env.close()
        metrics_saved = True
        try:
            write_metrics()
        except OSError as error:
            metrics_saved = False
            warnings.warn(f"最终 CSV 保存失败，模型仍可用：{error}")
        (path / "summary.json").write_text(json.dumps(dict(
            status=status, steps=model.num_timesteps, best_eval_mean=best,
            best_update=best_update, normalizer_count=wrapped.normalizer.count,
            metrics_saved=metrics_saved,
            checkpoint_note="zip includes normalization. Resume restarts environment/RNG, not an exact uninterrupted trajectory."), indent=2), encoding="utf-8")
        if rows:
            plot_metrics(rows, path / "curve.png")


if __name__ == "__main__":
    main()
