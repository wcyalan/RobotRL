"""实验保存工具；不包含策略梯度算法。"""

import csv
import hashlib
import json
import platform
import subprocess
import re
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "update", "steps", "train_mean", "loss", "critic_loss", "eval_mean", "eval_std",
    "approx_kl", "clip_fraction", "entropy",
]


def plot_metrics(rows, destination):
    critic = [r for r in rows if r.get("critic_loss") not in (None, "")]
    panels = 3 if critic else 2
    fig, axes = plt.subplots(panels, 1, figsize=(8, 3.5 * panels), sharex=True)
    train = [r for r in rows if r.get("train_mean") not in (None, "")]
    evaluation = [r for r in rows if r.get("eval_mean") not in (None, "")]
    axes[0].plot([r["steps"] for r in train], [r["train_mean"] for r in train],
                 label="Training: completed-episode mean", alpha=0.7)
    axes[0].plot([r["steps"] for r in evaluation], [r["eval_mean"] for r in evaluation],
                 "o-", label="Deterministic evaluation: mean")
    if evaluation:
        means = np.array([r["eval_mean"] for r in evaluation])
        stds = np.array([r["eval_std"] for r in evaluation])
        axes[0].fill_between([r["steps"] for r in evaluation], means - stds,
                             means + stds, alpha=0.15, label="Evaluation +/- 1 std")
    axes[0].set_ylabel("Undiscounted episode return")
    axes[0].legend(fontsize=8)
    actor = [r for r in rows if r.get("loss") not in (None, "")]
    axes[1].plot([r["steps"] for r in actor], [r["loss"] for r in actor])
    axes[1].set_ylabel("Actor surrogate loss")
    if critic:
        axes[2].plot([r["steps"] for r in critic], [r["critic_loss"] for r in critic])
        axes[2].set_ylabel("Critic MSE (changing batches)")
    axes[-1].set_xlabel("Training environment steps (evaluation excluded)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


class RunRecorder:
    def __init__(self, config, output_root=None):
        base = Path(output_root) if output_root else ROOT / "experiments" / "phase1_reinforce"
        base.mkdir(parents=True, exist_ok=True)
        numbers = [int(match.group(1)) for path in base.iterdir()
                   if (match := re.fullmatch(r"run_(\d+)", path.name))]
        number = max(numbers, default=0) + 1
        # mkdir is atomic: concurrent starts cannot overwrite the same run.
        while True:
            self.path = base / f"run_{number:03d}"
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                number += 1
        self.config = config
        self.rows = []
        self.best_mean = -float("inf")
        self.best_update = None
        self.best_steps = None
        self.last_eval = None

        metadata = dict(config)
        metadata["versions"] = {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "gymnasium": gym.__version__, "numpy": np.__version__,
        }
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True)
        metadata["git_head"] = revision.stdout.strip() if revision.returncode == 0 else None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True)
        metadata["working_tree_status"] = status.stdout
        # 保存源码快照：未提交的学习代码不能只依赖 git HEAD 来定位。
        snapshots = self.path / "source"
        snapshots.mkdir()
        metadata["source_sha256"] = {}
        for name in (
            "networks.py", "reinforce.py", "actor_critic.py", "ppo.py", "ppo_continuous.py",
            "evaluate.py", "experiment_io.py",
        ):
            data = (Path(__file__).parent / name).read_bytes()
            (snapshots / name).write_bytes(data)
            metadata["source_sha256"][name] = hashlib.sha256(data).hexdigest()
        self._json("config.json", metadata)
        with (self.path / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def _json(self, name, data):
        (self.path / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")

    def record(self, policy, optimizer, update, steps, train_mean=None, loss=None, evaluation=None,
               *, value_net=None, critic_optimizer=None, critic_loss=None, diagnostics=None,
               observation_normalizer_state=None):
        if (value_net is None) != (critic_optimizer is None):
            raise ValueError("value_net and critic_optimizer must be provided together")
        row = dict(update=update, steps=steps, train_mean=train_mean, loss=loss,
                   critic_loss=critic_loss,
                   eval_mean=evaluation[0] if evaluation else None,
                   eval_std=evaluation[1] if evaluation else None)
        diagnostics = diagnostics or {}
        for name in ("approx_kl", "clip_fraction", "entropy"):
            row[name] = diagnostics.get(name)
        self.rows.append(row)
        with (self.path / "metrics.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
        if evaluation is not None:
            self.last_eval = {"update": update, "steps": steps,
                              "mean": evaluation[0], "std": evaluation[1]}
        checkpoint = {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update, "steps": steps, "config": self.config,
            "last_evaluation": self.last_eval,
        }
        if value_net is not None:
            checkpoint["value_state_dict"] = value_net.state_dict()
            checkpoint["critic_optimizer_state_dict"] = critic_optimizer.state_dict()
        if observation_normalizer_state is not None:
            checkpoint["observation_normalizer_state"] = observation_normalizer_state
        # 同目录临时文件替换，避免中断写入破坏已有 checkpoint。
        self._checkpoint("last.pt", checkpoint)
        if evaluation is not None and evaluation[0] > self.best_mean:
            self.best_mean = evaluation[0]
            self.best_update, self.best_steps = update, steps
            self._checkpoint("best.pt", checkpoint)

    def _checkpoint(self, name, data):
        temporary = self.path / (name + ".tmp")
        torch.save(data, temporary)
        temporary.replace(self.path / name)

    def finish(self, status):
        plot_metrics(self.rows, self.path / "curve.png")
        self._json("summary.json", {
            "status": status,
            "last_completed_update": self.rows[-1]["update"] if self.rows else None,
            "best_eval_mean": self.best_mean if self.best_update is not None else None,
            "best_update": self.best_update, "best_steps": self.best_steps,
            "last_evaluation": self.last_eval,
            "checkpoint_note": "Model/optimizer snapshots; exact environment/RNG resume is not implemented.",
        })
