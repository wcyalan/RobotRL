"""Finite-value and reset smoke test using the pinned kit-less runtime."""
import json
from pathlib import Path
import warp as wp

wp.config.enable_backward = False
import torch
import gymnasium as gym
from isaaclab_rl.entrypoints.simple_agents import _parse_args
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import resolve_task_config


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), "Nonfinite observation/reward/state"


args = _parse_args(None, "zero")
cfg, _ = resolve_task_config(args.task, "")
cfg.scene.num_envs = 4
cfg.episode_length_s = 0.2  # Force time-limit resets; training retains its own default.
cfg.seed = 0
args.device = cfg.sim.device
torch.manual_seed(0)
counts = {"terminated": 0, "truncated": 0, "steps": 0}
with launch_simulation(cfg, args):
    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset()
        finite(obs)
        assert obs["policy"].shape == (4, 48)
        reset_seen = torch.zeros(4, dtype=torch.bool, device=env.unwrapped.device)
        with torch.inference_mode():
            for step in range(100):
                action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                if step >= 50:
                    action.uniform_(-0.1, 0.1)
                obs, reward, terminated, truncated, _ = env.step(action)
                finite(obs)
                finite(reward)
                assert reward.shape == (4,)
                done = terminated | truncated
                reset_seen |= done
                if done.any():
                    assert (env.unwrapped.episode_length_buf[done] == 0).all().item()
                counts["terminated"] += terminated.sum().item()
                counts["truncated"] += truncated.sum().item()
                counts["steps"] += 1
            assert reset_seen.all().item(), "Each environment must exercise auto-reset"
            assert counts["truncated"] > 0, "Time-limit reset was not exercised"
            obs, _ = env.reset()
            finite(obs)
            assert (env.unwrapped.episode_length_buf == 0).all().item()
        counts.update(passed=True, observation_shape=list(obs["policy"].shape),
                      action_shape=list(env.action_space.shape), explicit_reset_passed=True,
                      note="Termination counts are observed, not a test of every termination condition.")
        Path("validation.json").write_text(json.dumps(counts, indent=2))
        print("ROBOTRL_VALIDATION_PASSED", counts, flush=True)
    finally:
        env.close()
