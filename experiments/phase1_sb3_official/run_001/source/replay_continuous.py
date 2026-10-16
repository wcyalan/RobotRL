"""回放连续动作 PPO；默认录制视频，--render 打开实时窗口。"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from networks import GaussianPolicyNetwork
from ppo_continuous import ObservationNormalizer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=20000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render", action="store_true", help="实时窗口，不录制视频")
    mode.add_argument("--no-video", action="store_true", help="只评估回报和姿态，保存 JSON")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    if args.checkpoint.suffix == ".zip":
        from sb3_continuous import load_bundle

        policy, normalizer, config, steps = load_bundle(args.checkpoint)
        checkpoint = {"update": steps // config["rollout_steps"]}
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        policy = GaussianPolicyNetwork(config["obs_dim"], config["action_dim"])
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        normalizer = None
        if config.get("observation_normalization", False):
            normalizer = ObservationNormalizer(config["obs_dim"])
            # 缺少统计量时直接报错，不能用未归一化的输入冒充正确回放。
            normalizer.load_state_dict(checkpoint["observation_normalizer_state"])

    output = None
    if not args.render:
        if not args.no_video:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401: 提前检查视频编码依赖

        index = 1
        while True:
            output = args.checkpoint.resolve().parent / f"replay_{index:03d}"
            try:
                output.mkdir()
                break
            except FileExistsError:
                index += 1

    camera_options = {"camera_name": "track"} if config["env"].startswith("HalfCheetah") else {}
    if args.render and config["env"].startswith("HalfCheetah"):
        # human viewer 使用自己的相机，需要显式启用躯干跟随。
        import mujoco

        camera_options["default_camera_config"] = {
            "type": int(mujoco.mjtCamera.mjCAMERA_TRACKING),
            "trackbodyid": 1, "distance": 4.0, "elevation": -20.0,
        }
    render_mode = "human" if args.render else (None if args.no_video else "rgb_array")
    env = gym.make(config["env"], render_mode=render_mode,
                   **camera_options)
    results = []
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            obs, _ = env.reset(seed=seed)
            total, steps, clipped, components = 0.0, 0, 0, 0
            x_start = float(env.unwrapped.data.qpos[0])
            pitches, torso_heights = [], []
            writer = None
            try:
                if output and not args.no_video:
                    video = output / f"episode_{episode + 1:03d}.mp4"
                    writer = imageio.get_writer(str(video), fps=env.metadata["render_fps"])
                    writer.append_data(env.render())
                while True:
                    network_obs = normalizer.normalize(obs) if normalizer else obs
                    mean, _ = policy(torch.as_tensor(network_obs, dtype=torch.float32))
                    raw_action = mean.numpy()
                    action = np.clip(raw_action, env.action_space.low, env.action_space.high)
                    clipped += int(np.count_nonzero(raw_action != action))
                    components += action.size
                    obs, reward, terminated, truncated, _ = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(reward):
                        raise RuntimeError("Replay produced non-finite observations/reward")
                    total += float(reward)
                    steps += 1
                    # HalfCheetah 的 rooty 是躯干在二维平面内的转角。
                    if config["env"].startswith("HalfCheetah"):
                        pitches.append(float(env.unwrapped.data.qpos[2]))
                        torso_heights.append(float(env.unwrapped.data.body("torso").xpos[2]))
                    if writer:
                        frame = env.render()
                        writer.append_data(frame)
                        # 保存连续步态帧与后期姿态，便于对照视频检查。
                        if episode == 0 and steps in (1, 100, 102, 104, 106, 108, 110, 300, 600, 1000):
                            imageio.imwrite(output / f"frame_{steps:04d}.png", frame)
                    if terminated or truncated:
                        break
            finally:
                if writer is not None:
                    writer.close()
            displacement = float(env.unwrapped.data.qpos[0]) - x_start
            result = {"seed": seed, "return": total, "steps": steps,
                      "terminated": bool(terminated), "truncated": bool(truncated),
                      "x_displacement": displacement,
                      "mean_x_velocity": displacement / (steps * env.unwrapped.dt),
                      "action_clipped_fraction": clipped / components}
            if pitches:
                result["torso_pitch_range_radians"] = [min(pitches), max(pitches)]
                result["torso_height_range"] = [min(torso_heights), max(torso_heights)]
                # 躯干局部向上方向指向地面的时间比例；不等同于接触检测。
                result["torso_inverted_fraction"] = float(np.mean(np.cos(pitches) < 0))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.close()
    if output:
        report = {"checkpoint": str(args.checkpoint.resolve()),
                  "update": checkpoint["update"], "action": "clipped Gaussian mean",
                  "normalizer_count": normalizer.count if normalizer else None,
                  "return_mean": float(np.mean([r["return"] for r in results])),
                  "return_std": float(np.std([r["return"] for r in results])),
                  "episodes": results}
        (output / "replay_summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Replay saved: {output}")


if __name__ == "__main__":
    main()
