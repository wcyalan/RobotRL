"""Inspect the installed G1 task, optionally reset/step it without training.

Run inside the pinned Isaac Lab container. Metadata is recorded before simulation
so a missing asset does not hide which official configuration was selected.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import subprocess
from pathlib import Path


def encode(value):
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=encode) + "\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset")
    parser.add_argument("--steps", type=int, default=0)
    opts, remainder = parser.parse_known_args()
    out = Path(opts.output)
    out.mkdir(parents=True, exist_ok=True)

    import warp as wp
    wp.config.enable_backward = False
    import gymnasium as gym
    import torch
    from isaaclab_rl.entrypoints.simple_agents import _parse_args
    from isaaclab_tasks.utils import resolve_task_config

    args = _parse_args(remainder, "zero")
    cfg, agent = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    save(out / "official_env.json", cfg.to_dict())
    save(out / "official_agent.json", agent.to_dict())
    source_paths = set()
    for cls in (*type(cfg).__mro__, *type(agent).__mro__):
        try:
            path = Path(inspect.getfile(cls))
            if path.is_file():
                source_paths.add(path)
        except TypeError:
            pass
    import isaaclab_assets.robots.unitree as unitree
    source_paths.add(Path(unitree.__file__))
    sources = {}
    for path in sorted(source_paths):
        relative = str(path).removeprefix("/workspace/isaaclab/").lstrip("/")
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for package in ["isaaclab", "torch", "newton", "warp-lang", "rsl-rl-lib", "gymnasium"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    git = subprocess.run(["git", "-C", "/workspace/isaaclab", "rev-parse", "HEAD"], capture_output=True, text=True)
    summary = {
        "task": args.task,
        "registered_g1_tasks": sorted(k for k in gym.registry if "G1" in k),
        "entry_points": gym.spec(args.task).kwargs,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "isaaclab_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": sources,
        "source_asset_url": cfg.scene.robot.spawn.usd_path,
        "official_num_envs": cfg.scene.num_envs,
        "max_iterations": agent.max_iterations,
        "num_steps_per_env": agent.num_steps_per_env,
        "total_environment_steps": cfg.scene.num_envs * agent.max_iterations * agent.num_steps_per_env,
        "physics_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "episode_length_s": cfg.episode_length_s,
        "simulation_requested": opts.steps > 0,
    }
    save(out / "inventory.json", summary)
    print("G1_INVENTORY", json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, default=encode), flush=True)
    if opts.steps == 0:
        return

    from isaaclab.app import launch_simulation
    cfg.scene.num_envs = args.num_envs or 4
    cfg.seed = 0
    if opts.asset:
        cfg.scene.robot.spawn.usd_path = opts.asset
        from pxr import Usd, UsdPhysics, Sdf
        asset_path = Path(opts.asset)
        layer = Sdf.Layer.FindOrOpen(str(asset_path))
        # Empty asset identifiers are internal USD references, not missing files.
        references = [p for p in layer.GetExternalReferences() if p]
        missing = [p for p in references if not (asset_path.parent / p).is_file()]
        assert not missing, f"Missing USD dependencies: {missing}"
        stage = Usd.Stage.Open(str(asset_path))
        colliders, masses, inertias = [], {}, {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.MassAPI):
                api = UsdPhysics.MassAPI(prim)
                mass = api.GetMassAttr().Get()
                inertia = api.GetDiagonalInertiaAttr().Get()
                if mass is not None:
                    masses[str(prim.GetPath())] = mass
                if inertia is not None:
                    inertias[str(prim.GetPath())] = list(inertia)
        save(out / "asset.json", {
            "source_url": summary["source_asset_url"], "path": str(asset_path),
            "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "bytes": asset_path.stat().st_size, "external_references": references,
            "collision_prims": colliders, "authored_masses": masses,
            "authored_diagonal_inertias": inertias,
        })
    args.device = cfg.sim.device
    save(out / "smoke_env.json", cfg.to_dict())
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            base = env.unwrapped
            obs, _ = env.reset(seed=0)
            robot = base.scene["robot"]
            result = {
                "num_envs": base.num_envs, "device": str(base.device),
                "joint_names": robot.joint_names, "body_names": robot.body_names,
                "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                "action_dim": base.action_manager.total_action_dim,
                "steps": opts.steps, "seed": 0,
                "asset": cfg.scene.robot.spawn.usd_path,
                "overrides": {"num_envs": base.num_envs, "seed": 0, "asset": cfg.scene.robot.spawn.usd_path},
                "zero_action_meaning": "See resolved actions.joint_pos: default joint position plus scale * zero",
            }
            def tensor(value):
                return value if isinstance(value, torch.Tensor) else value.torch
            for name in ["body_mass", "body_inertia", "default_joint_pos"]:
                if hasattr(robot.data, name):
                    value = tensor(getattr(robot.data, name))
                    assert torch.isfinite(value).all(), name
                    result[name] = value[0].cpu().tolist()
            contacts = base.scene["contact_forces"]
            result["contact_body_names"] = contacts.body_names
            action = torch.zeros((base.num_envs, result["action_dim"]), device=base.device)
            terminated_count = truncated_count = 0
            reward_min, reward_max, max_contact = float("inf"), -float("inf"), 0.0
            for value in obs.values():
                assert torch.isfinite(value).all(), "nonfinite reset observation"
            for step in range(opts.steps):
                obs, reward, terminated, truncated, _ = env.step(action)
                for value in obs.values():
                    assert torch.isfinite(value).all(), f"nonfinite observation at {step}"
                assert torch.isfinite(reward).all(), f"nonfinite reward at {step}"
                force = tensor(contacts.data.net_forces_w)
                assert torch.isfinite(force).all(), "nonfinite contact force"
                max_contact = max(max_contact, force.norm(dim=-1).max().item())
                terminated_count += int(terminated.sum().item())
                truncated_count += int(truncated.sum().item())
                reward_min = min(reward_min, reward.min().item())
                reward_max = max(reward_max, reward.max().item())
            # Explicit reset after stepping also exercises reinitialization.
            obs, _ = env.reset(seed=1)
            assert all(torch.isfinite(v).all() for v in obs.values())
            result.update(passed=True, terminated_count=terminated_count,
                          truncated_count=truncated_count, reward_range=[reward_min, reward_max],
                          max_contact_force_n=max_contact, explicit_resets=2,
                          automatic_reset_observed=terminated_count + truncated_count > 0)
            save(out / "smoke.json", result)
            print("G1_SMOKE_PASSED", json.dumps({k: result[k] for k in ["action_dim", "observation_shapes", "terminated_count", "truncated_count", "max_contact_force_n"]}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
