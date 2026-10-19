"""Parse official training logs and plot both complete G1 baseline runs.

Run from the repository root: python phase3/analysis/summarize_baseline.py
Training rewards are the logger's rolling episode statistics, not behavior scores.
"""
import argparse
import csv
import hashlib
import json
from statistics import mean
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/phase3_g1_baseline"


def parse_log(path, seed):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8"))
    blocks = re.split(r"Learning iteration (\d+)/1500", text)
    rows = []
    fields = {"environment_steps": "Total steps", "mean_reward": "Mean reward",
              "mean_episode_length": "Mean episode length",
              "value_loss": "Mean value loss", "surrogate_loss": "Mean surrogate loss"}
    for index in range(1, len(blocks), 2):
        row = {"training_seed": seed, "iteration": int(blocks[index])}
        for key, label in fields.items():
            match = re.search(r"^\s*" + re.escape(label) + r":\s*([-+\d.eE]+)", blocks[index+1], re.M)
            if match is None:
                raise ValueError(f"Missing {label} at iteration {row['iteration']}")
            row[key] = float(match[1])
            assert math.isfinite(row[key]), (seed, row)
        row["environment_steps"] = int(row["environment_steps"])
        assert row["environment_steps"] == (row["iteration"] + 1) * 4096 * 24
        rows.append(row)
    assert [r["iteration"] for r in rows] == list(range(1500))
    return rows



def summarize_run(seed, protocol, protocol_hash, evaluation_root=None, expected_policy_hash=None, task_id=None):
    """Reject partial/mismatched evidence and recompute pass decisions per lane."""
    directory = RESULTS / f"run_{seed+1:03d}"
    check = json.loads((directory / "training_check.json").read_text())
    assert check["passed"] and check["all_logged_scalars_finite"]
    assert (check["training_envs"], check["iterations"], check["rollout_steps"]) == (4096, 1500, 24)
    models = list(directory.glob("logs/**/model_1499.pt"))
    assert len(models) == 1
    model_hash = hashlib.sha256(models[0].read_bytes()).hexdigest()
    assert model_hash == check["final_checkpoint_sha256"]
    assert hashlib.sha256((directory / "source/protocol.json").read_bytes()).hexdigest() == protocol_hash
    groups = []
    for eval_seed in protocol["eval_seeds"]:
        for condition, case in protocol["conditions"].items():
            path = (evaluation_root or directory / 'evaluation') / f"eval_{eval_seed}/{condition}"
            raw = json.loads((path / "summary.json").read_text())
            if expected_policy_hash is not None:
                assert raw['policy_sha256'] == expected_policy_hash and raw['ppo_runner_constructed'] is False
            assert raw["completed"] and not raw["evaluator_smoke"]
            assert raw["model_sha256"] == model_hash and raw["protocol_sha256"] == protocol_hash
            assert raw["eval_seed"] == eval_seed and raw["condition"] == condition
            n = protocol["num_envs"]
            assert raw["num_envs"] == n == 32
            for key in ("lane_passed", "early_termination", "time_out", "lengths", "returns", "stop_stable_time_s"):
                assert len(raw[key]) == n
            assert all(math.isfinite(x) for x in raw["returns"])
            assert len(raw["windows"]) == len(case["measurement_windows"])
            assert [x["command"] for x in raw["command_switches"]] == [x[2] for x in case["segments"]]
            assert [x["time_s"] for x in raw["command_switches"]] == [x[0] for x in case["segments"]]
            horizon = round(case["duration_s"] / raw["step_dt"])
            decisions = [not raw["early_termination"][i] and raw["time_out"][i] and raw["lengths"][i] == horizon for i in range(n)]
            thresholds = protocol["thresholds"]
            windows = []
            for window, specification in zip(raw["windows"], case["measurement_windows"]):
                start, end, kind = specification
                assert (window["start_s"], window["end_s"], window["kind"]) == (start, end, kind)
                expected = round((end-start)/raw["step_dt"])
                assert window["expected_samples"] == expected
                xy = thresholds["stopped_xy_speed_m_s" if kind == "stopped" else "moving_xy_error_m_s"]
                yaw = thresholds["stopped_yaw_speed_rad_s" if kind == "stopped" else "moving_yaw_error_rad_s"]
                for key in ("counts", "mean_xy_error_m_s", "mean_yaw_error_rad_s", "mean_tilt_deg"):
                    assert len(window[key]) == n and all(math.isfinite(x) for x in window[key])
                for i in range(n):
                    decisions[i] &= (window["counts"][i] == expected and window["mean_xy_error_m_s"][i] <= xy
                                     and window["mean_yaw_error_rad_s"][i] <= yaw
                                     and window["mean_tilt_deg"][i] <= thresholds["mean_tilt_deg"])
                windows.append({"start_s": start, "end_s": end, "kind": kind,
                                "mean_xy_error_m_s": mean(window["mean_xy_error_m_s"]),
                                "mean_yaw_error_rad_s": mean(window["mean_yaw_error_rad_s"]),
                                "mean_tilt_deg": mean(window["mean_tilt_deg"])})
            before_stability = sum(decisions)
            if condition == "stop_restart":
                for i, value in enumerate(raw["stop_stable_time_s"]):
                    decisions[i] &= value is not None and value <= thresholds["stop_stable_by_s"]
            assert decisions == raw["lane_passed"] and sum(decisions) == raw["pass_count"]
            assert raw["group_passed"] == (sum(decisions) >= protocol["minimum_pass_count"])
            assert (path / "trajectory.npz").exists()
            groups.append({"eval_seed": eval_seed, "condition": condition, "passed": sum(decisions), "total": n,
                           "group_passed": raw["group_passed"], "early_termination": sum(raw["early_termination"]),
                           "time_out": sum(raw["time_out"]), "mean_return": mean(raw["returns"]),
                           "mean_xy_errors": [w["mean_xy_error_m_s"] for w in windows],
                           "mean_yaw_errors": [w["mean_yaw_error_rad_s"] for w in windows], "windows": windows,
                           "full_episode_and_window_pass_count": before_stability,
                           "stop_stability_reached_count": sum(v is not None for v in raw["stop_stable_time_s"]) if condition == "stop_restart" else None})
    seconds = float(re.search(r"Training time: ([\d.]+) seconds", (directory / "training.log").read_text(encoding="utf-8"))[1])
    plan_path = RESULTS / "plan.json"
    task_id = task_id or (json.loads(plan_path.read_text())["task_id"] if plan_path.exists() else
               ("a491ade1-dec7-43a9-ac95-bf075b683cd0", "bd110e66-0e44-47f4-82f8-3d37144eb14f")[seed])
    result = {"training_seed": seed, "task_id": task_id,
              "training_completed": True, "training_seconds": seconds, "iterations": 1500, "environment_steps": 147456000,
              "model_sha256": model_hash, "protocol_sha256": protocol_hash,
              "groups_completed": len(groups), "groups_passed": sum(g["group_passed"] for g in groups),
              "total_episodes": sum(g["total"] for g in groups), "early_terminations": sum(g["early_termination"] for g in groups),
              "results": groups}
    if expected_policy_hash is not None:
        result.update(policy_sha256=expected_policy_hash, ppo_runner_constructed=False,
                      purpose='Independent TorchScript closed-loop evaluation; original training metadata above')
    ((evaluation_root or directory) / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Baseline result root; override for the NAS code/ layout")
    RESULTS = parser.parse_args().results.resolve()
    series = [parse_log(RESULTS / f"run_{seed+1:03d}/training.log", seed) for seed in (0, 1)]
    with (RESULTS / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=series[0][0].keys())
        writer.writeheader()
        writer.writerows(row for rows in series for row in rows)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    for seed, (rows, color) in enumerate(zip(series, ("#0072B2", "#D55E00"))):
        ax.plot([r["environment_steps"] / 1e6 for r in rows],
                [r["mean_reward"] for r in rows], lw=1.0, color=color,
                label=f"Training seed {seed}", linestyle="-" if seed == 0 else "--")
    ax.set(xlabel="Environment steps (millions)", ylabel="Training mean episode reward", xlim=(0, 150))
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(direction="out", length=3)
    fig.savefig(RESULTS / "learning_curve.png", dpi=600)
    fig.savefig(RESULTS / "learning_curve.svg")
    fig.savefig(RESULTS / "learning_curve.pdf")
    plt.close(fig)
    protocol_path = ROOT / "phase3/evaluation/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for seed in (0, 1):
        result = summarize_run(seed, protocol, protocol_hash)
        print(f"Seed {seed}: {result['groups_completed']} groups, {result['groups_passed']} pass, {result['early_terminations']} early terminations")
    print("Saved 3000 log rows and PNG/SVG/PDF; no additional smoothing or aggregation.")


if __name__ == "__main__":
    main()
