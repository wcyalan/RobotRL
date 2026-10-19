"""Describe frozen stop failures from existing evidence; never rescore with pose derivatives.

Only two lanes per evaluation group have saved poses. All-lane acceptance comes
from the original instantaneous-velocity evaluator, not the pose estimates here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/phase3_g1_baseline"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_window(trace, lane, start, end, dt, joint_names):
    first, last = round(start / dt), round(end / dt)
    # Include the endpoint pose for displacement; commands cover [start, end).
    pose = trace["root_pose"][first:last+1, lane].astype(float)
    angles = trace["joint_pos"][first:last+1, lane].astype(float)
    assert len(pose) == last-first+1
    assert np.all(trace["active"][first:last+1, lane])
    assert np.all(trace["command"][first:last, lane] == 0)
    assert np.isfinite(pose).all() and np.isfinite(angles).all()
    quat = pose[:, 3:]
    assert np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4)
    x, y, z, w = quat.T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    delta = pose[-1, :2]-pose[0, :2]
    forward = float(delta @ [np.cos(yaw[0]), np.sin(yaw[0])])
    lateral = float(delta @ [-np.sin(yaw[0]), np.cos(yaw[0])])
    path_length = float(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1).sum())
    knees = {}
    for name in ("left_knee_joint", "right_knee_joint"):
        values = np.rad2deg(angles[:, joint_names.index(name)])
        knees[name] = {"min_deg": float(values.min()), "max_deg": float(values.max()), "range_deg": float(np.ptp(values))}
    return {"start_s": start, "end_s": end, "net_xy_displacement_m": float(np.linalg.norm(delta)),
            "net_forward_from_start_heading_m": forward, "net_lateral_from_start_heading_m": lateral,
            "sampled_xy_path_length_m": path_length, "interval_mean_path_speed_m_s": path_length/(end-start),
            "net_heading_change_deg": float(np.rad2deg(yaw[-1]-yaw[0])),
            "heading_range_deg": float(np.rad2deg(np.ptp(yaw))),
            "pelvis_height_min_m": float(pose[:, 2].min()), "pelvis_height_max_m": float(pose[:, 2].max()),
            "knees": knees}


def main():
    protocol_path = ROOT / "phase3/evaluation/protocol.json"
    protocol_hash = sha(protocol_path)
    names_path = ROOT / "experiments/phase3_g1_environment/run_001/smoke/smoke.json"
    names = json.loads(names_path.read_text())["joint_names"]
    manifest_path = BASE / "diagnosis/source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["sha256"].items():
        assert sha(BASE / "diagnosis/source" / relative) == expected
    groups, lanes = [], []
    for seed in (0, 1):
        run = BASE / f"run_{seed+1:03d}"
        model_hash = json.loads((run / "training_check.json").read_text())["final_checkpoint_sha256"]
        for eval_seed in (20000, 21000):
            directory = run / f"evaluation/eval_{eval_seed}/stop_restart"
            summary_path, trace_path = directory / "summary.json", directory / "trajectory.npz"
            summary = json.loads(summary_path.read_text())
            assert summary["completed"] and summary["num_envs"] == 32
            assert summary["protocol_sha256"] == protocol_hash and summary["model_sha256"] == model_hash
            trace = np.load(trace_path)
            assert trace["root_pose"].shape == (1500, 2, 7)
            stable = summary["stop_stable_time_s"]
            groups.append({"training_seed": seed, "eval_seed": eval_seed,
                           "pass_count": summary["pass_count"], "total": 32,
                           "early_terminations": sum(summary["early_termination"]),
                           "stable_window_reached_count": sum(t is not None for t in stable),
                           "summary_sha256": sha(summary_path), "trace_sha256": sha(trace_path), "model_sha256": model_hash})
            for lane in (0, 1):
                lanes.append({"training_seed": seed, "eval_seed": eval_seed, "lane": lane,
                              "whole_stop": describe_window(trace, lane, 12, 22, .02, names),
                              "late_stop": describe_window(trace, lane, 15, 22, .02, names)})
    report = {"method": "No new training/evaluation. Frozen acceptance for 128 episodes; pose-only descriptions for 8 preselected lanes.",
              "limitations": "Pose sampling gives displacement and interval-average speed, not instantaneous simulator velocity. Do not use these derivatives to replace the frozen stable-window result or infer contact forces/reward attribution.",
              "protocol_sha256": protocol_hash, "joint_names_evidence_sha256": sha(names_path),
              "runtime_source_manifest_sha256": sha(manifest_path), "groups": groups, "saved_lanes": lanes}
    (BASE / "stop_diagnosis.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for seed in (0, 1):
        selected = [l for l in lanes if l["training_seed"] == seed]
        for window in ("whole_stop", "late_stop"):
            for metric in ("net_xy_displacement_m", "net_forward_from_start_heading_m", "net_lateral_from_start_heading_m", "net_heading_change_deg"):
                values = [l[window][metric] for l in selected]
                print(seed, window, metric, f"{min(values):.4f}..{max(values):.4f}")


if __name__ == "__main__":
    main()
