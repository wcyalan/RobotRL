"""Validate all independent G1 evaluations and compare each group to the baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import summarize_baseline as baseline


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root/'experiments/phase3_g1_deployment')
    parser.add_argument('--baseline', type=Path, default=root/'experiments/phase3_g1_baseline')
    args = parser.parse_args()
    result = args.results.resolve()
    baseline.RESULTS = args.baseline.resolve()
    assert (result/'closed_loop_completed.txt').exists(), 'Independent evaluation is incomplete'
    protocol_path = root/'phase3/evaluation/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    digest = sha(protocol_path)
    for name, value in json.loads((result/'closed_loop_source_manifest.json').read_text()).items():
        assert sha(result/'closed_loop_source'/name) == value
    assert sha(result/'closed_loop_source/protocol.json') == digest
    task = json.loads((result/'closed_loop_task.json').read_text())
    records = []
    for seed in (0, 1):
        run = result/f'run_{seed+1:03d}'
        export = json.loads((run/'export/manifest.json').read_text())
        policy_hash = sha(run/'export/policy.pt')
        assert policy_hash == export['policy_sha256']
        new = baseline.summarize_run(seed, protocol, digest, evaluation_root=run/'closed_loop',
                                     expected_policy_hash=policy_hash, task_id=task['task_id'])
        assert new['model_sha256'] == export['checkpoint_sha256']
        old = json.loads((baseline.RESULTS/f'run_{seed+1:03d}/summary.json').read_text())
        for a, b in zip(old['results'], new['results']):
            assert (a['eval_seed'], a['condition']) == (b['eval_seed'], b['condition'])
            raw = json.loads((run/f"closed_loop/eval_{b['eval_seed']}/{b['condition']}/summary.json").read_text())
            assert raw['evaluator_sha256'] == sha(result/'closed_loop_source/evaluate_export.py')
            records.append(dict(training_seed=seed, eval_seed=b['eval_seed'], condition=b['condition'],
                                original_passed=a['passed'], exported_passed=b['passed'],
                                original_early=a['early_termination'], exported_early=b['early_termination'],
                                original_xy=a['mean_xy_errors'], exported_xy=b['mean_xy_errors'],
                                original_yaw=a['mean_yaw_errors'], exported_yaw=b['mean_yaw_errors'],
                                original_stable_count=a['stop_stability_reached_count'],
                                exported_stable_count=b['stop_stability_reached_count']))
    comparison = dict(completed=True, task_id=task['task_id'], protocol_sha256=digest,
                      groups=len(records), episodes=len(records)*32, results=records,
                      matching_pass_counts=all(r['original_passed']==r['exported_passed'] for r in records),
                      exported_early_terminations=sum(r['exported_early'] for r in records))
    (result/'closed_loop_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(f"Validated {comparison['groups']} groups / {comparison['episodes']} episodes; "
          f"matching pass counts: {comparison['matching_pass_counts']}; "
          f"early terminations: {comparison['exported_early_terminations']}. "
          f"Details: {result/'closed_loop_comparison.json'}")


if __name__ == '__main__':
    main()
