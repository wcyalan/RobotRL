"""One-GPU sequential G1 standing-probability comparison inside the pinned image.

Only training rel_standing_envs changes from .02 to .20. The frozen evaluator
and baseline evidence remain untouched. Partial stages are never overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

BASE = Path('/robotrl/experiments/phase3_g1_baseline/run_001')
ASSET = Path('/robotrl/assets/G1/g1_minimal.usd')
PROTOCOL_HASH = '894cad975e0ca4bf06fe1c605e7fe0dc21b46caddfd2a5d786ddafa2fdcf17dc'
ASSET_HASH = '9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result.extend(differences(a.get(key), b.get(key), path+'/'+key))
        return result
    return [] if a == b else [dict(path=path, baseline=a, treatment=b)]


def run(command, log, cwd):
    print('RUN', command, flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f'Command failed ({status}); see {log}')


def audit_resolved(out):
    old = read(BASE/'configuration/official_env.json')
    new = read(out/'configuration/official_env.json')
    expected = copy.deepcopy(old)
    expected['commands']['base_velocity']['rel_standing_envs'] = .20
    assert not differences(expected, new), differences(expected, new)
    assert read(BASE/'configuration/official_agent.json') == read(out/'configuration/official_agent.json')
    result = dict(passed=True, expected_standing_probability=.20,
                  differences=differences(old, new))
    save(out/'resolved_configuration_check.json', result)


def audit_saved(out, seed, envs, iterations):
    reference = list(BASE.glob('logs/**/params'))
    actual = list(out.glob('logs/**/params'))
    assert len(reference) == len(actual) == 1
    changes = {}
    for name in ('env.yaml', 'agent.yaml'):
        before = yaml.load((reference[0]/name).read_text(), Loader=yaml.BaseLoader)
        after = yaml.load((actual[0]/name).read_text(), Loader=yaml.BaseLoader)
        expected = copy.deepcopy(before)
        expected['seed'] = str(seed)
        if name == 'env.yaml':
            expected['scene']['num_envs'] = str(envs)
            expected['scene']['terrain']['num_envs'] = str(envs)
            expected['commands']['base_velocity']['rel_standing_envs'] = '0.2'
            assert Path(after['log_dir']).is_relative_to(out)
            expected['log_dir'] = after['log_dir']
        else:
            expected['max_iterations'] = str(iterations)
        assert not differences(expected, after), differences(expected, after)
        changes[name] = differences(before, after)
    result = dict(passed=True, training_seed=seed, training_envs=envs, iterations=iterations,
                  standing_probability=.20, baseline_standing_probability=.02,
                  comparison='All saved environment/agent fields checked; only probability, seed, log path and explicit smoke budget may differ.', differences=changes)
    save(out/'configuration_check.json', result)


def stage(context, root, name, resume_validation=False):
    smoke = name == 'smoke'
    seed = 0 if smoke else int(name[-1])
    out = root / ('smoke' if smoke else f'run_{seed+1:03d}')
    envs, iterations = (128, 20) if smoke else (4096, 1500)
    if (out/'stage_complete.json').exists():
        result = read(out/'stage_complete.json')
        assert result['completed'] and result['protocol_sha256'] == PROTOCOL_HASH
        assert sha(Path((out/'final_checkpoint.txt').read_text().strip())) == result['model_sha256']
        audit_saved(out, seed, envs, iterations)
        print('ALREADY_COMPLETE', name, flush=True)
        return
    trained = (out/'training.log').exists()
    assert not trained or resume_validation, f'Partial run exists: {out}; diagnose rather than overwrite'
    if trained:
        assert len(list(out.glob(f'logs/**/model_{iterations-1}.pt'))) == 1
        assert 'Training time:' in (out/'training.log').read_text()
    out.mkdir(parents=True, exist_ok=True)
    source = out/'source'
    source.mkdir(exist_ok=True)
    if not trained:
        for filename in ('train_standing_comparison.py', 'inspect_g1.py', 'verify_training.py',
                         'evaluate_g1.py', 'evaluate_behaviors.py', 'protocol.json', 'environment.yaml'):
            shutil.copy2(context/filename, source/filename)
    else:
        shutil.copy2(context/'train_standing_comparison.py', source/'validation_train_standing_comparison.py')
    official = Path('/workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py')
    if not trained:
        shutil.copy2(official, source/'official_train_rsl_rl.py')
    assert sha(official) == sha(BASE/'source/official_train_rsl_rl.py')
    save(out/'source_manifest.json', {p.name: sha(p) for p in sorted(source.iterdir())})
    override = 'env.commands.base_velocity.rel_standing_envs=0.2'
    common = ['--task', 'Isaac-Velocity-Flat-G1', 'physics=newton_mjwarp']
    if not trained:
        run([sys.executable, str(source/'inspect_g1.py'), '--output', str(out/'configuration'), *common, override], out/'configuration.log', out)
    audit_resolved(out)
    train = [sys.executable, '/workspace/isaaclab/scripts/reinforcement_learning/train.py',
             '--rl_library', 'rsl_rl', '--seed', str(seed)]
    if smoke:
        train += ['--num_envs', str(envs), '--max_iterations', str(iterations)]
    if not trained:
        run(train + common + [f'env.scene.robot.spawn.usd_path={ASSET}', override], out/'training.log', out)
    audit_saved(out, seed, envs, iterations)
    run([sys.executable, str(source/'verify_training.py'), str(out), '--iterations', str(iterations),
         '--num_envs', str(envs)], out/'training_check.log', out)
    check = read(out/'training_check.json')
    assert check['passed']
    model = (out/'final_checkpoint.txt').read_text().strip()
    model_hash = sha(Path(model))
    assert model_hash == check['final_checkpoint_sha256']
    if smoke:
        run([sys.executable, str(source/'evaluate_g1.py'), '--model', model, '--output', str(out/'reload'),
             '--eval_seed', '10000', '--num_envs', '4', *common, f'env.scene.robot.spawn.usd_path={ASSET}'], out/'reload.log', out)
        assert read(out/'reload/summary.json')['passed']
        groups = []
    else:
        assert read(root/'smoke/stage_complete.json')['completed']
        groups = []
        for eval_seed in (20000, 21000):
            for condition in read(source/'protocol.json')['conditions']:
                dest = out/f'evaluation/eval_{eval_seed}/{condition}'
                dest.mkdir(parents=True, exist_ok=True)
                run([sys.executable, str(source/'evaluate_behaviors.py'), '--model', model,
                     '--protocol', str(source/'protocol.json'), '--condition', condition, '--output', str(dest),
                     '--eval_seed', str(eval_seed), *common, f'env.scene.robot.spawn.usd_path={ASSET}'], dest/'evaluation.log', out)
                result = read(dest/'summary.json')
                assert result['completed'] and result['num_envs'] == 32
                assert result['model_sha256'] == model_hash and result['protocol_sha256'] == PROTOCOL_HASH
                groups.append(dict(eval_seed=eval_seed, condition=condition, passed=result['pass_count'],
                                   early_terminations=sum(result['early_termination'])))
    save(out/'stage_complete.json', dict(completed=True, stage=name, training_seed=seed,
                                        model_sha256=model_hash, protocol_sha256=PROTOCOL_HASH, evaluations=groups))
    print('STAGE_COMPLETE', name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('all', 'smoke', 'seed0', 'seed1'), default='all')
    parser.add_argument('--resume-validation', action='store_true', help='Reuse completed training after a diagnosed validation failure; never rerun training')
    opts = parser.parse_args()
    context = Path(__file__).resolve().parent
    root = opts.output.resolve()
    assert root.is_relative_to(Path('/robotrl/experiments/phase3_g1_standing'))
    assert sha(ASSET) == ASSET_HASH and sha(context/'protocol.json') == PROTOCOL_HASH
    assert sha(context/'evaluate_behaviors.py') == sha(BASE/'source/evaluate_behaviors.py')
    for run_id in ('run_001', 'run_002'):
        baseline = read(BASE.parent/run_id/'summary.json')
        assert baseline['training_completed'] and baseline['groups_completed'] == 12
    for name in (('smoke', 'seed0', 'seed1') if opts.stage == 'all' else (opts.stage,)):
        if name != 'smoke':
            assert read(root/'smoke/stage_complete.json')['completed']
        stage(context, root, name, opts.resume_validation)
    print('REQUESTED_STAGES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
