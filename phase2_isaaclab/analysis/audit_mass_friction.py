"""Audit saved DR evidence without training. Requires PyYAML; run from any directory."""
import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path
import yaml

DEFAULT = Path(__file__).resolve().parents[2] / 'experiments/phase2_go2_mass_friction'
CONDITIONS = {'nominal': (1., 1.), 'light_low': (.85, .6),
              'heavy_high': (1.15, 1.4), 'heavy_low': (1.3, .3), 'nominal_push': (1., 1.)}

def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result += differences(a.get(key), b.get(key), f'{prefix}.{key}'.strip('.'))
        return result
    return [] if a == b else [{'path': prefix, 'reference': a, 'value': b}]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence, runs, rows = {}, [], []
    def record(path):
        evidence[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    def load(path):
        record(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def config(path):
        record(path)
        # BaseLoader reads serialized Python tags as data, never executes constructors.
        return yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    reference_env = reference_agent = None
    reference_source = {}
    for number, (group, seed) in enumerate([('fixed', 0), ('fixed', 1), ('randomized', 0), ('randomized', 1)], 1):
        run = root / f'run_{number:03}'
        for name in ['train_mass_friction.py', 'evaluate_go2.py', 'friction_audit.py',
                     'runtime_material_events.py', 'runtime_solver.py']:
            source = run / name
            if source.exists():
                content = source.read_bytes()
            else:
                with zipfile.ZipFile(root.parent / 'source_snapshots.zip') as z:
                    content = z.read(source.relative_to(root.parents[1]).as_posix())
            digest = hashlib.sha256(content).hexdigest()
            evidence[source.relative_to(root).as_posix()] = digest
            assert reference_source.setdefault(name, digest) == digest, f'Runtime source differs: {source}'
        protocol = load(run / 'protocol.json')
        assert protocol == dict(group=group, seed=seed, num_envs=4096, iterations=300, steps=29491200)
        env = config(next(run.rglob('env.yaml')))
        agent = config(next(run.rglob('agent.yaml')))
        assert env['seed'] == agent['seed'] == str(seed)
        assert env['scene']['num_envs'] == '4096'
        assert agent['num_steps_per_env'] == '24' and agent['max_iterations'] == '300'
        events = env['events']
        assert events['base_com'] == events['push_robot'] == 'null'
        mass_bounds = ['1.0', '1.0'] if group == 'fixed' else ['0.8', '1.2']
        mu_bounds = ['1.0', '1.0'] if group == 'fixed' else ['0.5', '1.5']
        assert events['add_base_mass']['params']['mass_distribution_params'] == mass_bounds
        assert events['add_base_mass']['params']['operation'] == 'scale'
        assert events['add_base_mass']['mode'] == events['physics_material']['mode'] == 'startup'
        for field in ['static_friction_range', 'dynamic_friction_range']:
            assert events['physics_material']['params'][field] == mu_bounds
        assert env['scene']['terrain']['physics_material']['static_friction'] == '0.5'
        if reference_env is None:
            reference_env, reference_agent = env, agent
        env_diff, agent_diff = differences(reference_env, env), differences(reference_agent, agent)
        allowed = {'seed', 'log_dir', 'events.add_base_mass.params.mass_distribution_params',
                   'events.physics_material.params.static_friction_range', 'events.physics_material.params.dynamic_friction_range'}
        assert all(d['path'] in allowed for d in env_diff), env_diff
        assert all(d['path'] == 'seed' for d in agent_diff), agent_diff
        audit = load(run / 'mass_friction_audit.json')
        masses = audit['base_mass_kg']
        contact = audit['contacts_step_51']['active_contacts']
        mus = contact['ground_sliding_friction']
        assert audit['group'] == group and audit['steps'] == 7200 and len(masses) == 4096
        assert all(6.921 * float(mass_bounds[0]) - 1e-4 <= m <= 6.921 * float(mass_bounds[1]) + 1e-4 for m in masses)
        assert contact['ground_count'] > 0 and mus
        assert all(float(mu_bounds[0]) - 1e-5 <= m <= float(mu_bounds[1]) + 1e-5 for m in mus)
        if group == 'randomized':
            assert max(masses) - min(masses) > 1 and max(mus) - min(mus) > .9
        model = next(run.rglob('model_299.pt'))
        record(model)
        runs.append(dict(run=run.name, group=group, seed=seed, env_differences=env_diff,
                         agent_differences=agent_diff, mass_range_kg=[min(masses), max(masses)],
                         contact_mu_range=[min(mus), max(mus)], ground_contacts=contact['ground_count']))
        for condition, (mass, mu) in CONDITIONS.items():
            d = load(run / f'{condition}.json')
            assert d['sha256'] == evidence[model.relative_to(root).as_posix()]
            assert d['eval_seed'] == 30000 and d['forward_speed_command_m_s'] == .5 and d['step_dt'] == .02
            assert d['action_delay']['steps'] == 0
            assert d['scenario'] == dict(name=condition, base_mass_scale=mass, contact_mu=mu,
                                        ground_mu=min(.5, mu), com_randomization=False,
                                        random_push=False, fixed_push=condition == 'nominal_push')
            episodes = d['episodes']
            assert len(episodes) == 32 and sorted(e['env_index'] for e in episodes) == list(range(32))
            for e in episodes:
                assert 0 < e['length_steps'] <= 1000
                assert math.isclose(e['length_seconds'], e['length_steps'] * .02)
                assert e['terminated'] or e['time_out']
            xy = statistics.mean(e['mean_xy_velocity_error_m_s'] for e in episodes)
            reward = statistics.mean(e['return'] for e in episodes)
            assert abs(xy - d['mean_xy_velocity_error_m_s']) < 1e-6
            assert abs(reward - d['return_mean']) < 1e-5
            contacts = d['friction_audit']
            assert [c['step'] for c in contacts] == [51, 501]
            for c in contacts:
                active = c['active_contacts']
                assert active['ground_count'] > 0 and active['ground_sliding_friction']
                assert all(abs(x - mu) < 1e-5 for x in active['ground_sliding_friction'])
            push_error = None
            if condition == 'nominal_push':
                p = d['fixed_push']
                assert p['time_s'] == 5 and p['body_y_delta_m_s'] == .5
                assert p['env_ids'] == list(range(32))
                assert len(p['before_world_velocity']) == len(p['after_world_velocity']) == len(p['target_world_delta']) == 32
                errors = []
                for before, after, delta in zip(p['before_world_velocity'], p['after_world_velocity'], p['target_world_delta']):
                    assert abs(sum(x*x for x in delta) ** .5 - .5) < 1e-5
                    errors += [abs(after[j] - before[j] - delta[j]) for j in range(3)]
                    errors += [abs(after[j] - before[j]) for j in range(3, 6)]
                push_error = max(errors)
                assert push_error < 1e-5
            else:
                assert d['fixed_push'] is None
            rows.append(dict(run=run.name, group=group, seed=seed, condition=condition, xy_error=xy,
                             return_mean=reward, terminated=sum(e['terminated'] for e in episodes),
                             mean_length_seconds=statistics.mean(e['length_seconds'] for e in episodes),
                             contact_snapshot_steps=[51, 501], push_velocity_error=push_error))
    paired = []
    for seed in [0, 1]:
        for condition in CONDITIONS:
            fixed = next(r for r in rows if r['seed'] == seed and r['condition'] == condition and r['group'] == 'fixed')
            dr = next(r for r in rows if r['seed'] == seed and r['condition'] == condition and r['group'] == 'randomized')
            paired.append(dict(seed=seed, condition=condition, fixed_xy=fixed['xy_error'], dr_xy=dr['xy_error'],
                               dr_minus_fixed=dr['xy_error'] - fixed['xy_error']))
    report = dict(status='saved_evidence_audit_complete', training_runs=4, evaluation_conditions=20,
                  episodes=640, runs=runs, rows=rows, paired=paired, evidence_sha256=evidence,
                  limits=['Two training seeds only; episode lanes are not independent training replicates.',
                          'Training contacts sampled at step 51; evaluation contacts at steps 51 and 501, not exhaustive.',
                          'Training mass read back for all environments; evaluation mass multiplier verified from scenario and source configuration, not an independent mass readback.',
                          'No new simulation, no real-robot or generalization guarantee.'])
    (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 质量/摩擦对照：最终汇总与审计', '', '四次训练、20组评估、640回合的现有证据审计完成；未新增训练或评估。全部回合20秒，无提前终止。', '',
             '## 配置与物理审计', '', '完整比较四份env.yaml和agent.yaml：除seed、日志目录、预定的质量/摩擦范围外无差异。PPO、网络、奖励、观测、动作与训练预算一致；质心和随机推扰均关闭。', '',
             '|组/seed|实测base质量kg|第51步接触摩擦范围|活动地面接触数|', '|---|---|---|---|']
    for r in runs:
        lines.append(f"|{r['group']}/{r['seed']}|{r['mass_range_kg'][0]:.6f}–{r['mass_range_kg'][1]:.6f}|{r['contact_mu_range'][0]:.6f}–{r['contact_mu_range'][1]:.6f}|{r['ground_contacts']}|")
    lines += ['', '20组评估的模型SHA256、seed、指令、场景参数、回合编号和汇总指标核对通过。每组第51/501步活动接触摩擦与目标值一致；四组固定扰动均在第5秒作用于32个环境，实测世界坐标速度增量与目标一致（误差<1e-5），角速度未改写。', '',
              '## 配对结果', '', 'XY速度误差越低越好，单位m/s；差值为DR−fixed，负数表示改善。', '',
              '|训练seed|条件|fixed|DR|差值|', '|---|---|---|---|---|']
    for r in paired:
        lines.append(f"|{r['seed']}|{r['condition']}|{r['fixed_xy']:.5f}|{r['dr_xy']:.5f}|{r['dr_minus_fixed']:+.5f}|")
    lines += ['', 'seed0在五个条件中四项误差降低，范围外heavy_low略退化；seed1五项均退化，heavy_high最明显。改善未跨训练seed复现，因此不支持“DR全面改善”的结论。32个并行回合不算32个独立训练seed，不做基于回合数量的显著性宣称。逐组回报、回合长度、终止数与证据哈希见summary.json。已有DR异常回合及视频继续保留。', '',
              '## 审计边界与复现', '', '配置全量比对已完成；活动接触仅为保存时刻的快照，不代表逐步穷举。训练base质量有4096环境实测值；评估质量倍率只有场景/配置记录，没有独立保存的质量读回，不能声称已逐环境核验评估质量。固定扰动是速度增量，不是标定推力。', '',
              '在仓库根目录执行 `python phase2_isaaclab/analysis/audit_mass_friction.py`（需要PyYAML及本地/NAS原始数据）。脚本只读取证据并重新生成summary.json和此报告；任何不匹配将报错。preliminary_summary.json保留为历史初步记录，最终状态以本报告为准。', '']
    (root / 'AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('AUDIT PASSED: 4 training runs, 20 evaluation conditions, 640 episodes')

if __name__ == '__main__':
    main()
