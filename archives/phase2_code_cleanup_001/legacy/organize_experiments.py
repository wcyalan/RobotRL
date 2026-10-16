"""Prepare a verified backup and manifest; finalize metadata after PowerShell moves."""
from pathlib import Path
import json
import re
import sys
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archives/experiment_cleanup_001'
REMOVED = {
    'phase1_ppo_continuous/run_001': '102400-step preliminary training; superseded by two full runs',
    'phase1_sb3_continuous/run_001': '10240-step integration smoke test',
    'phase1_sb3_continuous/run_002': 'interrupted training and recovery test',
    'phase2_go2_smoke/run_002': 'training integration smoke test',
    'phase2_go2_pilot/run_004': 'reduced-scale pilot, superseded by full official baseline',
    'phase2_go2_diagnostic/run_005': 'pilot diagnostic',
    'phase2_go2_control/run_008': 'control diagnostic with capacity overflow',
    'phase2_go2_control/run_009': 'confirmatory control diagnostic',
    'phase1_sb3_continuous/recovery_verification.json': 'recovery integration check',
    'phase1_ppo_continuous/baseline_evaluation.log': 'preliminary evaluation console log',
}
RENAMED = {
    'phase1_ppo_continuous/run_002': 'phase1_ppo_continuous/run_001',
    'phase1_ppo_continuous/run_003': 'phase1_ppo_continuous/run_002',
    'phase1_sb3_continuous/run_003': 'phase1_sb3_continuous/run_001',
    'phase1_sb3_continuous/run_004': 'phase1_sb3_continuous/run_002',
    'phase2_go2_baseline/run_010': 'phase2_go2_baseline/run_001',
}

def prepare():
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    backup = ARCHIVE / 'experiments_before_cleanup.zip'
    with zipfile.ZipFile(backup, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT / 'experiments').rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))
    with zipfile.ZipFile(backup) as z:
        assert z.testzip() is None
        for p in (ROOT / 'experiments').rglob('*'):
            if p.is_file():
                assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).digest()
    manifest = {'backup': str(backup.relative_to(ROOT)), 'backup_verified': True,
                'removed': REMOVED, 'renamed': RENAMED,
                'scope': 'Local experiment directories only. Remote original runs and historical source/log contents preserved.'}
    (ARCHIVE / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

def finalize():
    def remap(text):
        replacements = {}
        for old, new in RENAMED.items():
            for sep in ['/', '\\', '\\\\']:
                replacements[('experiments/' + old).replace('/',sep)] = ('experiments/' + new).replace('/',sep)
        return re.sub('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)), lambda m: replacements[m.group()], text)

    # Historical training snapshots and console logs retain original paths.
    for p in (ROOT / 'experiments').rglob('*'):
        if not p.is_file() or p.suffix not in ['.json', '.md']: continue
        if 'source' in p.parts or 'official_go2_source' in p.parts: continue
        p.write_text(remap(p.read_text(encoding='utf-8')), encoding='utf-8')

    p = ROOT / 'experiments/phase1_ppo_continuous/comparison_001.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    # The run field contains the original short ID, unlike full paths above.
    d['results'] = [row for row in d['results'] if row['run'] != 'run_001']
    for row in d['results']:
        row['run'] = {'run_002':'run_001','run_003':'run_002'}[row['run']]
    d['training_protocol']['note'] = 'Two full runs retained; preliminary 102400-step run archived in archives/experiment_cleanup_001.'
    def keys_remap(obj):
        if isinstance(obj, dict):
            return {{'run_002':'run_001','run_003':'run_002'}.get(k,k):keys_remap(v) for k,v in obj.items()}
        if isinstance(obj, list): return [keys_remap(x) for x in obj]
        return obj
    d = keys_remap(d)
    p.write_text(json.dumps(d,indent=2), encoding='utf-8')

    p = ROOT / 'phase2_isaaclab/remote_validation.json'
    d = json.loads(remap(p.read_text(encoding='utf-8')))
    def archived_paths(obj):
        if isinstance(obj, dict): return {k:archived_paths(v) for k,v in obj.items()}
        if isinstance(obj, list): return [archived_paths(v) for v in obj]
        if isinstance(obj, str):
            norm = obj.replace('\\','/')
            if any(norm.startswith('experiments/'+old) for old in REMOVED):
                return 'archives/experiment_cleanup_001/experiments_before_cleanup.zip::' + norm
        return obj
    d = archived_paths(d)
    d['local_cleanup_manifest'] = 'archives/experiment_cleanup_001/manifest.json'
    p.write_text(json.dumps(d,indent=2),encoding='utf-8')

    p = ROOT / 'phase2_isaaclab/render_go2_trajectory.py'
    text = remap(p.read_text(encoding='utf-8'))
    text = re.sub(r"joint_names = json.loads\(.*\)\['joint_names'\]", "joint_names = [f'{leg}_{joint}_joint' for leg in ['FL', 'FR', 'RL', 'RR'] for joint in ['hip', 'thigh', 'calf']]", text)
    p.write_text(text,encoding='utf-8')

    lines = ['# 实验结果索引', '', '各算法目录独立从 `run_001` 连续编号。保留正式训练及独立评估、模型、曲线和回放；低回报不等于运行失败，因此完整 SB3 对照仍保留。', '',
             '失败、短测试和诊断记录已从主目录清出，完整原件见 [备份与编号映射](../archives/experiment_cleanup_001/manifest.json)。远程原始编号不变；历史日志和源码快照中的旧编号可据此追溯。', '',
             '| 实验 | 编号 | 训练步数 | 结果说明 |', '|---|---|---:|---|']
    catalog = []
    for group in sorted((ROOT / 'experiments').iterdir()):
        if not group.is_dir(): continue
        for run in sorted(group.glob('run_*')):
            summary = run / 'summary.json'
            if summary.exists():
                d = json.loads(summary.read_text(encoding='utf-8'))
                steps = d.get('steps',d.get('last_evaluation',{}).get('steps',d.get('final_training_steps','—')))
                best = d.get('best_eval_mean',d.get('final_eval_mean'))
                note = f'训练期间最佳评估均值 {best:.2f}' if best is not None else d.get('status','已保存')
                if not (run/'config.json').exists(): note += '；原始日志记录，无模型'
            elif group.name == 'phase2_go2_baseline':
                steps = 29491200
                note = '完整官方基线；0.5 m/s 指令实测均速 0.511 m/s，20 秒视频已保存'
            else:
                steps, note = '—', '初始环境基线，保留模型及曲线'
            rel = run.relative_to(ROOT / 'experiments').as_posix()
            lines.append(f'| {group.name} | [{run.name}]({rel}/) | {steps} | {note} |')
            catalog.append({'directory':rel,'steps':steps,'note':note})
    lines += ['', 'Go2 视频：[20 秒前进回放](phase2_go2_baseline/run_001/videos/go2_forward_20s.mp4)。', '',
              '表中训练期间最佳评估用于索引，不代替独立测试结果；详细结果保留在各运行的评估文件中。']
    (ROOT / 'experiments/README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT / 'experiments/catalog.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False),encoding='utf-8')
    for p in (ROOT / 'experiments').rglob('*.json'):
        json.loads(p.read_text(encoding='utf-8'))
    for group in (ROOT / 'experiments').iterdir():
        if group.is_dir():
            names = sorted(p.name for p in group.glob('run_*'))
            assert names == [f'run_{i:03d}' for i in range(1,len(names)+1)], (group,names)
    print(f'Validated JSON and sequential numbering for {len(catalog)} retained runs.')

if __name__ == '__main__':
    {'prepare':prepare,'finalize':finalize}[sys.argv[1]]()
