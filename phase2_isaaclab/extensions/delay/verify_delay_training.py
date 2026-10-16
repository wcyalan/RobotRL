"""Verify saved wrapper trace against an independent Python-list latency model."""
import json
from pathlib import Path
d=json.loads(Path('delay_training_audit.json').read_text())
assert d['policy_steps']>0 and len(d['sampled_delay_counts'])==3
assert all(n>0 for n in d['sampled_delay_counts'])
history=None
for row in d['audit_first_16_envs']:
    if history is None: history=[[] for _ in row['commanded']]
    for env, command in enumerate(row['commanded']):
        history[env].append(command)
        lag=row['delays'][env]
        expected=history[env][-1-lag] if len(history[env])>lag else [0.0]*len(command)
        assert row['applied'][env]==expected,(row['step'],env)
        if row['done'][env]: history[env]=[]
print('DELAY_TRAINING_AUDIT_PASSED',d['policy_steps'],d['sampled_delay_counts'])
