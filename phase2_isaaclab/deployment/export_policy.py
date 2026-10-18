"""Export the frozen Go2 actor to TorchScript; no RSL-RL dependency at inference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
config_dir = args.checkpoint.parent/'params'
agent_text = (config_dir/'agent.yaml').read_text(encoding='utf-8')
# This exporter deliberately supports only the audited frozen architecture.
assert agent_text.count('activation: elu') == 2
assert agent_text.count('obs_normalization: false') == 2 and 'clip_actions: null' in agent_text
args.output.mkdir(parents=True, exist_ok=False)
for name in ['agent.yaml','env.yaml']:
    shutil.copyfile(config_dir/name,args.output/name)
state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
assert set(state) == {'distribution.std_param', *[f'mlp.{i}.{k}' for i in [0,2,4,6] for k in ['weight','bias']]}
actor = nn.Sequential(nn.Linear(48,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,12)).eval()
actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')}, strict=True)
policy = torch.jit.script(actor)
path = args.output/'policy.pt'
policy.save(str(path))
loaded = torch.jit.load(str(path)).eval()
torch.manual_seed(20261018)
maximum = 0.
with torch.inference_mode():
    for n in [1,32,256]:
        for scale in [.1,1.,5.]:
            x=torch.randn(n,48)*scale
            a,b=actor(x),loaded(x)
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            maximum=max(maximum,(a-b).abs().max().item())
report={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'config_sha256':{name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['agent.yaml','env.yaml']},
        'policy_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'torch_version':torch.__version__,
        'input_shape':['batch',48],'output_shape':['batch',12],'dtype':'float32',
        'observation_normalization':False,'deterministic_mean_action':True,'clip_actions':None,
        'synthetic_max_abs_error':maximum,'synthetic_tolerance':{'atol':1e-5,'rtol':1e-5},
        'status':'Synthetic reconstruction check passed; official runtime equivalence and closed-loop validation still required.'}
(args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
