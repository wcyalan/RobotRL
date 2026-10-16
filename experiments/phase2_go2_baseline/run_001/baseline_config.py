"""Record the pinned image's resolved official defaults without starting simulation."""
import json
import importlib.metadata as metadata
from pathlib import Path
from isaaclab_rl.entrypoints.simple_agents import _parse_args
from isaaclab_tasks.utils import resolve_task_config
args = _parse_args(['--task', 'Isaac-Velocity-Flat-UnitreeGo2', 'physics=newton_mjwarp'], 'zero')
env, agent = resolve_task_config(args.task, 'rsl_rl_cfg_entry_point')
data = {'task': args.task, 'num_envs': env.scene.num_envs, 'max_iterations': agent.max_iterations,
        'num_steps_per_env': agent.num_steps_per_env, 'seed': agent.seed,
        'total_environment_steps': env.scene.num_envs * agent.max_iterations * agent.num_steps_per_env,
        'versions': {p: metadata.version(p) for p in ['torch', 'newton', 'warp-lang', 'rsl-rl-lib']},
        'training_overrides': {'physics': 'newton_mjwarp', 'asset_location_only': '/robotrl/assets/Go2/go2.usd'},
        'source_asset_url': env.scene.robot.spawn.usd_path,
        'note': 'No overrides of training size, seed, PPO, rewards, terminations, actuators, or contact capacity.'}
Path('official_defaults.json').write_text(json.dumps(data, indent=2))
print('OFFICIAL_DEFAULTS', data, flush=True)
