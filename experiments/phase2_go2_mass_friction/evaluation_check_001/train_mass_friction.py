"""Official trainer with an explicit, audited mass/friction-only intervention."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import warp as wp
wp.config.enable_backward = False
import gymnasium as gym
import torch
from friction_audit import snapshot

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dr_group', choices=['fixed', 'randomized'], required=True)
opts, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

class AuditWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.contacts = None

    def step(self, actions):
        result = self.env.step(actions)
        self.steps += 1
        if self.steps == 51:
            self.contacts = snapshot(self.unwrapped)
        return result

    def close(self):
        base = self.unwrapped
        robot = base.scene['robot']
        ids, _ = robot.find_bodies('base')
        mass = robot.data.body_mass.torch[:, ids].flatten().detach().cpu()
        report = {'group': opts.dr_group, 'steps': self.steps,
                  'base_mass_kg': mass.tolist(), 'contacts_step_51': self.contacts}
        assert self.steps >= 51 and self.contacts is not None
        ground = self.contacts['active_contacts']['ground_sliding_friction']
        assert ground and all(0.4999 <= mu <= 1.5001 for mu in ground)
        if opts.dr_group == 'fixed':
            assert max(ground)-min(ground) < 1e-5 and abs(ground[0]-1.0) < 1e-5
            assert float(mass.max()-mass.min()) < 1e-5
        else:
            assert max(ground)-min(ground) > 0.05
            assert float(mass.max()-mass.min()) > 0.05
        Path('mass_friction_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('MASS_FRICTION_AUDIT_PASSED', opts.dr_group, self.steps, flush=True)
        self.env.close()

original_make = gym.make
def make(*args, **kwargs):
    if not args or args[0] != 'Isaac-Velocity-Flat-UnitreeGo2':
        return original_make(*args, **kwargs)
    cfg = kwargs['cfg']
    cfg.events.base_com = None
    cfg.events.push_robot = None
    cfg.events.add_base_mass.params['mass_distribution_params'] = (1.0,1.0) if opts.dr_group == 'fixed' else (0.8,1.2)
    cfg.events.add_base_mass.params['distribution'] = 'uniform'
    cfg.events.add_base_mass.params['operation'] = 'scale'
    bounds = (1.0,1.0) if opts.dr_group == 'fixed' else (0.5,1.5)
    cfg.events.physics_material.params['static_friction_range'] = bounds
    cfg.events.physics_material.params['dynamic_friction_range'] = bounds
    cfg.scene.terrain.physics_material.static_friction = 0.5
    cfg.scene.terrain.physics_material.dynamic_friction = 0.5
    return AuditWrapper(original_make(*args, **kwargs))

gym.make = make
runpy.run_path('/workspace/isaaclab/scripts/reinforcement_learning/train.py', run_name='__main__')
