"""Official PPO trainer with a command-distribution-only intervention."""
import argparse
import inspect
import json
from pathlib import Path
import runpy
import sys

import warp as wp
wp.config.enable_backward = False
import gymnasium as gym
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import RewardTermCfg

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--coverage_variant', choices=['mixed', 'turn_only', 'turn_stop', 'stop_sequence'], default='mixed')
opts, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
MOTION, STANDING, INTERVAL = (.6, .2, 5.) if opts.coverage_variant == 'mixed' else (.78, .02, 10.)
if opts.coverage_variant == 'stop_sequence':
    MOTION, STANDING = .725, .025


def zero_command_yaw_l2(env):
    command = env.command_manager.get_command('base_velocity')
    standing = torch.all(torch.abs(command) < 1.e-6, dim=-1)
    yaw = env.scene['robot'].data.root_ang_vel_b.torch[:, 2]
    return yaw.square() * standing


class CoverageCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sample_counts = torch.zeros(3, dtype=torch.long, device=self.device)
        self.transition_counts = torch.zeros(3, 3, dtype=torch.long, device=self.device)
        self.seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.turn_sign_counts = torch.zeros(2, dtype=torch.long, device=self.device)
        self.sequence_mask = torch.arange(self.num_envs, device=self.device) % 5 == 0 if opts.coverage_variant == 'stop_sequence' else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ids = ids[~self.sequence_mask[ids]]
        u = torch.rand(len(ids), device=self.device)
        mode = torch.where(u < MOTION, 0, torch.where(u < MOTION + STANDING, 1, 2))
        old = self.mode[ids]
        seen = self.seen[ids]
        self.transition_counts += torch.bincount(old[seen] * 3 + mode[seen], minlength=9).reshape(3, 3)
        self.sample_counts += torch.bincount(mode, minlength=3)
        self.mode[ids] = mode
        self.seen[ids] = True
        self.is_standing_env[ids] = mode == 1
        self.is_heading_env[ids] = mode == 0
        turns = ids[mode == 2]
        self.vel_command_b[turns, :2] = 0.
        # Both directions, broad nonzero magnitudes; not just the evaluation target.
        magnitude = torch.empty(len(turns), device=self.device).uniform_(.3, .8)
        sign = torch.where(torch.rand(len(turns), device=self.device) < .5, -1., 1.)
        self.vel_command_b[turns, 2] = magnitude * sign
        self.turn_sign_counts += torch.stack(((sign > 0).sum(), (sign < 0).sum()))

    def _update_command(self):
        super()._update_command()
        if opts.coverage_variant == 'stop_sequence':
            ids = self.sequence_mask.nonzero(as_tuple=False).flatten()
            phase = (self._env.episode_length_buf[ids] // round(5. / self._env.step_dt)) % 2
            self.vel_command_b[ids] = 0.
            self.vel_command_b[ids, 0] = (phase == 0).float() * .5
            self.is_heading_env[ids] = False
            self.is_standing_env[ids] = phase == 1
            self.mode[ids] = phase


class AuditWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.exposure = torch.zeros(4, dtype=torch.long, device=env.unwrapped.device)
        self.sequence_exposure = torch.zeros(2, dtype=torch.long, device=env.unwrapped.device)
        self.sequence_switches = torch.zeros(2, dtype=torch.long, device=env.unwrapped.device)
        self.previous_episode_steps = None
        self.previous_sequence_phase = None

    def step(self, actions):
        # Count the command governing this transition, before resampling/reset.
        term = self.unwrapped.command_manager.get_term('base_velocity')
        category = torch.where(term.mode == 2, torch.where(term.command[:, 2] > 0, 2, 3), term.mode)
        self.exposure += torch.bincount(category, minlength=4)
        if opts.coverage_variant == 'stop_sequence':
            episode_steps = self.unwrapped.episode_length_buf[term.sequence_mask].clone()
            phase = (episode_steps // round(5. / self.unwrapped.step_dt)) % 2
            self.sequence_exposure += torch.bincount(phase, minlength=2)
            if self.previous_episode_steps is not None:
                changed = (episode_steps > self.previous_episode_steps) & (phase != self.previous_sequence_phase)
                self.sequence_switches += torch.bincount(phase[changed], minlength=2)
            self.previous_episode_steps, self.previous_sequence_phase = episode_steps, phase
        result = self.env.step(actions)
        self.steps += 1
        term = self.unwrapped.command_manager.get_term('base_velocity')
        command = term.command
        if opts.coverage_variant == 'stop_sequence':
            assert torch.allclose(result[0]['policy'][:, 9:12], command)
            ids = term.sequence_mask
            phase = (self.unwrapped.episode_length_buf[ids] // round(5. / self.unwrapped.step_dt)) % 2
            assert torch.allclose(command[ids, 0], (phase == 0).float() * .5)
            assert (command[ids, 1:] == 0).all()
        assert torch.isfinite(command).all()
        assert (command[term.mode == 1] == 0).all()
        turn = command[term.mode == 2]
        assert (turn[:, :2] == 0).all()
        assert ((turn[:, 2].abs() >= .2999) & (turn[:, 2].abs() <= .8001)).all()
        return result

    def close(self):
        term = self.unwrapped.command_manager.get_term('base_velocity')
        report = {'steps': self.steps, 'mode_order': ['original_heading_motion', 'standing', 'pure_turn'],
                  'variant': opts.coverage_variant,
                  'sequence_lanes': int(term.sequence_mask.sum()),
                  'sequence_exposure_forward_stop': self.sequence_exposure.cpu().tolist(),
                  'sequence_switches_restart_stop_excluding_resets': self.sequence_switches.cpu().tolist(),
                  'sample_count_note': 'Random command counts exclude dedicated sequence lanes.',
                  'sample_counts': term.sample_counts.cpu().tolist(),
                  'turn_sign_order': ['left', 'right'],
                  'turn_sign_sample_counts': term.turn_sign_counts.cpu().tolist(),
                  'exposure_order': ['original_heading_motion', 'standing', 'left_turn', 'right_turn'],
                  'exposure_environment_steps': self.exposure.cpu().tolist(),
                  'exposure_environment_seconds': (self.exposure * self.unwrapped.step_dt).cpu().tolist(),
                  'exposure_fractions': (self.exposure / self.exposure.sum()).cpu().tolist(),
                  'transition_counts_including_episode_resets': term.transition_counts.cpu().tolist()}
        assert self.steps > 0 and min(report['sample_counts']) > 0
        assert min(report['turn_sign_sample_counts']) > 0
        assert sum(report['exposure_environment_steps']) == self.steps * self.unwrapped.num_envs
        if opts.coverage_variant == 'stop_sequence':
            assert min(report['sequence_switches_restart_stop_excluding_resets']) > 0
        Path('command_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('COMMAND_COVERAGE_AUDIT_PASSED', report, flush=True)
        self.env.close()


original_make = gym.make


def make(*args, **kwargs):
    if not args or args[0] != 'Isaac-Velocity-Flat-UnitreeGo2':
        return original_make(*args, **kwargs)
    cfg = kwargs['cfg']
    before = cfg.to_dict()
    command = cfg.commands.base_velocity
    assert command.heading_command and command.rel_heading_envs == 1.
    command.class_type = CoverageCommand
    command.resampling_time_range = (INTERVAL, INTERVAL)
    command.rel_standing_envs = STANDING
    if opts.coverage_variant == 'turn_stop':
        cfg.rewards.zero_command_yaw_l2 = RewardTermCfg(func=zero_command_yaw_l2, weight=-2.0)
    after = cfg.to_dict()
    assert all(before[key] == after[key] for key in before if key not in ['commands', 'rewards'])
    reward_after = dict(after['rewards'])
    if opts.coverage_variant == 'turn_stop':
        assert 'zero_command_yaw_l2' not in before['rewards']
        reward_after.pop('zero_command_yaw_l2')
    assert before['rewards'] == reward_after
    Path('command_intervention.json').write_text(json.dumps({
        'variant': opts.coverage_variant,
        'probabilities': {'original_heading_motion': MOTION, 'standing': STANDING, 'pure_turn': 1-MOTION-STANDING},
        'sequence': {'lane_rule': 'env_id % 5 == 0', 'phase_seconds': 5, 'forward_speed': .5, 'clock': 'episode_length_buf; resets restart forward'} if opts.coverage_variant == 'stop_sequence' else None,
        'resampling_seconds': INTERVAL, 'turn_abs_yaw_range': [.3, .8], 'turn_sign_probability': .5,
        'before_commands': before['commands'], 'after_commands': after['commands'],
        'other_environment_config_equal': opts.coverage_variant != 'turn_stop',
        'existing_rewards_and_other_environment_config_equal': True,
        'added_reward': {'zero_command_yaw_l2': -2.0} if opts.coverage_variant == 'turn_stop' else {},
        'note': 'Official DR, existing rewards, terminations and PPO retained; only declared intervention added. See resume_audit.json when resuming a checkpoint.'
    }, indent=2, default=str), encoding='utf-8')
    Path('official_velocity_command.py').write_text(inspect.getsource(UniformVelocityCommand), encoding='utf-8')
    return AuditWrapper(original_make(*args, **kwargs))


gym.make = make
runpy.run_path('/workspace/isaaclab/scripts/reinforcement_learning/train.py', run_name='__main__')
