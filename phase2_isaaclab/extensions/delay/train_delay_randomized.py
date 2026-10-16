"""Use the official trainer unchanged, wrapping task actions with episode-wise delay."""
import json
from pathlib import Path
import runpy
import warp as wp
wp.config.enable_backward = False
import gymnasium as gym
import torch
from random_action_delay import RandomActionDelay

class DelayedCommands(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        base=env.unwrapped
        self.delay=RandomActionDelay(base.num_envs,2,base.device,seed=73042)
        self.steps=0
        self.audit=[]
        self.first_delays=self.delay.delays[:16].cpu().tolist()

    def reset(self, **kwargs):
        result=self.env.reset(**kwargs)
        self.delay.reset(torch.ones(self.unwrapped.num_envs,dtype=torch.bool,device=self.unwrapped.device))
        return result

    def step(self, actions):
        applied=self.delay.apply(actions)
        if self.steps<8:
            self.audit.append({'step':self.steps,'delays':self.delay.delays[:16].cpu().tolist(),
                               'commanded':actions[:16].detach().cpu().tolist(),
                               'applied':applied[:16].detach().cpu().tolist()})
        result=self.env.step(applied)
        done=result[2] | result[3]
        if self.steps<8: self.audit[-1]['done']=done[:16].cpu().tolist()
        self.delay.reset(done)
        self.steps+=1
        return result

    def close(self):
        report={'policy_steps':self.steps,'sampled_delay_counts':self.delay.histogram.cpu().tolist(),
                'delay_steps':[0,1,2],'control_step_seconds':self.unwrapped.step_dt,
                'distribution':'uniform integers per episode; independent RNG seed 73042',
                'previous_action_observation':'applied action', 'initial_queue':'zeros',
                'audit_first_16_envs':self.audit}
        Path('delay_training_audit.json').write_text(json.dumps(report,indent=2))
        self.env.close()

original_make=gym.make
def make(*args,**kwargs):
    env=original_make(*args,**kwargs)
    if args and args[0]=='Isaac-Velocity-Flat-UnitreeGo2':
        return DelayedCommands(env)
    return env

if __name__=='__main__':
    gym.make=make
    runpy.run_path('/workspace/isaaclab/scripts/reinforcement_learning/train.py',run_name='__main__')
