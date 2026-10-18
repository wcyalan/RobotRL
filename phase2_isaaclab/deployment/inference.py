"""Pure PyTorch inference interface for the frozen Go2 actor (no simulator/RSL-RL)."""
from pathlib import Path
import torch

JOINTS = [f'{leg}_{joint}_joint' for leg in ['FL','FR','RL','RR'] for joint in ['hip','thigh','calf']]
DEFAULT_POSITION = [.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5]

def observation(linear_b, angular_b, gravity_b, command, joint_position, joint_velocity, previous_action):
    """All tensors [batch, dim], float32; joint tensors use JOINTS order. No added sensor noise."""
    parts = [linear_b, angular_b, gravity_b, command, joint_position, joint_velocity, previous_action]
    batch = linear_b.shape[0]
    for p, dim in zip(parts, [3,3,3,3,12,12,12]):
        assert p.shape == (batch,dim) and p.dtype == torch.float32 and torch.isfinite(p).all()
    default = joint_position.new_tensor(DEFAULT_POSITION)
    return torch.cat([linear_b, angular_b, gravity_b, command, joint_position-default, joint_velocity, previous_action],dim=-1)

class Go2Policy:
    def __init__(self, path: str | Path, device='cpu'):
        self.device = torch.device(device)
        self.actor = torch.jit.load(str(path), map_location=self.device).eval()

    @torch.inference_mode()
    def __call__(self, obs):
        assert obs.ndim == 2 and obs.shape[1] == 48 and obs.dtype == torch.float32
        assert torch.isfinite(obs).all()
        action = self.actor(obs.to(self.device))
        assert action.shape == (obs.shape[0],12) and torch.isfinite(action).all()
        return action

    @staticmethod
    def position_targets(action):
        # No new clipping: preserve the trained simulation controller's semantics.
        return action.new_tensor(DEFAULT_POSITION) + .25*action
