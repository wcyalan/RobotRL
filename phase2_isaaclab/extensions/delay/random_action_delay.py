"""Episode-wise command latency with an RNG independent of policy initialization."""
import torch

class RandomActionDelay:
    def __init__(self, num_envs, max_steps, device, seed):
        self.num_envs, self.max_steps, self.device = num_envs, max_steps, device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.delays = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.buffer = None
        self.histogram = torch.zeros(max_steps+1, dtype=torch.long, device=device)
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, done):
        ids = done.nonzero(as_tuple=False).flatten()
        values = torch.randint(self.max_steps+1, (ids.numel(),), device=self.device, generator=self.generator)
        self.delays[ids] = values
        self.histogram += torch.bincount(values, minlength=self.max_steps+1)
        if self.buffer is not None:
            self.buffer[:,ids] = 0

    def apply(self, action):
        if self.buffer is None:
            self.buffer = torch.zeros((self.max_steps+1, *action.shape), device=action.device, dtype=action.dtype)
        self.buffer[1:] = self.buffer[:-1].clone()
        self.buffer[0] = action
        return self.buffer[self.delays, torch.arange(self.num_envs, device=self.device)].clone()
