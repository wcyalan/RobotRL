"""Fixed command latency in policy steps; zero action means nominal joint target."""
from collections import deque
import torch

class ActionDelay:
    def __init__(self, steps):
        if steps < 0:
            raise ValueError('Delay must be nonnegative')
        self.steps = steps
        self.queue = None

    def apply(self, action):
        if self.queue is None:
            self.queue = deque(torch.zeros_like(action) for _ in range(self.steps))
        self.queue.append(action.clone())
        return self.queue.popleft()

    def reset(self, done):
        for action in self.queue:
            action[done] = 0
