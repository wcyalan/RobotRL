"""策略与价值网络：先实现 CartPole 的 PolicyNetwork，再加入 ValueNetwork。"""

# TODO：实现 PolicyNetwork，输入 [..., 4]，输出动作 logits [..., 2]。
# TODO：检查采样动作与 log-prob 的形状，以及 log-prob 的梯度连接。
# TODO：学习 baseline 时再实现 ValueNetwork。

import torch
from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    
if __name__ == "__main__":
    policy = PolicyNetwork()

    obs = torch.zeros(8, 4)
    logits = policy(obs)

    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    print("logits:", logits.shape)
    print("actions:", actions.shape, actions.dtype)
    print("log_probs:", log_probs.shape)
    print("probability sums:", dist.probs.sum(dim=-1))
    print("log_probs requires_grad:", log_probs.requires_grad)