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

    def get_distribution(self, observations):
        logits = self(observations)
        return torch.distributions.Categorical(logits=logits)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class GaussianPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # 根据状态，预测各动作维度的均值。
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # 每个动作维度有一个可学习的 log 标准差。
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations):
        mean = self.mean_net(observations)

        std = self.log_std.exp().expand_as(mean)

        return mean, std

    def get_distribution(self, observations):
        mean, std = self(observations)

        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            1,
        )


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

    value_net = ValueNetwork()

    obs = torch.zeros(8, 4)
    values = value_net(obs)

    print("values shape:", values.shape)
    print("values requires_grad:", values.requires_grad)

    assert values.shape == (8,)
    assert values.requires_grad
    assert torch.isfinite(values).all()

    # 检查只有一个样本的批次。
    single_batch = torch.zeros(1, 4)
    single_value = value_net(single_batch)

    print("single batch shape:", single_value.shape)
    assert single_value.shape == (1,)
