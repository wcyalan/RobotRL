r"""检查 rollout 采集，不训练网络、不修改已有模型。

运行：.\.venv\Scripts\python.exe phase1_ppo/check_rollout.py
"""

import gymnasium as gym
import numpy as np
import torch

from actor_critic import (
    collect_rollout, compute_bootstrapped_returns, prepare_rollout_targets,
    update_actor, update_critic,
)
from networks import PolicyNetwork, ValueNetwork


class AuditedEnv(gym.Wrapper):
    """独立记录环境真实返回值，核对 collector 是否正确保存。"""

    def __init__(self, env):
        super().__init__(env)
        self.transitions = []
        self.current_obs = None
        self.reset_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs.copy()
        self.reset_count += 1
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), "动作不是合法的环境动作"
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.transitions.append({
            "observations": self.current_obs.copy(),
            "actions": action,
            "rewards": reward,
            "next_observations": next_obs.copy(),
            "terminated": terminated,
            "truncated": truncated,
        })
        self.current_obs = next_obs.copy()
        return next_obs, reward, terminated, truncated, info


def check_batch(env, policy, obs, n_steps):
    start_obs = obs.copy()
    first_transition = len(env.transitions)
    initial_resets = env.reset_count
    rollout, obs = collect_rollout(env, policy, obs, n_steps)
    recorded = env.transitions[first_transition:]
    assert len(recorded) == n_steps, "实际环境交互次数不正确"

    for key in ("observations", "next_observations"):
        assert rollout[key].shape == (n_steps, 4), f"{key} 形状错误"
        assert rollout[key].dtype == np.float32
        assert np.isfinite(rollout[key]).all()
    for key in ("actions", "rewards", "terminated", "truncated"):
        assert rollout[key].shape == (n_steps,), f"{key} 形状错误"
    assert rollout["actions"].dtype == np.int64
    assert rollout["rewards"].dtype == np.float32
    assert np.isfinite(rollout["rewards"]).all()
    assert rollout["terminated"].dtype == np.bool_
    assert rollout["truncated"].dtype == np.bool_

    # 包括 episode 边界：必须保存 step 返回的末状态，而非 reset 的新状态。
    for key in recorded[0]:
        expected = np.asarray([item[key] for item in recorded])
        np.testing.assert_allclose(rollout[key], expected, err_msg=key)

    np.testing.assert_allclose(rollout["observations"][0], start_obs)
    np.testing.assert_allclose(obs, env.current_obs)
    boundaries = rollout["terminated"] | rollout["truncated"]
    assert env.reset_count - initial_resets == int(boundaries.sum()), "多余或遗漏 reset"

    for t in range(n_steps - 1):
        if not boundaries[t]:
            np.testing.assert_allclose(
                rollout["next_observations"][t], rollout["observations"][t + 1]
            )
    if not boundaries[-1]:
        np.testing.assert_allclose(obs, rollout["next_observations"][-1])

    terminations = int(rollout["terminated"].sum())
    truncations = int(rollout["truncated"].sum())
    print(f"PASS: steps={n_steps}, terminated={terminations}, truncated={truncations}")
    return obs, terminations, truncations


def check_targets_and_updates():
    cases = [
        ([1, 1], [123, 10], [False, False], [False, False], [10, 10]),
        ([1, 1], [123, 10], [False, True], [False, False], [1.9, 1]),
        ([1, 100], [5, 999], [False, True], [True, False], [5.5, 100]),
        ([1, 100], [5, 999], [True, True], [False, False], [1, 100]),
        ([2], [99], [True], [True], [2]),
    ]
    for rewards, values, term, trunc, expected in cases:
        actual = compute_bootstrapped_returns(rewards, values, term, trunc, 0.9)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    torch.manual_seed(0)
    actor, critic = PolicyNetwork(), ValueNetwork()
    env = gym.make("CartPole-v1", max_episode_steps=3)
    try:
        obs, _ = env.reset(seed=0)
        rollout, _ = collect_rollout(env, actor, obs, 16)
    finally:
        env.close()
    observations, actions, targets, advantages = prepare_rollout_targets(critic, rollout, 0.99)
    assert targets.shape == advantages.shape == (16,)
    assert not targets.requires_grad and not advantages.requires_grad
    with torch.no_grad():
        torch.testing.assert_close(targets, advantages + critic(observations))
        dist = torch.distributions.Categorical(logits=actor(observations))
        expected_loss = -(dist.log_prob(actions) * advantages).mean().item()

    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]
    target_before, advantage_before = targets.clone(), advantages.clone()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    loss = update_actor(actor, actor_optimizer, observations, actions, advantages)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    assert any(not torch.equal(p, old) for p, old in zip(actor.parameters(), actor_before))
    assert all(p.grad is None for p in critic.parameters())
    assert all(torch.equal(p, old) for p, old in zip(critic.parameters(), critic_before))

    actor_after = [p.detach().clone() for p in actor.parameters()]
    actor_optimizer.zero_grad(set_to_none=True)
    critic_loss = update_critic(critic, critic_optimizer, observations, targets)
    assert np.isfinite(critic_loss)
    assert any(not torch.equal(p, old) for p, old in zip(critic.parameters(), critic_before))
    assert all(p.grad is None for p in actor.parameters())
    assert all(torch.equal(p, old) for p, old in zip(actor.parameters(), actor_after))
    assert torch.equal(targets, target_before) and torch.equal(advantages, advantage_before)
    print("PASS: bootstrap boundaries, frozen targets, sample-mean loss, isolated updates")


if __name__ == "__main__":
    check_targets_and_updates()
    torch.manual_seed(0)
    policy = PolicyNetwork()
    before = {key: value.clone() for key, value in policy.state_dict().items()}

    print("Normal CartPole: two consecutive 128-step rollouts")
    env = AuditedEnv(gym.make("CartPole-v1"))
    try:
        obs, _ = env.reset(seed=0)
        terminal_count = 0
        for _ in range(2):
            obs, terminations, _ = check_batch(env, policy, obs, 128)
            terminal_count += terminations
        assert terminal_count > 0, "本次没有覆盖真正终止的路径"
    finally:
        env.close()

    print("Time-limit test: force truncation after each step")
    env = AuditedEnv(gym.make("CartPole-v1", max_episode_steps=1))
    try:
        obs, _ = env.reset(seed=0)
        for n_steps in (4, 1):
            obs, _, truncations = check_batch(env, policy, obs, n_steps)
            assert truncations == n_steps, "没有覆盖预期的截断路径"
    finally:
        env.close()

    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key]), "采样期间网络参数发生变化"
    assert all(parameter.grad is None for parameter in policy.parameters())
    print("All checks passed: shapes, transitions, resets, continuation and unchanged policy.")
