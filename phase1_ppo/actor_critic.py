"""Actor-critic：从学习到的 baseline 过渡到 n-step 更新与 GAE。"""

# TODO：先使用完整轨迹回报训练 critic，并构造固定的 advantage。
# TODO：加入固定长度 rollout，区分终止、截断和 rollout 尾部。
# TODO：实现 GAE 与 return target，分别更新独立的 actor 和 critic。
