"""PPO-Clip：在 actor-critic 与 GAE 的基础上加入近端策略更新。"""

# TODO：采样时保存 old log-prob，更新前验证 ratio 约等于 1。
# TODO：实现 clipped surrogate 与多 epoch minibatch 更新。
# TODO：记录 KL、clip fraction、entropy 和评估回报。
