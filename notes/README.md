# 理论笔记

- [x] [RL 方法框架与策略梯度](01_policy_gradient.md) — value/policy/actor-critic 三分法、策略梯度定理、REINFORCE、softmax/高斯策略、on-policy
- [x] [λ-Return 与 GAE(λ)](02_gae.md) — 定义、在 PPO 中的角色（policy 信号 + critic target）、γ vs λ、默认值诊断
- [x] [PPO：从策略梯度到近端策略优化](03_ppo.md) — REINFORCE→自然梯度→TRPO→PPO 演进、Clip 目标与六情形、完整训练循环、实现细节清单
- [ ] 特权学习：LUPI / 非对称 actor-critic / teacher-student 蒸馏 / RMA
- [ ] 机器人奖励设计三分类（任务/塑形/正则）与 reward hacking 案例
- [ ] sim-to-real 方法总览（DR、ADR、SysID、real2sim、残差策略、world model）
- [ ] 虚拟细胞项目：机器人 RL → 细胞干预的映射表与 MDP 形式化
