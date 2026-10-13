# RobotRL — 机器人强化学习学习仓库

从零到 sim-to-real 的学习主线：手写 PPO → 训练双足机器人行走（带 Domain Randomization）→ 特权学习 mini 实验。

## 学习计划与进度（2026-10-13 周一开工，11-07 前完成）

### 阶段 0：工具链跑通 — 2026-10-08 完成
- [x] 安装 `gymnasium[mujoco]` / `mujoco` / `stable-baselines3` / `torch`（仓库内 `.venv`：torch 2.14.1+cpu / gymnasium 1.4.0 / mujoco 3.15.0 / sb3 2.9.0，清华镜像源）
- [x] 随机策略 CartPole：10 局平均 **30.3 ± 18.2**（`phase0_setup/01_random_cartpole.py`，加 `--render` 弹窗观看）
- [x] SB3 PPO 训 CartPole 满分：**500.0 ± 0.0**，40,363 步提前收敛（`phase0_setup/02_sb3_ppo_cartpole.py`，曲线 `experiments/phase0_sb3_ppo/curve.png`，加 `--watch` 观看策略）

### 第 1 周（10.13–10.19）：手写 REINFORCE → A2C + GAE
- [x] REINFORCE（策略梯度定理落地）：2026-10-13 单次 seed=0 运行，182,228 步后确定性评估 20 局 **500.0 ± 0.0**；中途存在回落，跨训练 seed 稳定性待验证。[实验记录](experiments/phase1_reinforce/20261013_original_log/README.md)（原次权重未保存；后续运行已支持自动保存模型）
- [ ] 加 baseline（V 网络）→ advantage actor-critic，对比训练曲线方差
  - 2026-10-13：已完成 Monte Carlo baseline 过渡版及 200 次更新；最佳模型新种子评估 **475.6 ± 41.7**，actor/critic 与两个优化器均已保存。[实验记录](experiments/phase1_baseline/README.md)。n-step bootstrap 与多 seed 方差比较待完成。
- [ ] 加 GAE(λ)：λ=0 / 0.95 各 2 seed 对照即可
- [ ] **里程碑**：自己的 A2C 在 CartPole 到 500

### 第 2 周（10.20–10.26）：完整 PPO + HalfCheetah 验收
- [ ] ratio clipping + 多 epoch minibatch → PPO
- [ ] 向量化采样（DummyVecEnv 起步 → SubprocVecEnv）
- [ ] obs 归一化 + advantage 归一化，对照 *37 Implementation Details* 清单查漏
- [ ] **验收**：HalfCheetah 1M 步 ≥ SB3 同预算的 80%（达不到先 diff SB3 源码再调）

### 第 3 周（10.27–11.02）：Walker2d + Domain Randomization
- [ ] 用自己的 PPO 训 Walker2d 到 3000+
- [ ] MJCF 最小改动练习：改 torso 质量、加一个 sensor
- [ ] DR 实验：质量 ±20% / 摩擦 [0.5, 1.5] vs 确定性训练，各 2 seed，在 3 组未见参数上评估
- [ ] **里程碑**：robustness 对比图 + DR 策略行走视频

### 第 4 周（11.03–11.07，短周）：特权学习 mini 实验（纯 MuJoCo，不装 Isaac Lab）
- [ ] teacher：obs 拼接特权信息（真实质量/摩擦）训练
- [ ] student：仅普通 obs，DAgger 式蒸馏 teacher
- [ ] **里程碑**：跨参数评估满足 teacher ≥ 普通 PPO ≥ student 的排序
- [ ] 选读半天：Isaac Lab Go2 tutorial / MuJoCo Playground 浏览器 demo（认知性了解）

### 相对十周版砍掉的内容（防自欺清单）
- 阶段 4 全部分叉（操作 / 视觉 / 真机）→ 后续再加
- Isaac Lab 动手实操 → 特权学习实验改在 MuJoCo 里做（概念不损失，省去服务器环境搭建）
- 多 seed 统计、λ 详尽扫描 → 各保留最小规模（2 seed / 2 配置）

## 目录结构

| 目录 | 用途 |
|---|---|
| `notes/` | 理论笔记（策略梯度定理、GAE、特权学习、奖励设计、sim-to-real 总览）|
| `phase0_setup/` | 阶段 0 工具链验证脚本 |
| `phase1_ppo/` | 阶段 1 手写算法代码 |
| `phase2_walker/` | 阶段 2 Walker2d + DR 实验代码 |
| `experiments/` | 实验记录：config + 曲线 + 结论 |

## 约定

- 每个实验保留 config、训练曲线与对应 commit，形成审计轨迹
- 调奖励/看结果必须看渲染视频，不能只看曲线（防 reward hacking 自欺）
- 参考：OpenAI Spinning Up、《动手学强化学习》、*The 37 Implementation Details of PPO*
