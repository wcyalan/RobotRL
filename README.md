# RobotRL — 机器人强化学习学习仓库

从零到 sim-to-real 的学习主线：手写 PPO → 训练双足机器人行走（带 Domain Randomization）→ GPU 并行仿真与特权学习。

## 学习计划与进度

### 阶段 0：工具链跑通（第 1 周）
- [ ] 安装 `gymnasium[mujoco]` / `mujoco` / `stable-baselines3` / `torch`
- [ ] 随机策略渲染 CartPole
- [ ] SB3 PPO 训 CartPole 到满分（作为后续手写实现的对照组）

### 阶段 1：手写 REINFORCE → PPO（第 2–4 周，重心）
- [ ] REINFORCE（策略梯度定理落地），CartPole 收敛
- [ ] 加 baseline（V 网络）→ advantage actor-critic
- [ ] 加 GAE(λ)，做 λ=0 / 0.95 训练曲线对照
- [ ] 加 ratio clipping + 多 epoch minibatch → PPO
- [ ] **验收**：HalfCheetah 1M 步得分 ≥ SB3 同预算的 80%

### 阶段 2：Walker2d 主线（第 5–7 周）
- [ ] 向量化环境（SubprocVecEnv，一次收集 N×2048 步）
- [ ] 读懂 MJCF：改 Walker2d 质量、加 sensor
- [ ] Domain Randomization：torso 质量 ±20%、摩擦 [0.5, 1.5] vs 确定性训练
- [ ] **里程碑**：DR 策略行走视频 + 两组策略跨参数 robustness 对比图

### 阶段 3：GPU 大规模并行仿真（第 8–10 周）
- [ ] Isaac Lab（服务器）或 MuJoCo Playground（MJX，本地可看）
- [ ] 跑通 Go2/ANYmal 平地行走 tutorial
- [ ] 特权学习 mini 实验：teacher 特权 obs 训练 → student 蒸馏

### 阶段 4（可选分叉）
- 操作方向：ManiSkill 3 / robosuite
- 视觉方向：obs 换渲染图像 + CNN encoder
- 真机方向：LeRobot SO-100

## 目录结构

| 目录 | 用途 |
|---|---|
| `notes/` | 理论笔记（策略梯度定理、GAE、特权学习、奖励设计、sim-to-real 总览）|
| `phase1_ppo/` | 阶段 1 手写算法代码 |
| `phase2_walker/` | 阶段 2 Walker2d + DR 实验代码 |
| `experiments/` | 实验记录：config + 曲线 + 结论 |

## 约定

- 每个实验保留 config、训练曲线与对应 commit，形成审计轨迹
- 调奖励/看结果必须看渲染视频，不能只看曲线（防 reward hacking 自欺）
- 参考：OpenAI Spinning Up、《动手学强化学习》、*The 37 Implementation Details of PPO*
