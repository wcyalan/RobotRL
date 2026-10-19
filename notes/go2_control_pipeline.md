# Go2：策略输出与控制链路

本文以[官方基线实际环境配置](go2_reference/env.yaml)和 [PPO 配置](go2_reference/agent.yaml)为依据。任务整体说明见[Go2 任务配置](go2_task_config.md)。数值例子用于理解映射，不是一次实测力矩记录。

## 1. 一次策略决策发生了什么

```text
48 维状态/指令观测
  → actor MLP（128 → 128 → 128，ELU）
  → 12 维动作 a
  → q_target = q_default + 0.25 × a
  → 关节目标交给执行器/物理后端
  → 4 次环境物理步（目标保持不变）
  → 接触与关节运动改变机器人状态
  → 计算奖励、终止和下一次观测
```

训练时从高斯策略采样动作，独立评估使用确定性策略输出。actor 和 critic 都读同一 `policy` 观测组；critic 输出价值，不负责生成电机控制命令。

动作单位是无量纲。`0.25` 把动作转成弧度偏移；它不是速度倍率，也不是力矩倍率。零动作意味着回到默认关节目标，不意味着电机不出力。

## 2. 12 维动作的关节顺序

以下顺序来自[历史运行时控制诊断](go2_reference/control_diagnostic.json)中的 `joint_names`，不是根据名称排序猜测。FL/FR/RL/RR 分别为左前、右前、左后、右后。

| 索引 | 关节 | 默认目标 rad |
|---:|---|---:|
| 0 | FL_hip_joint | 0.1 |
| 1 | FL_thigh_joint | 0.8 |
| 2 | FL_calf_joint | −1.5 |
| 3 | FR_hip_joint | −0.1 |
| 4 | FR_thigh_joint | 0.8 |
| 5 | FR_calf_joint | −1.5 |
| 6 | RL_hip_joint | 0.1 |
| 7 | RL_thigh_joint | 1.0 |
| 8 | RL_calf_joint | −1.5 |
| 9 | RR_hip_joint | −0.1 |
| 10 | RR_thigh_joint | 1.0 |
| 11 | RR_calf_joint | −1.5 |

动作、关节角观测和关节速度观测需要使用一致映射。此顺序是已测资产/后端的结果；更换资产或后端后应重新读取 `robot.joint_names` 与 action term 的解析关节 ID。

历史诊断为了检查控制和 base 接触，调整了接触容量、回合长度并关闭 base-contact 终止；它可以证明当时的顺序和动作映射，不能用来代替官方训练结果。原件位置和 SHA-256 见[证据清单](go2_reference/manifest.json)。

## 3. 动作如何变成位置目标

`JointPositionAction` 的 `use_default_offset=true` 会使用默认关节位置作为偏移，不能仅看到 YAML 中 `offset: 0.0` 就认为目标是 `0.25a`。

$$q^{\mathrm{target}}_j=q^{\mathrm{default}}_j+0.25a_j.$$

例如只给四个 calf 关节动作 0.4，其他关节动作 0，则 calf 目标变为：

$$-1.5+0.25\times0.4=-1.4\;\mathrm{rad}.$$

它要求关节角增加 0.1 rad，并不是立即把真实关节状态改为 −1.4。关节要通过执行器施力、惯性和接触约束逐渐运动。历史诊断逐步验证了 `processed_actions == default_joint_pos + 0.25 * action`。

`JointPositionAction.clip=null`、runner `clip_actions=null`，高斯输出没有配置成固定 `[-1,1]` 截断。`soft_joint_pos_limit_factor=0.9` 是软限位相关配置，不等于将每个位置目标自动夹在该范围；本任务 `dof_pos_limits` 奖励权重也为 0。

## 4. PD 与电机约束应如何理解

配置中的执行器类型是 `DCMotor`，所有腿关节 stiffness=25.0、damping=0.5、armature=0.02、friction=0.0。理想位置 PD 可以写为：

$$\tau_{\mathrm{PD}}=K_p(q^{\mathrm{target}}-q)+K_d(\dot q^{\mathrm{target}}-\dot q).$$

单纯位置跟踪时可用零目标速度理解阻尼项。比如某 calf 的目标为 −1.4 rad、当前角度 −1.5 rad、当前角速度 +0.2 rad/s，则理想 PD 输出为：

$$25\times0.1+0.5\times(0-0.2)=2.4\;\mathrm{N\,m}.$$

这是施加约束前的示意计算。实际状态更新还受电机模型、约束求解、接触、惯量等影响；不能据此声称日志中的实际关节力矩就是 2.4 N·m。

| 配置字段 | hip / thigh | calf |
|---|---:|---:|
| `actuator_effort_limit` | 23.7 N·m | 45.43 N·m |
| `saturation_effort` | 23.7 N·m | 45.43 N·m |
| `actuator_velocity_limit` | 30.1 rad/s | 15.7 rad/s |

这些是模型配置值，不是真机安全操作许可。`joint_effort_limit`、`joint_velocity_limit` 以及旧式 `effort_limit_sim`/`velocity_limit_sim` 字段均为 null；不要把几个不同字段混为同一个限幅器。

**本次后端差异：** `use_newton_actuators=true`，控制通过 Newton 原生执行器进入 MuJoCo Warp。运行时求解器源码把位置目标与 `joint_target_ke`、速度目标与 `joint_target_kd` 对接到相应执行器；求解器采用 `implicitfast`。因此上面的公式是 PD 控制意图的解释，不能直接断言运行了 PhysX 路线的 Python `DCMotor.compute()`，也不能未经后端核验就声称完全执行了相同的速度相关力矩裁剪。

源码证据：[运行时求解器快照](../experiments/phase2_go2_robustness/run_003/runtime_solver.py)，检索 `CtrlType`、`joint_target_q`、`joint_target_ke`、`joint_target_kd`。当前已经核验动作到位置目标；部署前仍需核对目标后端的力矩约束与实际输出，不能把这份笔记当作完整部署验收。

## 5. 三层时间尺度

| 层次 | 设置 | 含义 |
|---|---|---|
| 环境物理步 | `sim.dt=0.005 s`，200 Hz | 环境调用一次物理推进的时间间隔 |
| 策略步 | `decimation=4` | 每 4 个物理步才换一次策略动作 |
| 策略周期 | 0.005×4=0.02 s，50 Hz | 20 s 回合包含 1000 次策略决策 |
| 后端子步 | `num_substeps=2` | 每个物理步内进一步细分；按该配置为 0.0025 s，不是策略以 400 Hz 更新 |

位置目标在 4 个物理步之间保持，机器人状态和接触持续变化；“动作不变”不意味着施加力矩不变。接触传感器配置更新周期为 0.005 s，与策略周期不同。

`render_interval=4` 是渲染配置，不是额外的动作延迟。当前训练不依赖图形渲染；本地视频使用已记录姿态做运动学外观回放，不重新执行物理仿真。

## 6. 一步与一次 PPO 更新的区别

每个环境先采集 24 个策略步，即 0.48 s 仿真时间；4096 环境合计 98,304 条 transition。随后 PPO 用这批数据做 5 个 epoch、4 个 minibatch 的更新。300 轮共 29,491,200 环境步。

PPO 更新频率不是电机控制频率：训练中策略参数按 rollout 更新，而执行时每 20 ms 根据当前观测输出动作。评估只做推理，不更新参数。

## 7. 从 Go2 推广到其他控制任务时检查什么

1. 观测维度、拼接顺序、噪声和归一化是否与训练一致。
2. 关节名与动作索引、角度正方向、默认姿态是否一致。
3. 动作表示的是位置偏移、速度还是力矩；缩放和限幅在哪里执行。
4. PD/执行器由 Python、仿真后端还是外部控制器实现，是否重复施加控制。
5. 物理步、策略步和传感器更新是否一致，reset 时上一动作如何初始化。

后续人形机器人可以复用这个检查框架，但关节顺序、默认姿态、增益、限幅和任务观测必须重新核对。
