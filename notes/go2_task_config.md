# Go2 任务配置：我们实际训练了什么

本文对应官方基线 `phase2_go2_baseline/run_001`（seed=42），任务为 `Isaac-Velocity-Flat-UnitreeGo2`，后端为 Newton/MuJoCo Warp。它不是旧版 PhysX 教程的配置，也不是后续质量/摩擦 DR 实验。

证据：[实际 env.yaml](go2_reference/env.yaml)、[实际 agent.yaml](go2_reference/agent.yaml)、[原始配置哈希](go2_reference/manifest.json)、[官方 Go2 flat 配置源码快照](../experiments/phase2_go2_baseline/run_001/official_go2_source/flat_env_cfg.py)、[rough 父配置](../experiments/phase2_go2_baseline/run_001/official_go2_source/rough_env_cfg.py)。阅读时以解析后的 YAML 为最终参数依据，不能只看继承链中的某一层。

## 1. 整个任务如何连接

```text
scene 建立机器人、地面、传感器
  → events 设置/随机化初态和物理参数
  → commands 生成希望机器人达到的速度
  → observations 将状态和指令拼成策略输入
  → policy 输出 12 个动作
  → actions 转换成关节位置目标
  → 执行器与物理引擎推进状态
  → rewards 评分，terminations 决定是否 reset
```

`curriculum` 用于随训练调整任务难度；本平地配置的 `terrain_levels` 为 null，没有地形等级课程。

## 2. scene：环境里有什么

| 项目 | 实际设置 | 意义 |
|---|---|---|
| 并行环境 | 4096 | 同一策略同时采集多个独立环境的经验 |
| 地面 | 平地；地面静/动态摩擦配置均为 1.0 | 实际接触合并规则见第 7 节 |
| 机器人 | Go2 USD；初始根节点高度 0.4 m | 初始悬空落地过程会影响早期失败 |
| 自碰撞 | 关闭 | 不等同于真实机器人的自碰撞行为 |
| height_scanner | null | 本策略不接收地形高度扫描 |
| contact_forces | 机器人各刚体接触传感器 | 用于脚离地奖励及 base 接触终止 |
| 接触更新 | 0.005 s；历史长度 3；记录 air time | 接触历史不是策略观测历史 |
| 单回合上限 | 20 s，即 1000 个策略步 | 到时触发 time_out |

`scene` 描述对象，`robot.init_state` 描述默认状态；reset 事件会在默认状态之上加入扰动，所以默认姿态不等于每局真实初态。

## 3. observations：48 维输入逐项解释

下表是 `observations.policy` 的拼接顺序，索引从 0 开始。左右/前后关节顺序见[控制链路](go2_control_pipeline.md#2-12-维动作的关节顺序)。

| 索引 | 配置项 / 函数 | 维度 | 含义与来源 | 坐标系 / 单位 | 加性均匀噪声 |
|---|---|---:|---|---|---|
| 0–2 | `base_lin_vel` | 3 | 根节点线速度，`root_lin_vel_b` | 机身系，m/s | ±0.1 |
| 3–5 | `base_ang_vel` | 3 | 根节点角速度，`root_ang_vel_b` | 机身系，rad/s | ±0.2 |
| 6–8 | `projected_gravity` | 3 | 世界重力方向变换到机身系，反映倾斜 | 机身系，单位方向向量 | ±0.05 |
| 9–11 | `velocity_commands` / `generated_commands` | 3 | 指令 `(vx, vy, yaw_rate)` | 平移为机身系 m/s，偏航角速度 rad/s | 无 |
| 12–23 | `joint_pos` / `joint_pos_rel` | 12 | 当前关节角减去默认关节角 | 各关节局部轴，rad | ±0.01 |
| 24–35 | `joint_vel` / `joint_vel_rel` | 12 | 当前关节速度减去默认关节速度（默认 0） | 各关节局部轴，rad/s | ±1.5 |
| 36–47 | `actions` / `last_action` | 12 | action manager 接收的上一动作 | 无量纲，尚未乘 0.25 | 无 |

合计 `3+3+3+3+12+12+12=48`。`enable_corruption=true`，上述噪声在观测上加入；每项 `scale`、`clip` 均为 null，没有额外逐项缩放或截断。无历史堆叠，无高度扫描项。

actor 和 critic 的 `obs_groups` 都指向 `policy`，本基线没有另设特权 critic 观测。两者 `obs_normalization=false`，不要把手写 PPO 的 observation normalization 自动套用到这里。机身线速度来自仿真状态；真机部署时必须明确估计来源。

源码定位：YAML 每个 `func` 字段给出函数入口，例如 `isaaclab.envs.mdp.observations:joint_pos_rel`；数据属性尾缀 `_w` 表示世界系、`_b` 表示机身系。当前记录四元数顺序是 XYZW，不能沿用其他版本的 WXYZ 假设。

## 4. commands：机器人被要求做什么

`commands.base_velocity` 使用 `UniformVelocityCommand`，每 10 s 重采样：

- `lin_vel_x`、`lin_vel_y`：各为 −1 到 1 m/s。
- `ang_vel_z`：−1 到 1 rad/s。
- `heading`：−π 到 π；`heading_command=true`，`rel_heading_envs=1.0`，heading 控制增益 0.5。
- `rel_standing_envs=0.02`：约 2% 环境采样站立指令。

heading 模式根据朝向误差形成偏航速度指令，并受范围限制；不是同时要求策略跟踪独立随机 yaw rate 和独立 heading。策略实际收到的是 command manager 输出的三维速度。

固定前进评估会关闭 heading 模式及 standing 采样，把指令固定为 `(0.5,0,0)`，并延长重采样间隔。这是评估覆盖，不是官方训练指令范围。

## 5. actions：策略控制什么

使用 `JointPositionAction`，匹配全部 12 个关节，`scale=0.25`、`use_default_offset=true`。动作转成目标关节角：

$$q^{\mathrm{target}}_j=q^{\mathrm{default}}_j+0.25a_j.$$

策略不直接输出力矩。动作配置 `clip=null`，runner 的 `clip_actions=null`；不能假定动作被限制在 `[-1,1]`。后续执行器约束与关节限位是不同层面的机制。完整例子见[控制链路笔记](go2_control_pipeline.md)。

## 6. rewards 和 terminations

实际启用的奖励项如下。权重来自 `env.yaml/rewards`；标准差参数为 `std=0.5` 的指数跟踪奖励用于鼓励接近目标速度。

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `track_lin_vel_xy_exp` | 1.5 | 鼓励平面速度跟踪 |
| `track_ang_vel_z_exp` | 0.75 | 鼓励偏航角速度跟踪 |
| `lin_vel_z_l2` | −2.0 | 惩罚机身上下运动 |
| `ang_vel_xy_l2` | −0.05 | 惩罚滚转/俯仰角速度 |
| `dof_torques_l2` | −0.0002 | 惩罚关节力矩平方和 |
| `dof_acc_l2` | −2.5e−7 | 惩罚关节加速度平方和 |
| `action_rate_l2` | −0.01 | 惩罚相邻动作变化 |
| `feet_air_time` | 0.25 | 利用脚接触/离地时间鼓励步态，threshold=0.5 s |
| `flat_orientation_l2` | −2.5 | 鼓励机身保持水平 |
| `dof_pos_limits` | 0.0 | 当前无奖励贡献 |
| `undesired_contacts` | null | 本项关闭 |

不能仅根据权重判断哪项贡献最大，还要查看项值和框架的时间步加权。脚离地奖励也不是“脚离地越久越好”；具体接触切换及指令门控由对应 reward 函数决定。

终止分两类：`time_out` 是达到 20 s 的时间截断；`base_contact` 使用 `illegal_contact`，检查 base 接触，threshold=1.0 N，是非超时终止。它是跌倒的代理指标，不覆盖所有异常姿态。关节限位、脚滑动或姿态不好不一定立即触发终止。

## 7. events：官方基线已经包含哪些随机化

| 事件 | 触发 | 实际配置 |
|---|---|---|
| `physics_material` | startup | 机器人静摩擦 0.8、动态摩擦 0.6、恢复系数 0；范围退化为常数 |
| `add_base_mass` | startup | 仅 base 质量乘以 log-uniform(0.8,1.25)；并相应缩放惯量 |
| `base_com` | startup | COM 偏移 x/y：±0.05 m，z：±0.01 m |
| `base_external_force_torque` | reset | 力和力矩范围均为 0，当前不产生有效扰动 |
| `reset_base` | reset | 平面位置 ±0.5 m，yaw 约 ±π；各线/角速度分量 ±0.5 |
| `reset_robot_joints` | reset | 默认关节角乘以 uniform(0.5,1.5)，速度为 0 |
| `push_robot` | interval | 每 10–15 s 加入世界系 x/y 速度扰动，各 ±0.5 m/s |

startup 表示建立环境时采样，不能写成“每局都随机”。reset 发生在新回合；interval 使用事件计时器。

特别注意当前后端的摩擦：Newton 材料事件使用单个滑动摩擦参数；现有审计发现等优先级 MuJoCo 接触取两侧摩擦较大值。官方 ground=1.0、robot=0.8 的活动接触实际为 1.0，而不是从 `friction_combine_mode=multiply` 推算出 0.8。只降低机器人摩擦到 0.3，仍会被地面 1.0 覆盖。

证据与工具：[低摩擦修正结果](../experiments/phase2_go2_robustness/run_003/README.md)、[活动接触审计代码](../phase2_isaaclab/friction_audit.py)、[镜像运行时事件源码](../experiments/phase2_go2_robustness/run_003/runtime_material_events.py)。因此官方基线不能直接命名为“无 DR”。

## 8. 与当前质量/摩擦对照的关系

当前[主线对照协议](../experiments/phase2_go2_mass_friction/README.md)单独建立 fixed / randomized 两组，关闭 COM 随机化及随机推扰，保留相同初态与观测噪声；只在 base 质量与机器人摩擦范围上做组间变化。地面统一 0.5，以保证机器人 0.5–1.5 的摩擦变化能进入接触计算。

阅读结果时应先确认自己看的是“官方基线”“固定参数组”“随机参数组”还是“固定条件评估”；它们不是同一份环境配置。
