# RobotRL — 机器人强化学习学习仓库

学习主线：手写 PPO → Isaac Lab Go2 仿真训练 → Domain Randomization 与鲁棒性评估 → 策略导出和部署准备。真机测试为可选扩展；没有真机也可以完成本轮学习。

实验结果统一入口：[实验索引](experiments/README.md)。各算法独立连续编号；失败和短测试记录已压缩归档。

## 项目路线与进度

按项目内容和验收结果推进，不设置周次或完成日期。核心目标是完成一个可复现的 Go2 仿真项目；进入下一阶段不代表前面的未完成项自动通过。

| 阶段 | 项目内容 | 交付成果 |
|---|---|---|
| 0：工具链验证 | Python、Gymnasium、PyTorch 与 SB3 跑通 | 随机策略结果、CartPole 基线与保存/加载验证 |
| 1：手写强化学习算法 | REINFORCE → Actor-Critic → GAE → PPO → 连续控制 | 算法代码、正确性检查、CartPole 结果、HalfCheetah 对照 |
| 2：Go2 仿真行走 | Isaac Lab 环境、任务与控制链路、官方 PPO 训练 | 自己训练的 checkpoint、行为评估、回放视频、配置说明 |
| 3：域随机化与鲁棒性 | 单项任务修改、质量/摩擦配对实验、独立压力测试 | 配置差异、学习曲线、评估表与失败案例 |
| 4：部署准备与复现 | 策略导出、推理接口、独立闭环验证、项目整理 | 导出模型、接口文档、验证结果与复现报告 |

### 阶段 0：工具链验证
- [x] 安装 `gymnasium[mujoco]` / `mujoco` / `stable-baselines3` / `torch`（仓库内 `.venv`：torch 2.14.1+cpu / gymnasium 1.4.0 / mujoco 3.15.0 / sb3 2.9.0，清华镜像源）
- [x] 随机策略 CartPole：10 局平均 **30.3 ± 18.2**（`phase0_setup/01_random_cartpole.py`，加 `--render` 弹窗观看）
- [x] SB3 PPO 训 CartPole 满分：**500.0 ± 0.0**，40,363 步提前收敛（`phase0_setup/02_sb3_ppo_cartpole.py`，曲线 `experiments/phase0_sb3_ppo/run_001/curve.png`，加 `--watch` 观看策略）

### 阶段 1：手写强化学习算法与连续控制

已完成的基础：

- [x] REINFORCE（策略梯度定理落地）：2026-10-13 单次 seed=0 运行，182,228 步后确定性评估 20 局 **500.0 ± 0.0**；中途存在回落，跨训练 seed 稳定性待验证。[实验记录](experiments/phase1_reinforce/run_001/README.md)（原次权重未保存；后续运行已支持自动保存模型）
- [x] Monte Carlo baseline：200 次更新，最佳模型新评估种子 **475.6 ± 41.7**；保存 actor、critic 和两个优化器。[实验记录](experiments/phase1_baseline/README.md)
- [x] 固定 128 步 rollout、n-step bootstrap、冻结 return / advantage 并接入更新：25,600 步训练，最佳模型新评估种子 **482.3 ± 36.6**。[实验记录](experiments/phase1_nstep/README.md)

阶段进度（2026-10-16 核验）：`[x]` 表示已有完成证据；部分完成、进行中和待做仍用 `[ ]`，并注明剩余工作。

1. [x] **GAE**：实现反向递推；已验证 λ=0、λ=1、真正终止、时间截断与 rollout 边界；return target 与 advantage 在更新前冻结。λ=0.95 的 seed=0 / 1 已完成，见[实验记录](experiments/phase1_gae/README.md)。
2. [ ] **最小稳定性对照（未完成）**：现有 λ=0.95 两个 seed 的结果不替代完整 λ 对照；仍需 λ=0 / 0.95，各训练 seed=0 / 1，使用相同环境步预算、评估协议和其余配置。保存每条曲线；两个 seed 用于初步比较，不作可靠统计结论。
3. [x] **PPO-Clip**：冻结旧 log-prob，加 ratio、clip、多 epoch、minibatch；已验证更新前 ratio≈1、裁剪分支、梯度和固定标签，并记录 approximate KL、clip fraction、entropy 和 value loss。
4. [x] **CartPole 验收**：PPO seed=0 / 1 各训练 25,600 步，均出现连续 3 次评估平均回报 ≥475；最终 checkpoint 新种子 20 局评估分别为 **496.8 ± 10.4 / 500.0 ± 0.0**，两组最佳模型均为 **500.0 ± 0.0**。两个训练 seed 为初步复现，详见[实验记录](experiments/phase1_ppo/README.md)。
5. [x] **连续动作 PPO 核心实现**：高斯策略、联合 log-prob、动作边界处理、advantage / observation 归一化、保存与冻结评估统计量已完成；单环境采样在 reset 前保存末状态，见 [ppo_continuous.py](phase1_ppo/ppo_continuous.py)。
   - [ ] **采样扩展（未完成）**：手写实现尚未改为同步向量环境；向量环境自动 reset 的末状态处理须另行验证，子进程优化按需进行。
6. [x] **HalfCheetah 初步对照**：手写 PPO 与 SB3 均完成 seed=0、1、每 seed 1,024,000 步训练，保留学习曲线、独立评估和回放；另完成 SB3 构造器默认配置对照（不等于 RL Zoo 调优配置）。见[实验索引](experiments/README.md)、[手写 PPO 比较](experiments/phase1_ppo_continuous/comparison_001.json)和 [SB3 默认配置比较](experiments/phase1_sb3_official/comparison_001.json)。配置及预处理差异随实验记录保留，两个 seed 只支持初步比较；80% 仍为挑战目标，不作为已完成项的依据。

本阶段验收：手写 PPO 的更新、终止处理和归一化有检查证据；CartPole 与 HalfCheetah 的训练、保存、加载和独立评估可复现。λ 配对实验和手写向量采样仍保留待办。

### 阶段 2：Go2 仿真行走

目标：能从环境配置解释机器人如何接收动作、获得奖励和结束回合，并独立完成训练与回放。

1. [x] **远程 Go2 环境链路**：固定 kitless 镜像、Newton/MuJoCo Warp 后端，完成最小场景、短训练、保存、加载和轨迹回放；见[环境记录](phase2_isaaclab/reports/remote_validation.json)与[代码说明](phase2_isaaclab/README.md)。
   - [ ] **原计划 Cartpole 练习**：未见该远程环境中完整训练/回放证据；当前链路验证实际使用 Go2，不将其记为 Cartpole 已完成。
2. [ ] **部分完成：任务配置梳理**。已审计随机化、奖励和终止配置；完整的逐项观测来源、坐标系和单位说明仍待整理。原要求：阅读一个 manager-based 任务：梳理 scene、observations、actions、commands、rewards、terminations、events。记录每个观测量的来源、坐标系和单位。
3. [x] **官方 Go2 PPO 训练**：当前固定镜像实际任务 ID 为 `Isaac-Velocity-Flat-UnitreeGo2`，使用配套 RSL-RL PPO；短训练通过后完成 4096 环境、300 轮、seed=0/1/42 三组训练。见[基线报告](experiments/phase2_go2_baseline/README.md)。
4. [ ] **部分完成：控制链路**。实际配置和关节映射已保留，完整接口说明尚未整理；仍需跟踪“策略输出 → action scale / offset → 关节目标 → 执行器 / PD → 物理步”的实际配置；记录 physics dt、decimation、控制频率和关节顺序。
5. [ ] **部分完成：Go2 行为验收**。自己的 checkpoint、独立随机指令/固定前进评估、20 秒前进视频与失败回放已完成；转向和停止的专项回放及指标仍待补齐。现有随机指令评估不能替代这两项专项验收。

验收：从保存配置重建环境并加载自己的模型；说明至少一个奖励项、一个终止条件及动作映射。预训练模型可用于排查环境，但不替代自己的训练结果。任务与后端以安装版本的[官方环境列表](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)为准。

### 阶段 3：任务修改、域随机化与鲁棒性

目标：在同一 Go2 任务上做可解释的对照，理解训练环境与测试环境不同会发生什么。

当前执行：[质量与摩擦最小 DR 对照协议](experiments/phase2_go2_mass_friction/README.md)。延迟测试与延迟随机化训练归入可选扩展，现有结果保留，暂停追加训练。已有官方基线不能直接标为“无 DR”；正式配对训练需另建固定物理参数组。

1. [x] **配置审计与分组**：已明确官方质量、质心、推扰、观测噪声与无地形 curriculum 的状态；新建固定/随机质量摩擦两组，两组关闭质心随机化和随机推扰，保留相同初始化与噪声。短训练已核验实际质量与接触摩擦。
2. [ ] **待做：单项任务修改练习**。先修改一个训练速度指令范围或奖励权重，观察行为与分项奖励；保存成功及失败视频，检查滑步、摔倒等现象。
3. [ ] **进行中：最小 DR 正式对照**。2026-10-16 核验时 run_001 已完成训练和五条件评估，run_002 正在评估，后两组排队；四组全部完成并核验前不勾选。最小 DR 实验只改质量与摩擦：质量可从标称值的 0.8–1.2 倍、摩擦系数从 0.5–1.5 起步，先核对模型与接触参数是否适用。每组 2 个训练 seed，匹配总环境交互步数和其余训练设置。
4. [x] **评估协议已固定**：标称、两组范围内组合、一组范围外组合及固定扰动，共五条件；相同前进指令、评估 seed=30000、每条件 32 局。第 5 秒机身横向速度增量 +0.5 m/s 的接口已验证；这是速度冲击，不是已标定推力。协议完成不代表所有正式评估已结束。
5. [ ] **待正式对照完成后汇总**：分别报告速度跟踪误差、跌倒率、回合长度和回报；按训练 seed 展示结果，不将一个模型的 20 局当成 20 个独立训练 seed。解释 DR 改善或损害了哪些条件。

验收：无 DR / 有 DR 的配置差异、曲线、评估表和视频齐全。是否改善由结果决定；观察到退化也要保留。此阶段验证的是仿真中的鲁棒性，真机迁移效果仍待测试。

### 阶段 4：部署准备与项目复现

核心任务：

1. [ ] **待做：策略导出验收**。按所用训练后端支持的格式导出策略，连同 observation normalization 等推理必需信息保存。
2. [ ] **待做：接口文档**。编写推理接口说明：观测顺序、单位、坐标系、历史窗口、动作顺序、缩放、关节限位、控制频率和 reset 状态。
3. [ ] **待做：导出一致性与闭环验证**。对一批相同输入检查导出前后动作一致性；随后在独立推理循环中回放同一仿真任务，确认数值一致之外的闭环行为也正常。
4. [ ] **部分完成：项目整理**。代码目录、实验索引和部分失败记录已整理；最终报告与导出链路尚未完成。继续整理环境安装、训练、评估、导出、回放的复现步骤；汇总失败案例、算力成本和后续待验证事项，形成最终报告。

选做（核心验收完成后，按项目需要选择）：

- **特权学习 mini 实验**：先区分“critic 有特权观测、actor 无特权观测”和“teacher actor 有特权观测”。若做蒸馏，明确 teacher / student 各自输入，先监督拟合 teacher 动作，再按需要做 DAgger；与普通 PPO 在相同测试条件下比较，不预设 teacher、普通 PPO、student 的性能排序。
- **sim-to-sim**：将同一机器人策略转到另一个物理引擎，核对模型、观测与执行器映射。只训练另一个引擎中的不同机器人不算策略迁移。

验收：形成“训练 → 保存 → 加载 → 独立评估 → 导出 → 推理回放”的完整链路。没有真机时，以此作为本轮完成条件；项目结论写“完成仿真验证与部署准备”。

### 可选扩展与优先级

- 后续以 Isaac Lab Go2 实操为主，暂不安排额外 MuJoCo 任务。
- 粗糙地形、观测延迟、执行器随机化、更多训练 seed、特权蒸馏和 sim-to-sim 后续逐项增加。
- 真机测试独立安排：取得硬件与测试条件后再完成接口适配、动作限幅、急停和低风险试运行；本轮不依赖机器人实物。
- 优先完成算法检查、一个 Go2 任务、最小 DR 对照以及导出 / 推理验收，再按需要增加粗糙地形、额外 DR 维度、特权学习、sim-to-sim 或性能优化。扩展实验不替代主线验收。

## 目录结构

| 目录 | 用途 |
|---|---|
| `notes/` | 理论笔记（策略梯度定理、GAE、特权学习、奖励设计、sim-to-real 总览）|
| `phase0_setup/` | 阶段 0 工具链验证脚本 |
| `phase1_ppo/` | 阶段 1 手写算法代码 |
| [`phase2_isaaclab/`](phase2_isaaclab/README.md) | Go2 训练、独立评估、鲁棒性测试和视频回放 |
| `experiments/` | 实验记录：config + 曲线 + 结论 |

Isaac Lab 代码统一维护在 `phase2_isaaclab/`，远程任务入口与配置分别位于 `jobs/`、`configs/`；Isaac Lab 本体和独立 Python 环境不纳入本仓库。实验按类别使用 `run_001`、`run_002` 顺序编号。

## 约定

- 每个实验保留 config、训练曲线与对应 commit，形成审计轨迹
- 调奖励/看结果必须看渲染视频，不能只看曲线（防 reward hacking 自欺）
- 参考：OpenAI Spinning Up、《动手学强化学习》、*The 37 Implementation Details of PPO*
