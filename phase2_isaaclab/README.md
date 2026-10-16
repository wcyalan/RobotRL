# Phase 2：Go2 仿真训练与验证

使用固定 Isaac Lab kitless 镜像、Newton/MuJoCo Warp 后端和 RSL-RL PPO。训练结果在 `experiments/phase2_go2_baseline/`，鲁棒性评估在 `experiments/phase2_go2_robustness/`。本目录只维护当前代码、任务入口和环境记录。

## 代码结构

当前主线为[质量与摩擦 DR 配对实验](../experiments/phase2_go2_mass_friction/README.md)。延迟实验属于保留的可选扩展，不再追加训练。

| 文件或目录 | 用途 |
|---|---|
| `baseline_config.py` | 保存官方默认值与本次训练配置 |
| `train_mass_friction.py` | 固定/随机质量与摩擦对照，审计实际 base 质量和活动接触 |
| `evaluate_go2.py` | 独立评估、轨迹记录、质量/摩擦/延迟条件 |
| `action_delay.py` | 固定动作延迟队列与分环境重置 |
| `random_action_delay.py` | 每局随机延迟、独立随机数生成器与队列清空 |
| `train_delay_randomized.py` | 为官方训练入口添加动作延迟包装器 |
| `verify_delay_training.py` | 独立核验训练时实际执行的延迟动作 |
| `friction_audit.py` | 读取求解器材质及有效接触摩擦 |
| `compare_go2_seeds.py` | 核验三组基线配置与模型哈希，汇总比较 |
| `summarize_low_friction.py` | 核验修正后的低摩擦结果 |
| `summarize_delay.py` | 核验延迟动作队列、模型哈希并汇总延迟评估 |
| `render_go2_trajectory.py` | 用记录的关节和根节点轨迹导出 MP4 |
| `download_go2_visual.py` | 获取 MuJoCo Menagerie Go2 外观资产及许可 |
| `prepare_context.py` | 将代码、任务脚本与配置打包为远程任务所需的平铺目录 |
| `jobs/` | 远程 Bash 任务入口 |
| `configs/` | 固定镜像、GPU 资源和缓存配置 |
| `reports/` | 已验证的环境与任务状态记录 |

## 远程任务入口

| jobs 下的文件 | 用途 |
|---|---|
| `train_baseline.sh` | 单组完整官方基线；参数为远程输出目录及可选训练 seed |
| `smoke_mass_friction.sh` | 两组各 64 环境、3 轮短训练，验证后端物理参数 |
| `train_mass_friction.sh` | 先验证固定扰动，再跑两组 × 两个 seed，每个模型评估五个固定条件 |
| `train_seeds.sh` | seed=0、1 的连续补跑 |
| `train_delay_randomized.sh` | 先短测验证，再训练随机延迟策略并评估 |
| `evaluate_robustness.sh` | 初始质量/摩擦测试；保留作复现，robot-only 低摩擦项已证实无效 |
| `audit_friction.sh` | 对照、只改机器人、同时修改机器人与地面的摩擦核验 |
| `evaluate_low_friction.sh` | 修正后的三 seed 低摩擦评估与有效接触核验 |
| `evaluate_delay.sh` | 三 seed × 0/20/40 ms 动作延迟评估 |
| `replay_failure.sh` | 复测并记录 seed=1 的 env 11 失败轨迹 |

这些脚本保留对应实验的输出编号和模型路径，包含防覆盖检查。启动新实验前应指定新的输出编号；不能直接重复运行已完成的历史任务覆盖原结果。

## 准备远程任务

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe .\phase2_isaaclab\prepare_context.py --output .\.task_context\go2_001
```

输出目录必须不存在。将其文件上传到远程独立任务目录，使用其中的 `train_baseline.yaml` 和相应 `.sh` 作为 Determined 的配置和入口。`--context` 指向这份平铺目录；不要直接指向整理后的源码目录，否则 Bash 的相对文件依赖不完整。打包过程会统一 Linux 换行并生成文件哈希清单，不会自动提交任务。

## 本地评估汇总与回放

```powershell
.\.venv\Scripts\python.exe .\phase2_isaaclab\compare_go2_seeds.py
.\.venv\Scripts\python.exe .\phase2_isaaclab\summarize_low_friction.py
.\.venv\Scripts\python.exe .\phase2_isaaclab\summarize_delay_training.py
.\.venv\Scripts\python.exe .\phase2_isaaclab\render_go2_trajectory.py --run run_002
```

视频是记录轨迹的运动学渲染，不是新的闭环仿真或 sim2real 验证。训练和评估脚本需在指定远程镜像中执行；本机主要做结果汇总与视频渲染。

## 历史文件

早期远程探测、冒烟训练、缩小规模训练、控制诊断和一次性整理脚本已归档到 `archives/phase2_code_cleanup_001/legacy/`。完整整理前代码备份为同目录的 `phase2_before_cleanup.zip`。实验内的源码快照保持原样，远程已提交任务不受本地目录调整影响。
