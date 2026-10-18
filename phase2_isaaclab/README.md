# Phase 2：Go2 仿真训练与验证

使用固定 Isaac Lab kitless 镜像、Newton/MuJoCo Warp 后端与 RSL-RL PPO。当前主线为[Go2 导出与独立闭环验证](../experiments/phase2_go2_deployment/run_001/README.md)：导出动作一致性通过，行为12项中10项通过，仍需处理重跑稳定性。质量/摩擦对照与延迟实验保留为历史证据，不追加训练。

学习笔记：[任务配置](../notes/go2_task_config.md) · [控制链路](../notes/go2_control_pipeline.md)。

## 先找哪个目录

| 目录 / 文件 | 用途 | 常用入口 |
|---|---|---|
| `training/` | 官方基线配置记录、质量/摩擦训练 | `train_mass_friction.py` |
| `evaluation/` | 独立评估与活动接触审计 | `evaluate_go2.py`、`friction_audit.py` |
| `analysis/` | 本地结果比较与诊断 | `compare_go2_seeds.py`、`summarize_low_friction.py` |
| `visualization/` | 轨迹视频和外观资产 | `render_go2_trajectory.py` |
| `deployment/` | 冻结策略导出、纯PyTorch接口和验收汇总 | `export_policy.py`、`inference.py`、`summarize.py` |
| `extensions/delay/` | 可选延迟实验的队列、训练和验证 | 不属于当前主线 |
| `jobs/baseline/` | 官方基线任务 | `train_baseline.sh`、`train_seeds.sh` |
| `jobs/domain_randomization/` | 当前质量/摩擦对照任务 | `smoke_mass_friction.sh`、`train_mass_friction.sh` |
| `jobs/diagnostics/` | 历史摩擦诊断与失败复现 | 按需使用 |
| `jobs/delay/` | 可选延迟实验任务 | 保留复现入口 |
| `configs/` | 固定镜像及任务配置 | `train_baseline.yaml` |
| `reports/` | 环境与目录验证记录 | 历史记录标注其适用阶段 |
| `prepare_context.py` | 按任务生成最小运行包 | 远程运行的统一准备入口 |

## 如何提交任务

当前模型与入口：`model_450`是筛选后通过新seed验证的检查点；不要替换成右转退化的最终`model_extended.pt`。推理接口见[部署说明](../notes/go2_deployment.md)。`jobs/diagnostics/`中`evaluate_checkpoints.sh`负责筛选/独立验证，`validate_export.sh`负责导出一致性/闭环评估，`resume_command_coverage.sh`保留延长训练配方。各任务只通过`prepare_context.py`打包执行，不重复维护另一套运行代码。

训练/评估源码依赖固定远程镜像，按下列方式打包后在远程运行。不要从源码子目录直接运行跨目录依赖的任务；打包器会把该任务所需脚本平铺到同一目录，并统一 Linux 换行。

```powershell
.\.venv\Scripts\python.exe .\phase2_isaaclab\prepare_context.py --job train_mass_friction --output .\.task_context\mass_friction_next
```

`--output` 必须为新目录；`--job` 为任务脚本去掉 `.sh` 的名字，所有选项可用 `--help` 查看。质量/摩擦正式任务包仅含训练入口、评估入口、摩擦审计、Bash 和 YAML，不包含本地分析、视频、延迟训练或无关历史入口。

**现有 jobs 是已完成实验的可复现配方，含固定输出编号和防覆盖检查。** 新实验必须先设置新的输出编号/评估协议再打包，不能直接重跑旧配方覆盖结果。当前正式四组已运行完成，结果审计与视频检查另行进行。

将生成的目录上传到服务器 `contexts/<任务名>/`，从该目录提交，例如：

```bash
det command run -d --config-file /labdata2/congyuwang/RobotRL_kitless/contexts/<任务名>/train_baseline.yaml --context /labdata2/congyuwang/RobotRL_kitless/contexts/<任务名> bash train_mass_friction.sh
```

行为验收任务：`prepare_context.py --job evaluate_behaviors`；数值结果由 `analysis/summarize_behaviors.py` 按预先固定的阈值评分。异常回合复测使用 `--job replay_dr_outlier`。

## 本地分析与视频

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe .\phase2_isaaclab\analysis\compare_go2_seeds.py
.\.venv\Scripts\python.exe .\phase2_isaaclab\analysis\summarize_low_friction.py
.\.venv\Scripts\python.exe .\phase2_isaaclab\visualization\render_go2_trajectory.py --run run_002
```

分析脚本目前针对对应已保存实验编号，运行前查看输入路径。视频是记录轨迹的运动学渲染，不是新的物理仿真。

指令覆盖对照使用 `prepare_context.py --job train_command_coverage --output <新目录>` 打包，在该运行包内执行 `bash train_command_coverage.sh smoke` 或 `full`。短训练先检查实际采样与模型加载，再进行两个 seed 的正式训练和新旧模型配对评估。配置、预算和任务状态见 `experiments/phase2_go2_command_coverage/README.md`。

## 服务器布局

根目录：`/labdata2/congyuwang/RobotRL_kitless`，容器挂载为 `/robotrl`。

```text
code/phase2_isaaclab/   与本地相同结构的当前源码
experiments/           正式实验、指标、模型和当时的源码快照
contexts/              新提交任务的最小运行包
archives/              旧任务包、早期诊断和迁移清单
assets/                机器人资产
cache/                 编译与运行缓存
```

历史 `/robotrl/run_010` 现为 `/robotrl/experiments/phase2_go2_baseline/run_001`；其他正式类别统一增加 `experiments/` 层。当前 job 路径已更新；历史实验中的源码快照和日志不改写，可通过服务器 `archives/reorganization_001/manifest.json` 查找原路径到新路径的映射。

实验目录内保存的源码是产生结果时的证据，不能与 `code/` 下正在维护的源码混用。旧快照的绝对路径可能已变化，复现时应依据迁移清单解析，而不是直接原样运行。

## 整理记录

本地整理前备份位于 `archives/phase2_code_cleanup_002/before_cleanup.zip`，文件映射在同目录 `path_mapping.json`。服务器迁移前先确认任务全部终止，迁移后逐文件核验哈希，不删除模型、结果、资产或缓存。更早的诊断代码仍保留于 `archives/phase2_code_cleanup_001/legacy/`。
