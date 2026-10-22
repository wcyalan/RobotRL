# RobotRL — 机器人强化学习学习仓库

学习主线：手写 PPO → Go2 仿真行走 → 域随机化与鲁棒性评估 → Go2 部署准备与复现 → 人形机器人仿真行走。真机测试为可选扩展；没有真机也可以完成本轮学习。

实验结果统一入口：[实验索引](experiments/README.md)。各算法独立连续编号。

## 本轮已收尾（2026-10-22）

已完成从手写PPO到Go2、G1仿真训练与部署准备的学习主线，本轮不再追加训练。精确停车和重复测试中的稳定性问题作为已知限制保留，历史未完成项不改记为通过。

**先看：[Sim-to-Real方法笔记](notes/10_sim_to_real.md) · [G1双模型演示](experiments/phase3_g1_sim2sim/demo_001/README.md) · [实验索引](experiments/README.md) · [文件与复现说明](experiments/ARTIFACTS.md)。**

|成果|结论与入口|
|---|---|
|手写强化学习|REINFORCE→GAE→PPO，CartPole验收与HalfCheetah初步对照完成；[算法笔记](notes/README.md)|
|Go2|训练、任务修改、DR对照和导出闭环已完成；重复行为验证未全过，见[最终交付](notes/06_go2_deployment.md)|
|G1原模型与导出|两训练seed，各次评估24组768回合，20组达标；精确停车未通过，见[导出闭环](experiments/phase3_g1_deployment/README.md)|
|G1原生MuJoCo迁移|24组768回合，20组达标，无提前终止；[832,000策略步保存结果复核](experiments/phase3_g1_sim2sim/audit_001/README.md)|
|演示与复现|130秒、双模型、六条件、3250帧校验通过；[查看回放](experiments/phase3_g1_sim2sim/demo_001/README.md)，属于保存轨迹回放|

当前桌面有最小渲染素材和视频，完整权重、全部实验及较新的迁移代码未同步。Git克隆不包含被忽略的大文件；完整评估需要服务器上的对应材料。工程流程完成与行为全部达标分开报告，尚未开展真机验证。

## 阅读顺序

1. [理论与方法笔记](notes/README.md)：算法基础、机器人控制和 Sim-to-Real。
2. [核心实验索引](experiments/README.md)：按手写 PPO、Go2、G1 分组，只展开 8 个重要目录。
3. [负结果与历史映射](experiments/HISTORY.md)：查看未采用的尝试及其结论。
4. [材料与复现边界](experiments/ARTIFACTS.md)：区分 GitHub、当前电脑与服务器上的文件。

原阶段验收清单保存在旧提交及本地归档中，未完成练习没有改记为通过。本轮不继续训练。

## 目录结构

| 目录 | 用途 |
|---|---|
| `notes/` | 理论笔记（策略梯度定理、GAE、特权学习、奖励设计、sim-to-real 总览）|
| `phase0_setup/` | 阶段 0 工具链验证脚本 |
| `phase1_ppo/` | 阶段 1 手写算法代码 |
| [`phase2_isaaclab/`](phase2_isaaclab/README.md) | Go2 训练、独立评估、鲁棒性测试和视频回放 |
| [`phase3/`](phase3/README.md) | G1任务、训练、评估、导出与回放实现 |
| `experiments/` | 实验记录：config + 曲线 + 结论 |

Go2的Isaac Lab代码维护在 `phase2_isaaclab/`，训练、评估、分析和渲染分别位于 `training/`、`evaluation/`、`analysis/`、`visualization/`；远程任务入口与配置位于 `jobs/`、`configs/`，通过 `prepare_context.py --job` 按需打包；Isaac Lab 本体和独立 Python 环境不纳入本仓库。实验按类别使用 `run_001`、`run_002` 顺序编号。人形项目使用 `phase3/`，已包含 `environment/`、`training/`、`evaluation/`、`analysis/`、`deployment/` 和 `visualization/`，相关笔记统一放在 `notes/`，实验存于 `experiments/phase3_g1_*`；路线阶段编号与代码目录编号不强制一致，Go2 原目录保持不变。

## 约定

- 每个实验保留 config、训练曲线与对应 commit，形成审计轨迹
- 调奖励/看结果必须看渲染视频，不能只看曲线（防 reward hacking 自欺）
- 参考：OpenAI Spinning Up、《动手学强化学习》、*The 37 Implementation Details of PPO*
