# PPO-Clip：CartPole 两个训练 seed

2026-10-15 完成核验。每组训练 200 次更新、25,600 个环境步；rollout=128，minibatch=32，epochs=4，gamma=0.99，GAE lambda=0.95，clip=0.2。actor / critic 学习率为 3e-4 / 1e-3，参数独立，未启用 advantage normalization。

训练期间使用确定性 argmax 动作，在 seeds 10000–10019 上每 10 次更新评估。best 由这组评估选择；独立复测加载已保存模型，使用 seeds 20000–20019，报告 20 局回报均值 ± 回合间标准差。

| 实验 | 训练 seed | 首次评估满分的环境步数 | 最终训练期评估 | best 独立复测 | last 独立复测 |
|---|---|---:|---|---|---|
| [run_001](run_001/) | 0 | 6,400 | 498.3 ± 7.4 | 500.0 ± 0.0 | 496.8 ± 10.4 |
| [run_002](run_002/) | 1 | 3,840 | 500.0 ± 0.0 | 500.0 ± 0.0 | 500.0 ± 0.0 |

两组均满足连续三次训练期评估 ≥475，且最终模型独立评估 ≥475 的阶段标准。全部 201 行日志（含初始评估）的已记录数值有限。两个训练 seed 仅构成初步复现；回合间标准差不是训练 seed 间的不确定性。PPO 重复使用每批样本，不能将相同环境步预算解释为与单次更新 actor-critic 相同的计算预算。

## 文件

- `config.json`：超参数、版本、训练时 Git 状态及源码哈希。
- `metrics.csv`：回报、actor / critic loss、更新后的 approximate KL、clip fraction 和 entropy。
- `curve.png`：训练与评估曲线。
- `summary.json`：完成状态和最佳 / 最终评估摘要。
- `source/`：训练时的源码快照。
- `reload_evaluation.json`：此次 best / last 重新加载评估结果及 checkpoint SHA-256。
- `best.pt`、`last.pt`：模型与优化器状态，仅保留本地，按仓库规则不上传 GitHub。

## 复现

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe .\phase1_ppo\ppo.py --seed 0 --updates 200
.\.venv\Scripts\python.exe .\phase1_ppo\ppo.py --seed 1 --updates 200
.\.venv\Scripts\python.exe .\phase1_ppo\evaluate.py .\experiments\phase1_ppo\run_001\best.pt --episodes 20 --seed-start 20000
```

新训练自动生成新的 `run_NNN` 目录；评估时替换为实际模型路径。下一阶段是连续动作策略与 HalfCheetah。
