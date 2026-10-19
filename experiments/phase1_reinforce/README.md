# REINFORCE 实验记录

- [2026-10-13 初次训练日志](run_001/README.md)：用户原始运行的日志与曲线；原次模型未保存。

## 新训练自动保存

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe .\phase1_ppo\reinforce.py
```

每次运行在本分类下按 `run_001`、`run_002` 顺序创建目录，接着已有最大编号递增，避免覆盖旧实验。`run_001` 是初次日志归档，`run_002` 是首次保存模型的重跑。保存：

- `config.json`：配置、依赖版本、Git 状态和源码版本。
- `source/`：实际运行的源码快照。
- `metrics.csv`：每次更新的训练指标及定期评估；初始评估标为 update=0。
- `curve.png`、`summary.json`：结束或正常捕获中断时生成的曲线与汇总。
- `last.pt`：最近完成并记录的更新对应的策略、优化器状态和训练步数，每次更新写入。
- `best.pt`：固定评估种子上平均回报最高的模型；同分保留最早达到该分数的模型。

`*.pt` 按仓库规则只保存在本地，不上传 GitHub。评估排名依据固定种子，应另用新种子测试泛化。快照可以加载评估；尚未实现环境状态与随机数状态的精确断点续训。

加载评估（将路径替换为终端输出目录中的模型路径）：

```powershell
.\.venv\Scripts\python.exe .\phase1_ppo\evaluate.py "experiments/phase1_reinforce/run_002/best.pt" --episodes 20 --seed-start 20000
```

添加 `--render` 可观看。新训练仍然从随机初始化开始，不会自动加载旧模型。
