# G1：模型导出与独立推理

使用原始官方配置的两个最终模型（训练seed=0、1，`model_1499.pt`）。20%零指令模型只作为对照，不替代主基线。不追加训练，不修改原行为验收阈值；精确停车作为已知局限保留。

当前：两个模型的导出、官方动作对照和本地CPU复核已通过，见[结果](../../experiments/phase3_g1_deployment/README.md)。独立闭环24组768回合已完成，逐组通过数与原模型一致，无提前终止；精确停车仍作为已知局限保留。

## 导出与动作一致性

入口：`export_policy.py` → `verify_export.py`，远程串行入口 `validate_export.sh`。正式导出在训练时的固定镜像内执行，避免本地与远程PyTorch版本差异。结果放在 `experiments/phase3_g1_deployment/run_001` 和 `run_002`。

- 输入：float32 `[batch,123]`，按任务的policy观测顺序，无归一化、历史窗口或循环状态。
- 网络：123 → 256 → 128 → 128 → 37，隐藏层ELU。
- 输出：确定性均值动作 `[batch,37]`，尚未进行位置缩放。环境再计算 `默认关节位置 + 0.5 × 动作`。不要把输出直接当成关节角。
- 观测顺序：机身线速度3、机身角速度3、投影重力3、速度指令3、相对默认关节位置37、相对默认关节速度37、上一动作37。确切关节顺序保存于验证结果，必须一致。
- 每个模型先在多批量、多输入幅度下核对重载TorchScript与重建MLP。随后由官方策略驱动32个仿真环境1500步，在相同真实观测上比较官方推理与导出推理，绝对/相对容差均为1e-5。
- 覆盖站立、低/中速前进、左右转、停止，以及自动reset和再次显式reset；保存4800条观测与参考动作供本地复核。

这一检查验证导出和推理接口。官方策略仍负责驱动环境，不据此宣称导出模型已完成独立闭环行为验收。

远程将本目录三个脚本及 `../environment/environment.yaml` 放入同一context，然后执行：

```bash
det command run -d --config-file environment.yaml --context . \
  --config description=congyuwang-G1-export-validation bash validate_export.sh
```

运行前确认本项目之前的GPU任务已退出，始终最多使用一张GPU。脚本拒绝覆盖已有验证；失败先检查日志，不删除结果重跑。

## 独立闭环

入口 `evaluate_export.py` 只加载TorchScript，不加载训练checkpoint，不构造训练runner或优化器。沿用Isaac Lab环境及RSL环境适配器以保持观测/步进语义；这不是一个脱离仿真依赖的机器人控制驱动。动作一致性通过后，`validate_closed_loop.sh` 串行运行两个模型×两个评估seed×六条件，仍使用原冻结协议和29/32门槛。

将 `evaluate_export.py`、`validate_closed_loop.sh`、`../evaluation/protocol.json` 和 `../environment/environment.yaml` 放入一个新context，执行：

```bash
det command run -d --config-file environment.yaml --context . \
  --config description=congyuwang-G1-independent-closed-loop bash validate_closed_loop.sh
```

结果保存于每个模型的 `closed_loop/eval_20000` 和 `eval_21000`。脚本拒绝覆盖已有运行；分组日志在对应条件目录旁。任务结束并同步后，从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe phase3/analysis/summarize_deployment.py
```

汇总器检查模型、协议、源码版本及结果完整性与每条轨迹的通过判定，并逐训练seed、评估seed、条件比较原模型。行为统计代码与原评估器保持一致，唯一策略入口改为 `policy(obs['policy'])`。独立闭环可能因数值与随机性产生小幅轨迹差异，因此如实报告误差变化，不要求整条轨迹逐位相同，也不事后调整行为门槛。

导出文件仍依赖仿真真值线速度等观测，不等同于真机部署就绪。
