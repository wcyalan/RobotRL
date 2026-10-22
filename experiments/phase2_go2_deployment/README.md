# Go2：最终部署证据

最终选定两个训练 seed 的 model_450。按下面顺序阅读即可区分模型选择、独立验收和导出后的重复性。

|层级|证据|结论|
|---|---|---|
|模型选择|[检查点筛选](../phase2_go2_stop_sequence_extended/checkpoints_001/README.md)|末期模型右转退化，中间 model_450 更均衡|
|独立验收|[新评估 seed 验收](../phase2_go2_stop_sequence_extended/validation_001/README.md)|12/12 行为条件达标；含对照共 32 组 1024 回合无提前终止|
|导出与闭环|[部署包](run_001/README.md)、[接口](run_001/INTERFACE_AND_REPRODUCTION.md)|每模型 32,000 条观测动作误差为 0；闭环行为 10/12 达标|
|重复验证|[薄弱行为与停车诊断](repeatability_001/README.md)|5/6 条件达标，稳定性未全部通过|

动作一致性通过不保证重跑行为全部通过。本项目完成仿真部署准备，未开展真机验证；权重与完整复现包不包含在 Git 克隆中。

[返回实验索引](../README.md)
