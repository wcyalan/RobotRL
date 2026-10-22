# HalfCheetah：连续动作 PPO 与 SB3 对照

手写高斯 PPO 两个训练 seed 各运行 1,024,000 步。保留配置、曲线、训练指标和独立评估；训练期间最佳回报不能替代独立测试。

|证据|入口|
|---|---|
|手写 PPO 独立评估与协议|[comparison_001.json](comparison_001.json)|
|seed 0 / 1 配置|[run_001](run_001/config.json)、[run_002](run_002/config.json)|
|seed 0 / 1 曲线|[run_001](run_001/curve.png)、[run_002](run_002/curve.png)|
|SB3 对照的配置与独立评估|[紧凑对照材料](comparisons/README.md)|

手写最佳检查点在 20 个独立评估初始种子上的平均回报约为 4793.24 / 4792.53；seed 0 仍有 2/20 回合出现倒置，seed 1 为 0/20。回报提高不等于步态可靠。

SB3 自定义配置组与构造器默认配置组分开记录，默认配置不等于 RL Zoo 调优配置。预处理、超参数及运行方式存在差异，两个训练 seed 只支持初步比较，不能据此宣称手写 PPO 普遍优于 SB3。

[返回实验索引](../README.md)
