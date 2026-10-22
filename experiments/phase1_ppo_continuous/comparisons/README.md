# SB3 紧凑对照证据

以下 JSON 原样复制自原实验目录，未改变数值和内部来源路径。JSON 中的旧路径用于标识原始证据，不保证精简克隆包含全部依赖。

|分组|配置|独立评估|
|---|---|---|
|自定义配置 SB3|[seed 0](phase1_sb3_continuous/run_001/config.json)、[seed 1](phase1_sb3_continuous/run_002/config.json)|[seed 0](phase1_sb3_continuous/run_001/replay_001/replay_summary.json)、[seed 1](phase1_sb3_continuous/run_002/replay_001/replay_summary.json)|
|构造器默认配置 SB3|[seed 0](phase1_sb3_official/run_001/config.json)、[seed 1](phase1_sb3_official/run_002/config.json)|[比较汇总](phase1_sb3_official/comparison_001.json)|

自定义配置组 seed 0 曾从中断模型续训，不应描述成所有运行方式完全一致。完整曲线、逐帧预览、日志与源码快照留在原本地目录；此前公开材料亦可查[固定历史版本](https://github.com/wcyalan/RobotRL/tree/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments)。

[返回 HalfCheetah](../README.md)
