# G1：固定镜像环境核查

日期：2026-10-19。仅检查任务、资产和最小仿真，不训练策略。

**最小仿真已通过。** 任务 `dd4b64fb-5d02-41b9-a1b8-90acfb54ea78` 于2026-10-19 16:29（北京时间）成功退出。

- 4环境、1100个策略步；观测123维、动作37维；观测、奖励、接触力均有限。
- 69次躯干触地终止及自动reset，另完成两次显式reset；未出现时间截断。零动作本身不代表平衡策略，跌倒不影响本次链路通过结论。
- 奖励范围约 -4.1044 至 0.03856，峰值接触力约2305.96 N；这些是零动作预检数据，不是行为验收指标。
- USD包含44个body、37个可动关节；3个碰撞几何位于躯干、左脚和右脚。资产质量/惯量已读取。
- 详细证据：[步进结果](smoke/smoke.json)、[资产](smoke/asset.json)、[运行环境与源码版本](smoke/inventory.json)。完整配置与日志另保留在本地/NAS。

这只证明当前任务和后端可加载、步进并重置，不证明训练收敛、真实动力学精度或G1视频渲染可用。后续已接入 [短训练验证](../../phase3_g1_smoke/run_001/README.md)。

## 固定环境

- 镜像：NVIDIA Isaac Lab，固定版本见[环境配置](../../../phase3/environment/environment.yaml)。
- 配置解析在 `64c128t_512_4090` 完成；后续4090被占满，最小仿真改用 `128c256t_768_6000Ada` 的单张 RTX 6000 Ada。镜像和仿真参数相同，正式训练的性能仍需另测。
- 实际版本：Isaac Lab 17.0.2、PyTorch 2.11.0+cu128、Newton 1.5.2、Warp 1.16.0、RSL-RL 5.4.1、Gymnasium 1.3.0。
- 实际任务：`Isaac-Velocity-Flat-G1`；另有 `Isaac-Velocity-Rough-G1` 及其他操作任务，首轮不使用。
- 原始资产：`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd`。
- NAS 资产：`/labdata2/congyuwang/RobotRL_kitless/assets/G1/g1_minimal.usd`，容器中为 `/robotrl/assets/G1/g1_minimal.usd`。仅改变资产存放位置。
- 资产大小：22,164,053字节。碰撞和惯量已由容器内 USD 解析。

## 复现入口

核查脚本与调度配置在 [phase3/environment](../../../phase3/environment/)。将其中三个文件放入远程任务 context，使用该目录的 `environment.yaml` 提交：

```bash
det command run -d --config-file environment.yaml --context . \
  bash inspect_g1.sh /robotrl/experiments/phase3_g1_environment/run_002 \
  --steps 1100 --num_envs 4 --asset /robotrl/assets/G1/g1_minimal.usd
```

不带 `--steps` 时只解析配置；运行仿真时覆盖环境数、seed=0和本地资产位置。默认奖励、执行器、终止、物理配置保持官方值。结果目录使用新编号。

零动作对应默认关节位置目标，不是零力矩；此测试不要求机器人站稳，也不证明训练收敛。后续已完成G1姿态回放，见[基线结果](../../phase3_g1_baseline/README.md)；姿态回放不运行新物理仿真。

本地/NAS 保留完整解析配置、资产信息、源码快照和运行日志；GitHub仅保留紧凑结果与文档，源码工具存于 `phase3/environment/`。
