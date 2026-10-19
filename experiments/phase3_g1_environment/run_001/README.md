# G1：固定镜像环境核查

日期：2026-10-19。仅检查任务、资产和最小仿真，不训练策略。

**任务与配置解析已通过，官方 G1 资产已下载；最小仿真尚未完成。** GPU 节点正在首次拉取固定镜像，不能将此状态当成后端兼容性通过。

- 已完成：实际容器内任务注册、完整环境/agent配置、软件版本和源码哈希；资产下载及文件哈希。
- 待运行：USD 碰撞/质量/惯量读取、实际关节/观测/动作维数、4环境 reset、1100个策略步（每环境22秒模拟时间，含可能的回合重置）、有限数值、接触与自动reset。
- 当前任务：`dd4b64fb-5d02-41b9-a1b8-90acfb54ea78`。远程输出：`/labdata2/congyuwang/RobotRL_kitless/experiments/phase3_g1_environment/run_001/smoke`。
- 首次步进尝试在检查脚本中把 USD 内部空引用误判成缺失依赖，尚未进入仿真就退出。已修正；未判定为资产或后端故障。之后排队的4090任务已取消，只保留上述一个预检任务。

在登录节点查看进展：

```bash
det command logs --tail 30 dd4b64fb-5d02-41b9-a1b8-90acfb54ea78
```

只有任务成功退出且生成 `smoke/smoke.json` 的 `passed=true`，才能更新最小仿真通过状态；还需查看 `automatic_reset_observed` 是否为真。不得仅凭有 `inventory.json` 判定成功。

## 固定环境

- 镜像：`nvcr.io/nvidia/isaac-lab@sha256:d5c31452ae21d0cb040b7195b3a66e202b2d2a6ac389719a04528f6cd93e7d3b`。
- 配置解析在 `64c128t_512_4090` 完成；后续4090被占满，最小仿真改用 `128c256t_768_6000Ada` 的单张 RTX 6000 Ada。镜像和仿真参数相同，正式训练的性能仍需另测。
- 实际版本：Isaac Lab 17.0.2、PyTorch 2.11.0+cu128、Newton 1.5.2、Warp 1.16.0、RSL-RL 5.4.1、Gymnasium 1.3.0。
- 镜像内没有可读取的 Isaac Lab Git HEAD；使用镜像 digest 和配置源码 SHA-256 固定来源，不编造源码提交号。
- 实际任务：`Isaac-Velocity-Flat-G1`；另有 `Isaac-Velocity-Rough-G1` 及其他操作任务，首轮不使用。
- 原始资产：`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd`。
- NAS 资产：`/labdata2/congyuwang/RobotRL_kitless/assets/G1/g1_minimal.usd`，容器中为 `/robotrl/assets/G1/g1_minimal.usd`。仅改变资产存放位置。
- 资产大小：22,164,053字节；SHA-256：`9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428`。碰撞和惯量尚待容器内 USD 解析。

## 复现入口

核查脚本与调度配置在 [phase3/environment](../../../phase3/environment/)。将其中三个文件放入远程任务 context，使用该目录的 `environment.yaml` 提交：

```bash
det command run -d --config-file environment.yaml --context . \
  bash inspect_g1.sh /robotrl/experiments/phase3_g1_environment/run_002 \
  --steps 1100 --num_envs 4 --asset /robotrl/assets/G1/g1_minimal.usd
```

不带 `--steps` 时只解析配置；运行仿真时覆盖环境数、seed=0和本地资产位置。默认奖励、执行器、终止、物理配置保持官方值。结果目录使用新编号。

零动作对应默认关节位置目标，不是零力矩；此测试不要求机器人站稳，也不证明训练收敛。尚未验证 G1 离线视频渲染。

本地/NAS 保留完整解析配置、资产信息、源码快照和运行日志；GitHub仅保留紧凑结果与文档，源码工具存于 `phase3/environment/`。
