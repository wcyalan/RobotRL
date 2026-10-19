# G1：独立 MuJoCo 的 sim-to-sim 迁移

当前完成第一步：由训练时同一份 `g1_minimal.usd` 建立可运行的原生 MuJoCo 模型，接入两个已导出的策略，完成接口检查和单回合闭环预检。结果见[首轮记录](../../experiments/phase3_g1_sim2sim/run_001/README.md)。没有重新训练模型。

## 这一步迁移了什么

原环境是 Isaac Lab → Newton → MuJoCo Warp；现在是本地 Python → MuJoCo CPU。`run_policy.py` 只依赖 MuJoCo、NumPy 和 PyTorch，视频另需 imageio；加载 MJCF 后不需要 Isaac Lab、Newton 或 USD 库。构建 MJCF 时才需要 `usd-core`。

每个策略周期：读取物理状态 → 组装123维观测 → TorchScript输出37维动作 → 计算目标关节位置 → MuJoCo推进8个物理子步。机器人下一帧的姿态由动力学积分产生，不是把旧轨迹写回机器人。

训练资产有37个受控关节，包括手指。当前保持这套资产；不能直接换成29自由度G1而保留原策略输入输出。转换代码仅支持当前简化USD，遇到不符的关节、碰撞体或网格会报错，不是通用USD转换器。

## 接口与目标物理设置

|项目|本轮设置|
|---|---|
|机器人|原USD的44个刚体、37个转动关节、浮动基座；质量约32.239 kg|
|碰撞|左右脚和躯干共3个碰撞体；原来的长方体凸网格转换为同尺寸box；不启用自碰撞|
|策略周期|0.02 s，50 Hz|
|物理子步|0.0025 s，每次动作推进8步；对应原0.005 s × 4 decimation，每物理步2个内部子步|
|积分与约束求解|MuJoCo 3.15.0，implicitfast、Newton、100 iterations、50 line-search iterations、tolerance=1e-6|
|输入|质心线速度3、角速度3、投影重力3、指令3、相对关节位置37、关节速度37、上一动作37|
|坐标|骨盆刚体质心的线速度，转到骨盆链接坐标轴；不是全机器人质心速度，也不是骨盆链接原点速度|
|预处理|无归一化/历史堆叠；保留原观测噪声幅度；随机数来自NumPy，不与GPU的随机序列逐值配对|
|动作|默认关节位置 + 0.5 × 原始动作；不裁剪策略动作；上一动作保存原始37维输出|
|执行器|按配置恢复P、D和armature；用37对位置/速度执行器，合力矩在关节处限幅；74个内部执行器仍只对应37维策略输出|
|力矩限制|脚踝20 N·m，其余配置为300 N·m；这些是该仿真任务参数，不是G1硬件额定值|
|接触|滑动摩擦0.8、扭转0.005、滚动0.0001；condim=3；solref=(0.02,1)，solimp=(0.9,0.95,0.001)|
|初始状态|每局默认关节姿态、基座高度0.74 m、yaw=0、速度0；不随机质量、不推扰|
|终止|检查每个物理子步的躯干净接触力，超过1 N结束首回合；不自动reset拼接成绩|

观测定义已检查固定镜像的 `BaseArticulationData.root_lin_vel_b`：它指向 `root_com_lin_vel_b`。原后端对同时具有刚度和阻尼的关节采用 `POSITION_VELOCITY`，在关节处限制P+D合力矩；此处按相同方式实现。MuJoCo的BODY局部坐标是惯性主轴坐标，因此先读取世界系质心速度，再转到骨盆链接坐标，避免混用坐标轴。

两个固定辅助链接的USD质心/惯性方向未指定：`pelvis_contour_link`、`imu_link`。其质量均约1e-6 kg、惯量各向同性，转换时明确使用链接原点与单位旋转；其他刚体直接读取原质量、质心及惯量。

**接触模型尚未标定为原后端的等价模型。** 原Newton使用自身的碰撞管线，原生MuJoCo使用自身的碰撞生成与约束求解。本轮接触参数在首次预检前固定，没有依据结果调参；没有宣称静/动摩擦、接触软硬程度、关节限位响应和接触传感器采样完全一致。相关机制见[Newton的MuJoCo集成](https://newton-physics.github.io/newton/1.3.0/integrations/mujoco.html)、[MuJoCo执行器与模型参数](https://mujoco.readthedocs.io/en/stable/XMLreference.html)和[MuJoCo速度接口](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity)；本项目的版本定义以固定镜像实际源码为准。

## 本地运行

已有虚拟环境具备依赖。准备原USD、环境核查的 `official_env.json` / `smoke.json`、两份导出 `policy.pt`；它们的原有目录不变。

```powershell
# 原资产转换为动态MJCF，可重复生成；不改写原USD。
.\.venv\Scripts\python.exe phase3/sim2sim/build_model.py

# 几何、坐标、观测、PD和失败处理的回归检查。
.\.venv\Scripts\python.exe -m unittest discover -s phase3/sim2sim -p test_interface.py -v

# 两模型、六条件，每条件1个首回合。新实验使用新的顺序编号。
.\.venv\Scripts\python.exe phase3/sim2sim/run_policy.py --video --output experiments/phase3_g1_sim2sim/run_002/pilot
```

模型和网格生成于 `assets/G1/native_mujoco/`；运行时接口位于同目录的 `interface.json`。源资产与生成网格不进入Git。结果目录拒绝覆盖，保存逐条件JSON、状态/动作轨迹及汇总；视频为两个模型各20秒低速前进。`--conditions` 可指定条件，`--seeds` 指定训练seed，`--no-noise` 只用于明确标注的额外诊断，不混入默认结果。

服务器沿用现有布局：代码在 `/robotrl/code/phase3/sim2sim/`，资产和实验在 `/robotrl/assets/`、`/robotrl/experiments/`。运行时设置 `ROBOTRL_ROOT=/robotrl`，协议仍从代码目录读取；不需要复制模型到代码目录。当前结果来自本地CPU，服务器仅保存同步副本，没有另跑一轮。

## 后续顺序

1. **已完成：** 资产转换、关节映射、观测坐标、力矩控制和两个策略的原生闭环预检。
2. 核对原后端实际生成模型的接触、关节限位和执行器参数，记录仍存在的目标域差异；用相同状态/动作检查，避免直接用长时间轨迹分叉判断接口有误。
3. 接口和目标设置固定后，再扩大初始朝向、评估seed和回合数，沿用原六条件及阈值，逐模型比较。当前单回合通过不等于达到29/32。
4. 有明确失败原因后才讨论适配或训练。本轮没有增加训练、奖励修改或DR，也没有将仿真迁移描述成真机部署完成。
