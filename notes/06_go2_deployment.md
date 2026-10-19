# Go2：策略导出、推理接口与复现

Go2学习项目已于2026-10-19收尾：完成仿真训练、独立评估、模型导出与推理链路；不追加训练、重复评估或噪声对照。当前模型保留为学习项目成果，尚未达到所有行为条件稳定通过，也未做真机或跨物理引擎部署。

|交付|最终结果与入口|
|---|---|
|模型|训练seed0、1各保留model_450；下文列出加载路径，不使用退化的最终迭代替换|
|独立行为评估|[检查点验收](../experiments/phase2_go2_stop_sequence_extended/validation_001/README.md)：12/12行为条件达标，含对照/直行的1024回合无提前终止|
|导出与闭环|[冻结报告](../experiments/phase2_go2_deployment/run_001/README.md)：每模型32,000条观测动作误差0；闭环10/12行为条件达标，512回合无提前终止|
|重复表现|[薄弱行为验证](../experiments/phase2_go2_deployment/repeatability_001/README.md)：5/6组达标，192回合无提前终止；启停仍有一组27/32，门槛29/32|
|DR对照|[最终审计](../experiments/phase2_go2_mass_friction/AUDIT.md)：4次训练、20条件、640回合，改善未跨训练seed复现|
|复现数据|[数据位置与获取](../experiments/ARTIFACTS.md)：模型、配置、原始结果、源码快照与复现包保留本地/NAS|

停车失败主要体现为转速未及时满足连续稳定窗口；未证明具体根因。原验收阈值和负结果保持不变。冻结报告中“下一步继续验证、暂不进入人形”的文字是当时决定；当前项目状态以本说明为准。`reproduction_bundle.zip`及其内部文档保持原样，65项文件校验通过，不改写历史证据。

## 固定模型

本轮最终保留的模型是 `experiments/phase2_go2_stop_sequence_extended/run_001`（训练seed0）和 `run_002`（seed1）各自logs下的model_450.pt。这两个检查点经筛选后在eval45000/46000完成独立行为验收，不能用model_extended.pt替代（seed0最终模型右转退化）。

导出文件在 `experiments/phase2_go2_deployment/run_001/run_001/policy.pt` 与 `run_002/policy.pt`。各manifest.json记录父模型和导出文件SHA256。TorchScript用于当前固定运行环境；本地PyTorch2.14发出deprecated提示，格式选择不意味着它是新项目通用首选。服务器2.11的兼容性由实际加载/推理验证确认，不能仅靠版本推测。

## 输入与输出契约

actor为48→128→128→128→12的ELU MLP。输入float32、形状[B,48]，输出[B,12]确定性均值动作；高斯标准差与critic不进入部署策略。当前actor没有观测归一化，没有历史堆叠，无需维护隐藏状态，也不做[-1,1]裁剪。

|索引|输入|坐标/单位|
|---|---|---|
|0–2|机身线速度|机身系，m/s|
|3–5|机身角速度|机身系，rad/s|
|6–8|重力单位方向投影|机身系，无量纲|
|9–11|目标vx、vy、yaw rate|机身系，m/s、m/s、rad/s|
|12–23|关节位置减默认姿态|rad|
|24–35|关节速度（默认速度0）|rad/s|
|36–47|上一时刻实际交给action manager的动作|无量纲，未乘0.25|

关节顺序依次是FL、FR、RL、RR，每条腿hip、thigh、calf。默认姿态为[0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1.0,-1.5,-0.1,1.0,-1.5] rad。目标角度 `q_target = q_default + 0.25 * action`。策略50Hz，物理步200Hz，每动作保持4个物理步；后端内部子步另计。reset时上一动作清零；自动reset环境不能带入上一回合动作。改变指令后应先刷新观测再推理。

`phase2_isaaclab/deployment/inference.py` 只依赖PyTorch，提供观测拼接、策略推理和目标角转换。输入应已完成世界系到机身系转换，不能将世界系速度直接传入。四元数存储顺序为XYZW；默认姿态和顺序须在更换资产时重新核验。训练和当前验收保留观测噪声；独立真实传感器接口不应在已有测量噪声上机械叠加仿真噪声。

当前训练配置未裁剪动作或位置目标；不能把软关节限位当作自动目标夹紧。执行器stiffness25、damping0.5；实际力矩与速度约束由当前Newton执行器实现。详见05_go2_control_pipeline.md。本轮复用同一仿真执行器，并未实现硬件电机控制或新物理引擎的力矩约束。

## 分层验证

1. 导出脚本严格匹配checkpoint键与权重尺寸，保存TorchScript。合成输入批量1/32/256、幅度0.1/1/5，比较重建PyTorch网络和重新加载的导出网络，atol=rtol=1e-5。该检查不能单独证明与官方actor一致。
2. 服务器官方OnPolicyRunner加载原checkpoint，每步将同一真实观测分别输入官方actor和导出actor，检查动作一致性；两个模型各32环境完整20秒启停轨迹，统计最大误差与样本数。
3. 只有上述比较成功后，关闭官方runner构造，直接用torch.jit.load的策略运行原任务闭环。两个训练seed × 两个eval seed × 左转/右转/启停/直行，共16项512回合。行为仍按原29/32门槛评分，直行报告误差。

独立闭环在这里指独立于PPO训练runner的推理循环；继续复用Isaac Lab观测管理器、action manager和物理执行器。不等于新引擎sim-to-sim或真机部署。相同seed重跑的轨迹可能有波动，比较动作必须用同一批输入，不能要求两次独立仿真逐帧相同。

## 复现入口

在仓库根目录使用`.venv/Scripts/python.exe phase2_isaaclab/deployment/export_policy.py --checkpoint <model_450.pt> --output <新的导出目录>`。导出目录必须不存在，避免覆盖冻结策略。

远程运行包使用 `python phase2_isaaclab/prepare_context.py --job validate_export --output <新运行包目录>`。固定镜像及NAS挂载见phase2_isaaclab/configs/train_baseline.yaml。先将导出策略及manifest上传到实验目录，再用 `bash validate_export.sh` 执行。该job固定run_001且有防覆盖检查，复现实验须先改成新的编号，保留原模型路径与哈希。

无需PPO训练runner的单项推理入口：`python evaluate_go2.py --model <父检查点> --jit_policy <policy.pt> --output <结果.json> --behavior stop_restart --eval_seed 45000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd`。其中model在此模式只用于来源哈希，不加载网络权重；网络来自jit_policy。添加`--verify_jit`才构造官方runner作参考比较。

远程验证已完整结束：官方actor比较各32,000条观测最大误差0；导出策略独立闭环16项512回合无提前终止，但12项行为中10项通过，不能标为全部行为验收通过。结果见experiments/phase2_go2_deployment/run_001/README.md和acceptance.json。工程链路与行为稳定性分开验收。

独立程序的最小调用（在仓库根目录运行；这里使用示例静态观测，仅测试加载，不表示机器人真实状态）：

```python
import sys
import torch
sys.path.insert(0, 'phase2_isaaclab/deployment')
from inference import Go2Policy, observation, DEFAULT_POSITION

policy = Go2Policy('experiments/phase2_go2_deployment/run_001/run_001/policy.pt')
zero3 = torch.zeros(1, 3)
gravity = torch.tensor([[0., 0., -1.]])
q = torch.tensor([DEFAULT_POSITION])
previous_action = torch.zeros(1, 12)
obs = observation(zero3, zero3, gravity, zero3, q, torch.zeros(1, 12), previous_action)
action = policy(obs)
q_target = policy.position_targets(action)
print(action, q_target)
```

进入实际闭环后，必须每20ms用新状态构造观测并保存刚执行的action作为下一步previous_action；每个回合reset清零。上述代码只计算目标，不发送电机指令。仿真入口evaluate_go2.py会通过action manager执行动作，不能将q_target再次当作未缩放action传入，否则会重复加默认姿态和缩放。
