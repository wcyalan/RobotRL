# 负结果与历史实验

当前版本只展开 8 个核心目录；其余 18 个目录的现有本地文件保持原路径，已退出 Git 跟踪并加入忽略规则。这里保留影响结论的失败和历史映射，避免只展示成功案例。

## 必须保留的负结果

|实验|结果与解释|证据|
|---|---|---|
|Go2 质量/摩擦 DR|四次训练、640 回合；seed 0 四项改善一项退化，seed 1 五项均退化，收益未跨 seed 复现|[核心审计](phase2_go2_mass_friction/AUDIT.md)|
|Go2 延迟随机化|单训练 seed；0/20/40 ms 三条件 XY 误差均高于原策略，不能替换基线|[历史报告](https://github.com/wcyalan/RobotRL/blob/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments/phase2_go2_delay_training/run_001/README.md)|
|Go2 混合指令覆盖|同时改变类别比例和重采样间隔；seed 0 左转由 2/32 降至 0/32，两个 seed 前进误差均增加；不能单独归因某一参数|[历史报告](https://github.com/wcyalan/RobotRL/blob/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments/phase2_go2_command_coverage/README.md)|
|Go2 停车偏航惩罚|新增权重 −2.0；seed 0 左转 32→19/32、启停 27→16/32，未获得一致改善，不采用|[历史报告](https://github.com/wcyalan/RobotRL/blob/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments/phase2_go2_stop_stability/README.md)|
|Go2 延长训练末期退化|启停改善但 seed 0 右转两评估 seed 均 32→9/32；最后检查点不等于最好模型|[核心记录](phase2_go2_stop_sequence_extended/README.md)|
|Go2 部署重复性|model_450 独立验收 12/12，导出闭环 10/12，薄弱行为复测 5/6；不能宣称全条件稳定|[核心部署证据](phase2_go2_deployment/README.md)|
|G1 零指令 2%→20%|达标从 20/24 降到 14/24；两训练 seed 的右转均为 0/32，停车仍未达门槛，保留原 2% 基线|[历史报告](https://github.com/wcyalan/RobotRL/blob/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments/phase3_g1_standing/README.md)|
|G1 基线与迁移停车失败|原基线、导出闭环、原生 MuJoCo 迁移均保留停车局限；迁移复核的 4 组启停全未通过|[基线](phase3_g1_baseline/README.md)、[迁移复核](phase3_g1_sim2sim/audit_001/README.md)|

以上是各自协议下的观察，训练 seed 数有限；既不证明某类方法普遍无效，也不把平均误差改善替代预定行为门槛。

## 本地历史目录对照

|目录|原来的实验用途|
|---|---|
|`phase0_sb3_ppo`|最初 SB3 CartPole 工具链验证|
|`phase1_baseline`|Monte Carlo baseline / Actor-Critic|
|`phase1_gae`|GAE 实现与初步两 seed 验证|
|`phase1_nstep`|固定 rollout 与 n-step bootstrap|
|`phase1_reinforce`|REINFORCE 入门与保存/加载|
|`phase1_sb3_continuous`|自定义配置 SB3 HalfCheetah 对照；紧凑证据已并入核心连续 PPO 目录|
|`phase1_sb3_official`|SB3 构造器默认配置对照；紧凑证据已并入核心连续 PPO 目录|
|`phase2_go2_baseline`|Go2 官方基线及早期回放|
|`phase2_go2_behavior`|固定指令行为验收与失败诊断|
|`phase2_go2_command_coverage`|混合指令比例与切换频率对照|
|`phase2_go2_delay_training`|延迟随机化训练负结果|
|`phase2_go2_robustness`|摩擦、延迟压力测试；早期摩擦设置曾需修正|
|`phase2_go2_stop_sequence`|显式启停序列训练及初步重复验证|
|`phase2_go2_stop_stability`|零指令偏航惩罚负结果|
|`phase2_go2_turn_coverage`|增加转向覆盖；转向改善但停车不足|
|`phase3_g1_environment`|G1 资产、接口与环境预检|
|`phase3_g1_smoke`|G1 短训练及独立加载检查|
|`phase3_g1_standing`|零指令采样 2%→20% 的负结果|

此外，核心目录中的重复源码快照、原始日志、逐帧拆图、临时完成标记，以及 G1 sim2sim 的早期 `run_001` 保留本地，不再占用当前发布目录。维护代码仍在 `phase1_ppo/`、`phase2_isaaclab/` 和 `phase3/`。

## 如何追溯

本地从上述原目录进入；GitHub 可查[整理前的固定版本](https://github.com/wcyalan/RobotRL/tree/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments)，以及[原实验索引](https://github.com/wcyalan/RobotRL/blob/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments/README.md)。历史详情不再随当前版本克隆检出，但此次没有改写 Git 历史，旧提交仍公开可查，也不因此缩小完整 Git 历史下载量。

本次只保留本机已经存在的文件，不代表完整权重或服务器数据已同步。G1 20% 零指令组的模型、轨迹和训练日志在此前已清理，本次不能恢复它们。复现边界见[文件说明](ARTIFACTS.md)。

[返回核心实验索引](README.md)
