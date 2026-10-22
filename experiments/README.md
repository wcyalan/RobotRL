# 实验索引

按“先看结果 → 查核心证据 → 追溯历史”阅读。本轮已结束，不再追加训练；GitHub 当前版本只保留下面 8 个核心实验目录。

## 1. 先看结果

|想看什么|入口|应同时记住的限制|
|---|---|---|
|G1 多指令演示|[双模型六条件回放](phase3_g1_sim2sim/demo_001/README.md)|保存状态回放，包含停车失败，不是新增实验|
|G1 跨仿真后端结果|[迁移复核](phase3_g1_sim2sim/audit_001/README.md)|24 组 768 回合，20 组达标；4 组启停未通过|
|Go2 最终模型|[部署与重复验证](phase2_go2_deployment/README.md)|独立验收 12/12；导出闭环 10/12，薄弱行为复测 5/6|
|哪些尝试没有成功|[负结果与历史目录](HISTORY.md)|保留失败结论、对照条件和证据入口|

## 2. 核心实验：按学习阶段查证据

### 手写算法（2 个目录）

|目录|保留理由|
|---|---|
|[phase1_ppo](phase1_ppo/README.md)|CartPole 两个训练 seed 的 PPO 验收、配置与曲线|
|[phase1_ppo_continuous](phase1_ppo_continuous/README.md)|HalfCheetah 连续控制、独立评估及紧凑 SB3 对照|

### Go2（3 个目录）

|目录|保留理由|
|---|---|
|[phase2_go2_stop_sequence_extended](phase2_go2_stop_sequence_extended/README.md)|最终 model_450 的检查点筛选和独立验收，保留末期退化证据|
|[phase2_go2_deployment](phase2_go2_deployment/README.md)|最终导出接口、闭环测试与重复验证|
|[phase2_go2_mass_friction](phase2_go2_mass_friction/AUDIT.md)|质量/摩擦 DR 配对实验；改善未跨训练 seed 复现|

### G1（3 个目录）

|目录|保留理由|
|---|---|
|[phase3_g1_baseline](phase3_g1_baseline/README.md)|两 seed 官方基线、学习曲线、行为评估与停车诊断|
|[phase3_g1_deployment](phase3_g1_deployment/README.md)|导出动作一致性与独立闭环验证|
|[phase3_g1_sim2sim](phase3_g1_sim2sim/README.md)|跨后端保存结果复核及多指令演示|

## 3. 历史与复现

[历史目录映射与负结果摘要](HISTORY.md)解释其余 18 个目录的用途。它们的现有本地文件保持原路径，GitHub 当前目录树不再收录；旧提交仍可追溯。

各实验内部继续使用原 run 编号，避免打断配置和脚本引用。训练权重、完整轨迹和大视频不随 Git 克隆提供；当前电脑与服务器的材料边界见[文件与复现说明](ARTIFACTS.md)。
