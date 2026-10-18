# 实验结果索引

各算法目录独立从 `run_001` 连续编号。保留正式训练及独立评估、模型、曲线和回放；低回报不等于运行失败，因此完整 SB3 对照仍保留。

GitHub与本地数据的区别、模型/轨迹获取及复现包说明见[实验文件说明](ARTIFACTS.md)。

Go2实验主线按发生顺序：官方基线 → 质量/摩擦对照 → 指令覆盖负结果 → 转向覆盖改善 → 停车惩罚负结果 → 启停序列 → 延长训练与检查点筛选 → model_450独立验证 → 导出闭环。各实验不再反复重编号；使用下列入口定位结果。

| Go2 实验类别 | 入口 | 结论 |
|---|---|---|
| 质量/摩擦 | [协议](phase2_go2_mass_friction/README.md)、[初步汇总](phase2_go2_mass_friction/preliminary_summary.json) | 四组完成，保留DR退化案例，完整配置/接触审计仍以报告状态为准 |
| 指令覆盖 | [记录](phase2_go2_command_coverage/README.md) | 第一轮混合覆盖未解决左转 |
| 转向覆盖 | [记录](phase2_go2_turn_coverage/README.md) | 转向改善，启停稳定性不足 |
| 停车惩罚 | [记录](phase2_go2_stop_stability/README.md) | 负结果，保留证据 |
| 启停序列 | [记录](phase2_go2_stop_sequence/README.md) | 启停有所改善，重复验证未全过 |
| 延长训练 | [记录](phase2_go2_stop_sequence_extended/README.md) | 最终seed0右转退化，中间model_450更均衡 |
| 检查点独立验证 | [验收](phase2_go2_stop_sequence_extended/validation_001/README.md) | 两seed的12项行为条件全部达标 |
| 导出闭环 | [报告](phase2_go2_deployment/run_001/README.md) | 动作一致性误差0；重跑行为10/12项达标，尚有稳定性问题 |

失败、短测试和诊断记录已从主目录清出，完整原件见 [备份与编号映射](../archives/experiment_cleanup_001/manifest.json)。远程原始编号不变；历史日志和源码快照中的旧编号可据此追溯。

| 实验 | 编号 | 训练步数 | 结果说明 |
|---|---|---:|---|
| phase0_sb3_ppo | [run_001](phase0_sb3_ppo/run_001/) | — | 初始环境基线，保留模型及曲线 |
| phase1_baseline | [run_001](phase1_baseline/run_001/) | 188461 | 训练期间最佳评估均值 500.00 |
| phase1_gae | [run_001](phase1_gae/run_001/) | 25600 | 训练期间最佳评估均值 500.00 |
| phase1_gae | [run_002](phase1_gae/run_002/) | 25600 | 训练期间最佳评估均值 500.00 |
| phase1_nstep | [run_001](phase1_nstep/run_001/) | 25600 | 训练期间最佳评估均值 493.25 |
| phase1_ppo | [run_001](phase1_ppo/run_001/) | 25600 | 训练期间最佳评估均值 500.00 |
| phase1_ppo | [run_002](phase1_ppo/run_002/) | 25600 | 训练期间最佳评估均值 500.00 |
| phase1_ppo_continuous | [run_001](phase1_ppo_continuous/run_001/) | 1024000 | 训练期间最佳评估均值 5282.71 |
| phase1_ppo_continuous | [run_002](phase1_ppo_continuous/run_002/) | 1024000 | 训练期间最佳评估均值 4723.13 |
| phase1_reinforce | [run_001](phase1_reinforce/run_001/) | 182228 | 训练期间最佳评估均值 500.00；原始日志记录，无模型 |
| phase1_reinforce | [run_002](phase1_reinforce/run_002/) | 182228 | 训练期间最佳评估均值 500.00 |
| phase1_sb3_continuous | [run_001](phase1_sb3_continuous/run_001/) | 1024000 | 训练期间最佳评估均值 2305.18；从中断模型续训 |
| phase1_sb3_continuous | [run_002](phase1_sb3_continuous/run_002/) | 1024000 | 训练期间最佳评估均值 1458.14 |
| phase1_sb3_official | [run_001](phase1_sb3_official/run_001/) | 1024000 | 训练期间最佳评估均值 1607.19 |
| phase1_sb3_official | [run_002](phase1_sb3_official/run_002/) | 1024000 | 训练期间最佳评估均值 1465.89 |
| phase2_go2_baseline | [run_001](phase2_go2_baseline/run_001/) | 29491200 | 完整官方基线；0.5 m/s 指令实测均速 0.511 m/s，20 秒视频已保存 |
| phase2_go2_stop_sequence_extended | [独立验收](phase2_go2_stop_sequence_extended/validation_001/) | — | 两seed的model_450通过新评估seed的全部12项行为条件；1024回合无提前终止 |
| phase2_go2_deployment | [run_001](phase2_go2_deployment/run_001/) | 无新增训练 | 两模型每个32,000条真实观测与官方actor最大误差0；独立闭环16项完成，12项行为中10项达标 |

Go2 视频：[20 秒前进回放](phase2_go2_baseline/run_001/videos/go2_forward_20s.mp4)。

表中训练期间最佳评估用于索引，不代替独立测试结果；详细结果保留在各运行的评估文件中。

- [Go2 run_002](phase2_go2_baseline/run_002/)：官方基线 seed=0；随机指令回报 40.37；固定前进提前终止 0/16。

- [Go2 run_003](phase2_go2_baseline/run_003/)：官方基线 seed=1；随机指令回报 40.23；固定前进提前终止 1/16。

三组完整比较：[Go2 多 seed 结果](phase2_go2_baseline/README.md)。

- [Go2 延迟随机化训练 run_001](phase2_go2_delay_training/run_001/README.md)：seed=42，4096 环境、300 轮；0/20/40 ms 三组评估的速度跟踪误差均高于原策略，保留为完整负结果，不替换基线。
