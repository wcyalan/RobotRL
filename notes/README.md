# 学习笔记

标题统一采用“主题：内容”的格式，文件按学习顺序连续编号：01–03 强化学习基础、04–06 Go2、07–09 人形机器人、10 仿真到真实迁移。

## 强化学习基础

- [策略梯度：强化学习方法与策略优化](01_policy_gradient.md)
- [GAE：λ-Return 与优势估计](02_gae.md)
- [PPO：从策略梯度到近端策略优化](03_ppo.md)

## Go2 仿真实践

- [Go2：任务配置与训练环境](04_go2_task_config.md)
- [Go2：策略输出与控制链路](05_go2_control_pipeline.md)
- [Go2：策略导出、推理接口与复现](06_go2_deployment.md)

## 人形机器人学习

学习流程见 [phase3 结果与历史流程](../phase3/README.md)。

- [人形机器人：双足平衡、接触与动力学](07_humanoid_balance.md)
- [人形机器人：RL 任务与 PPO 训练](08_humanoid_rl.md)
- [G1：任务配置与行为评估](09_g1_configuration_and_evaluation.md)

## 仿真到真实迁移

- [Sim-to-Real：主流方法、前沿路线与机器人部署](10_sim_to_real.md)

涵盖系统辨识、域随机化、鲁棒训练、特权信息、RMA、残差和视觉适配；介绍ASAP、RialTo、TRANSIC及2025–2026年联合训练/可微辨识工作，并区分实机证据、sim-to-sim和姿态回放。原论文链接随正文给出。

## 本轮范围外的可选阅读

本轮于2026-10-22收尾；以下内容未完成，不列为必须补齐的任务。已有实验见[实验索引](../experiments/README.md)。

- [ ] 特权学习：非对称 Actor-Critic、Teacher-Student 蒸馏与 RMA
- [ ] 奖励设计：任务目标、行为塑形与正则化
- [ ] 虚拟细胞：机器人 RL 与细胞干预的联系

## 写作约定（数学渲染）

- 行内公式 `$...$` 两侧与中文之间**留一个空格**（GitHub 的 `$` 识别对紧贴中文/全角标点的情况会失效）
- `$$` 公式块内**不放中文**（MathJax 对 `\text{}` 中的 CJK 字符渲染有 bug），说明写在块外
- 命名函数使用 `\mathrm{clip}` 等写法，避免被渲染器禁用的 `\operatorname` 命令
- 数学公式中使用 `\ell_t^{\mathrm{old}}` 等数学记号，避免带多个下划线的代码变量名触发重复下标错误
- PPO 笔记的伪代码以 SVG 矢量图直接嵌入正文

## 参考资料

- [知乎：PPO 算法理解——从策略梯度开始](https://zhuanlan.zhihu.com/p/614115887)（03 的演进叙事主线）
- [Berkeley CS285 Lecture 5: Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)（01 的极大似然类比、国际象棋方差例子、三板斧框架）
- OpenAI Spinning Up（算法 + 参考实现）
- *The 37 Implementation Details of PPO*（03 §4.4 清单出处）
