# 理论笔记

- [x] [RL 方法框架与策略梯度](01_policy_gradient.md) — value/policy/actor-critic 三分法、策略梯度定理、PG=加权极大似然类比、高方差与三板斧（causality/baseline/critic）、on-policy
- [x] [λ-Return 与 GAE(λ)](02_gae.md) — 定义、在 PPO 中的角色（policy 信号 + critic target）、γ vs λ、默认值诊断
- [x] [PPO：从策略梯度到近端策略优化](03_ppo.md) — 详细版：ratio / surrogate 推导、自然梯度与 TRPO、Clip 六情形及数值例子、REINFORCE / A2C / 自然梯度 / TRPO / PPO-Penalty / PPO-Clip 六套伪代码、GAE / loss / 边界处理、连续动作、诊断与 SB3 源码对照
- [ ] 特权学习：LUPI / 非对称 actor-critic / teacher-student 蒸馏 / RMA
- [ ] 机器人奖励设计三分类（任务/塑形/正则）与 reward hacking 案例
- [ ] sim-to-real 方法总览（DR、ADR、SysID、real2sim、残差策略、world model）
- [ ] 虚拟细胞项目：机器人 RL → 细胞干预的映射表与 MDP 形式化

## 人形机器人学习

人形机器人笔记与其他理论笔记统一存放在本目录；学习流程见 [phase3计划](../phase3/README.md)：

1. [双足平衡、接触与动力学](01_humanoid_balance.md)
2. [人形RL任务与PPO训练](02_humanoid_rl.md)
3. [G1配置阅读与行为评估](03_g1_configuration_and_evaluation.md)

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
