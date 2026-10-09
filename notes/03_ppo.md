# PPO：从策略梯度到近端策略优化

> 参考：[知乎：PPO 算法理解——从策略梯度开始](https://zhuanlan.zhihu.com/p/614115887)、Schulman et al. 2017 *Proximal Policy Optimization Algorithms*、Schulman et al. 2015 *Trust Region Policy Optimization*。
> 演进主线一句话：**普通策略梯度不知道一步该走多远 → 每一代算法用不同方式限制"每次策略更新的幅度"**。

```
REINFORCE ──→ 自然策略梯度 ──→ TRPO ──→ PPO(Penalty / Clip)
 定长步长        KL 约束+Fisher      共轭梯度+线搜索     一阶优化+clip/罚项
```

## 1. 普通 PG 的更新问题：overshooting

策略梯度只给出方向，步长靠固定学习率。**步长过大（overshooting）在监督学习里只是这一步学坏，在 RL 里是正反馈灾难**：变差的策略 → 采到更差的数据 → 更差的更新 → 崩溃。所以"限制每次策略更新幅度"是整条演进主线——限制的不是参数 $\theta$ 的变化量，而是**策略分布本身的变化量**（参数空间 ≠ 分布空间：高斯策略 std 从 0.01 到 0.02 参数只变一点，分布完全不同）。

## 2. 自然策略梯度（2002）：用 KL 散度度量"策略变了多少"

把约束写成 $\mathrm{KL}(\pi_{\mathrm{old}} \\|\ \pi_{\mathrm{new}}) \le \delta$ ，推出更新方向 $\propto F^{-1} \nabla_\theta J$ （ $F$ = Fisher 信息矩阵，衡量分布对参数的局部敏感度），得到"随分布曲率自动伸缩的动态步长"。

遗留问题： $F^{-1}$ 求逆 $O(N^3)$ ；用近似代替约束可能违反约束；算出的更新不验证是否真改进。

## 3. TRPO（2015）：surrogate objective + 三件工程套件

两个关键构件（PPO 直接继承）：

1. **重要性采样重写目标**：让旧策略采的数据在新策略下可用——

$$
L(\theta) = \mathbb{E}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\mathrm{old}}(a \mid s)} \cdot A^{\mathrm{old}}(s a) \right]
$$

   这就是 ratio 的来源（理论上要求新旧策略足够接近，否则方差爆炸）；

2. **单调改进保证**：surrogate 改进量减去一个 KL 惩罚项仍 > 0 时，真实目标 $J$ 一定改进（定理保证"小步走一定能进步"）。

工程三件套：共轭梯度法解 $Fx = \nabla J$ （免显式求逆）、线搜索（逐步缩短直到满足 KL 约束）、改进检查（surrogate 必须上升）。

遗留问题：二阶方法慢、实现复杂、不能用 Adam。

## 4. PPO（2017）：用一阶优化的简单机制替代严谨约束

### 4.1 PPO-Penalty

把 TRPO 的硬约束改成软罚项，惩罚系数自适应：

$$
L = \mathbb{E}\left[ r_t \cdot A_t - \beta \cdot \mathrm{KL}(\pi_{\mathrm{old}} \\|\ \pi_\theta) \right]
$$

$\beta$ 动态调整：实际 KL 超过 $1.5\delta$ → $\beta \times 2$ ；太小 → $\beta \div 2$。

### 4.2 PPO-Clip（默认形态）

**不加 KL 项，直接把 ratio 裁剪到 $[1-\epsilon\ 1+\epsilon]$**：

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}\left[ \min\Big( r_t(\theta) \cdot \hat{A}_t\ \ \mathrm{clip}\big(r_t(\theta)\ 1-\epsilon\ 1+\epsilon\big) \cdot \hat{A}_t \Big) \right]
\qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)}
$$

clip 如何起作用，六种情形（ratio 高于上界/界内/低于下界 × advantage 正/负）：

| $\hat{A}$ 符号 | ratio 界内 | ratio > 1+ε | ratio < 1−ε |
|---|---|---|---|
| **$\hat{A} > 0$** （好动作，想调大概率） | 正常梯度上升 | **梯度 = 0**（不再奖励继续调大） | 拉回界内 |
| **$\hat{A} < 0$** （差动作，想调小概率） | 正常梯度下降 | 拉回界内 | **梯度 = 0**（不再奖励继续调小） |

要点：

- $\min(\cdot)$ 取两式的保守者 → 目标是 clip 的**下界**，"只惩罚越界、不奖励越界"；
- 单侧截断：越界的样本不是被"夹回来"，而是**该方向上直接不更新**——信任域外的信息被丢弃；
- 经验值 $\epsilon = 0.2$ ，几乎所有任务通用；**整个机制只依赖一阶梯度**，Adam 可用。

### 4.3 现代标准 PPO 的完整训练循环（论文 Clip + GAE + 工程细节）

```
repeat:
    1. 用 π_old 在 N 个并行环境收集 T 步（如 N=8 T=2048）→ rollout buffer
    2. 算 GAE(λ)：δ_t → Â_t = Σ (γλ)^l δ_{t+l}；value target = V + Â
    3. 批内 advantage 归一化：(Â − μ)/σ
    4. 重复 K 个 epoch（如 10），每个 epoch 切 minibatch 更新：
         loss = policy_clip_loss + c_v·value_loss + c_e·entropy_bonus
    5. 丢弃全部数据（on-policy，数据只属于产生它的策略）
```

各部分一句话：

- **value loss**： $0.5 \cdot \big( V_\theta - G_t^\lambda \big)^2$ （可再加 clip 防值函数跳变）；
- **entropy bonus**：鼓励探索，防过早收敛到确定性策略；
- **为什么能多 epoch 复用数据**：ratio + clip 就是对"数据已过期"的修正与保险——这正是 PPO 相对 REINFORCE 的样本效率来源。

### 4.4 实现细节清单（调不通先查这里，来自 *37 Implementation Details of PPO*）

- advantage 批内归一化（优先级最高）
- observation 归一化（RunningMeanStd，训练时用、评估时冻结）
- value loss 系数 0.5、entropy 系数 ~0.0、梯度全局范数裁剪 0.5
- 默认超参： $\gamma = 0.99\ \lambda = 0.95\ \epsilon = 0.2$ ， lr=3e-4 n_steps=2048 epochs=10
- 早停诊断： $\mathrm{kl}(\pi_{\mathrm{old}} \| \pi_\theta)$ 持续 > 0.02 说明步子太大

## 5. 为什么 PPO 成了默认选择

牺牲 TRPO 的理论保证，换来：一阶优化器、实现百行以内、对超参不敏感。如今是 locomotion 训练的事实标准（第 2–3 周要用的就是它），也是 RLHF 的核心组件。

## 6. 与前两篇笔记的连接

- [01_policy_gradient.md](01_policy_gradient.md)：定理给出梯度方向 = ratio·A 的理论源头（重要性采样改写自策略梯度）；
- [02_gae.md](02_gae.md)：GAE 给出 $\hat{A}$ 的估计器和 critic 的回归目标；
- 本篇：解决"一步走多远 + 旧数据怎么复用"。
- 阶段 1 的验收目标：自己写出 4.3 这个循环，在 HalfCheetah 上对齐 SB3（SB3 源码 `stable_baselines3/ppo/ppo.py` 就是 4.3 的工业级实现，写完 diff 它）。
