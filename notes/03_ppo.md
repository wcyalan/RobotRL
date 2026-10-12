# PPO：从策略梯度到近端策略优化

> 本篇承接 [策略梯度笔记](01_policy_gradient.md) 和 [GAE 笔记](02_gae.md)，重点回答：**已经知道策略应该往哪里更新，如何让同一批数据支持多次更新，又不把策略推得太远？**
> 数学主线对照 PPO / TRPO 原论文；实现细节对照本仓库安装的 **Stable-Baselines3 2.9.0**。文中的数值例子用于解释机制，不是实验结果。

```text
REINFORCE / 普通 actor-critic
    ↓ 用策略分布的变化量衡量步长
自然策略梯度：KL 的局部几何 → Fisher 预条件方向
    ↓ 对实际候选更新做约束检查
TRPO：surrogate + KL 约束 + 共轭梯度 + 线搜索
    ↓ 用容易做 minibatch 优化的目标替代约束求解
PPO：Penalty 或 Clip + 多 epoch 更新
```

阅读路线：§1–3 解释为什么需要 PPO；§4 逐项拆 Clip；§5–8 连接到手写实现；§9–11 用于查错和对照 SB3。

伪代码导航：REINFORCE（§1.3）→ A2C（§1.4）→ 自然策略梯度（§3.1）→ TRPO（§3.2）→ PPO-Penalty（§3.3）→ PPO-Clip（§8.2）。共用的 GAE / return target 子过程在 §6.1，算法差异速查在 §8.4。

伪代码约定：`theta` 是 actor 参数，`phi` 是 critic 参数；`theta_old` 等旧量是复制并冻结的快照；`mean` 表示对指定样本取平均。`L` 表示要最大化的目标，`loss` 表示优化器要最小化的损失。以下 actor-critic 流程采用独立参数，便于看清各自的更新；共享网络时还需考虑 §6.4 的梯度耦合。代码块描述算法步骤，具体张量实现留给手写练习。

## 1. 问题起点：方向正确，不代表可以一直沿着它走

### 1.1 普通策略梯度在当前策略附近给出改进方向

设策略参数为 $\theta$ ，critic 参数为 $\phi$ 。策略梯度的基本更新是：

$$
\theta_{\mathrm{new}} = \theta_{\mathrm{old}} + \alpha \hat{g}
$$

$$
\hat{g} = \frac{1}{B}\sum_{t=1}^{B} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat{A}_t
$$

这里的动作和 advantage 来自旧策略，梯度在旧参数处计算。它告诉我们：在旧策略附近，提高正 advantage 动作的概率，降低负 advantage 动作的概率。

难点是梯度只提供**局部信息**。走得太远以后，旧策略下的 advantage、旧策略访问过的状态，未必还能准确说明新策略的表现。有限样本、critic 误差会进一步放大这个问题。

RL 中还有一个反馈环：策略改变会改变下一批训练数据的分布。一次过大的更新可能同时损害回报与后续数据质量。不过这不是说监督学习的大步更新一定容易恢复，或 RL 一定不可恢复；关键是 RL 的数据分布也由被更新的模型决定。

### 1.2 为什么不直接限制参数变化量

同样大小的参数变化，在不同网络、不同状态或不同参数位置，可能造成完全不同的策略变化。比如高斯策略的标准差从 0.01 变成 0.02，绝对变化只有 0.01，但分布的宽度翻倍。

我们真正关心的是：**对相同状态，新旧策略给出的动作分布差了多少？** 常用度量是旧策略到新策略的 KL：

$$
D_{\mathrm{KL}}\big(\pi_{\mathrm{old}}(\cdot \mid s)\|\pi_\theta(\cdot \mid s)\big)
= \mathbb{E}_{a\sim\pi_{\mathrm{old}}(\cdot\mid s)}
\left[\log\pi_{\mathrm{old}}(a\mid s)-\log\pi_\theta(a\mid s)\right]
$$

KL 是分布差异，不是参数差异；它不对称，也不是严格意义的距离。此处固定使用上式的方向，避免与不同实现中的命名混淆。

### 1.3 REINFORCE：伪代码

先写一个只用完整 episode 回报的版本。下面按“从初始状态出发的折扣总回报”写估计量，因此保留外层 `gamma^t`，并先在每条轨迹内求和，再按轨迹数平均。

```text
输入：折扣 gamma、学习率 alpha、每轮 episode 数 M
初始化策略参数 theta

repeat:
    固定当前 theta，采集 M 条完整 episode
    对每条轨迹 i，保存状态 s_t、动作 a_t、奖励 r_t

    对每条轨迹，从最后一步向前计算 reward-to-go：
        G = 0
        for t = 最后一步 ... 0:
            G = r_t + gamma * G
            保存 G_t = G

    用当前网络重新计算所有存储动作的 log_prob_theta(a_t | s_t)
    loss = -(1 / M) * sum_i sum_t [gamma^t * log_prob_theta(a_t | s_t) * G_t]
    # G_t 是固定权重，只有当前 log_prob 对 theta 求导
    用学习率 alpha 对 loss 做一次梯度下降
    记录回报，随后用更新后的策略重新采样
```

这里的完整 episode 以所优化任务的终点为准。若把人为时间上限也当作回报结束点，优化的是该有限长度回报；若任务在截断后仍应继续，就要按 §7 引入 bootstrap，不能把缺失的后续回报直接当成 0。后面的 A2C / PPO 采用常见的 rollout 样本平均形式，对应 §2.3 说明的工程采样约定。

### 1.4 A2C：伪代码

这一步加入 critic，并允许在 episode 尚未结束时更新。此处的 A2C 指同步采样的 n-step advantage actor-critic：用 GAE 的 `lambda=1` 得到分段 n-step 回报与 advantage；改成较小的 `lambda` 就得到使用 GAE 的版本。仅在完整 MC 回报上减去学习到的 baseline，还不包含这里的 bootstrap 机制。

```text
输入：并行环境数 N、每轮步数 T、gamma、actor / critic 学习率
初始化独立的 actor theta 与 critic phi

repeat:
    在采样阶段固定 theta、phi
    从 N 个环境各采 T 步，保存状态、动作、奖励、old_value 和边界信息
    用采样时的 critic 计算需要的 next_value，含截断位置与 rollout 尾部
    (A_raw, R_target) = GAE_固定标签(这批数据, gamma, lambda=1)  # 见 §6.1
    冻结 A_raw、R_target；完成时间递推后再展平样本

    用当前 actor 计算存储动作的 log_prob
    actor_loss = -mean(log_prob * A_raw)
    对 actor_loss 做一次梯度下降，只更新 theta

    critic_loss = mean((V_phi(s) - R_target)^2)
    对 critic_loss 做一次梯度下降，只更新 phi

    记录回报与 value 误差，用更新后的策略收集下一批数据
```

与后面的 PPO 相比，这里每批数据只做一次 actor 更新，没有 ratio、Clip 或多 epoch 的 actor 复用。GAE 决定如何估计 advantage，PPO 决定如何使用这批 advantage 更新策略；两者是不同环节。

## 2. ratio 从哪里来：从真实改进量到可计算的 surrogate

本篇用 $r_t$ 表示执行 $a_t$ 获得的即时奖励，用 $\rho_t$ 表示概率比。PPO 原论文把概率比记为 $r_t(\theta)$ ，两者只是符号选择不同。

### 2.1 先看真实目标为什么难直接优化

对无限时域折扣问题，令真实目标为：

$$
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

定义归一化的折扣状态访问分布：

$$
d_\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr(s_t=s\mid\pi)
$$

performance difference identity 把新策略相对旧策略的改进写成：

$$
J(\pi_\theta)-J(\pi_{\mathrm{old}})
=\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_{\pi_\theta},a\sim\pi_\theta(\cdot\mid s)}
\left[A^{\pi_{\mathrm{old}}}(s,a)\right]
$$

这一步还没有近似。注意：advantage 属于旧策略，但状态分布属于**新策略**。后者依赖整条未来轨迹，很难在每次梯度更新时重新采样求解。

### 2.2 用旧状态分布做局部近似

TRPO / PPO 的 surrogate 固定状态分布，改成在旧策略访问过的状态上评估候选动作分布：

$$
L_{\mathrm{old}}(\theta)
=\mathbb{E}_{s\sim d_{\pi_{\mathrm{old}}},a\sim\pi_\theta(\cdot\mid s)}
\left[A^{\pi_{\mathrm{old}}}(s,a)\right]
$$

用 $L_{\mathrm{old}}(\theta)/(1-\gamma)$ 近似真实改进量；优化时可以忽略这个固定的正比例系数。加回常数 $J(\pi_{\mathrm{old}})$ 后，得到的目标在旧参数处与真实目标的值和梯度一致。越远离旧策略，固定状态分布的近似越需要警惕。

### 2.3 动作不是新策略采的，怎么办

在固定状态 $s$ 上，重要性采样给出：

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}[A^{\pi_{\mathrm{old}}}(s,a)]
=\mathbb{E}_{a\sim\pi_{\mathrm{old}}(\cdot\mid s)}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}A^{\pi_{\mathrm{old}}}(s,a)\right]
$$

要求旧策略对新策略可能选择的动作具有足够的支持。常见的 softmax 和非退化高斯策略满足相应的支持条件，但概率极小时仍可能出现数值与方差问题。

因此一批旧数据上可计算的目标是：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}
$$

$$
\hat{L}^{\mathrm{PG}}(\theta)=\frac{1}{B}\sum_{t=1}^{B}\rho_t(\theta)\hat{A}_t
$$

理解概率比时，可以想象旧策略选择某动作的概率是 0.20，新策略改成 0.24，则 $\rho=1.2$ 。它表示相对增加 20%，不是增加 0.2 个概率点，更不是整个策略只变了 20%。连续动作下分子、分母是**概率密度**。

**ratio 修正的是固定状态下的动作分布，没有修正状态访问分布的变化。** 所以 PPO 不是拿到任意旧数据、乘个 ratio 就能无偏优化当前回报的通用 off-policy 方法。实际 PPO 通常直接平均 rollout 中的样本，也不显式按上面的折扣状态分布重新加权；应区分理论推导和工程采样约定。

### 2.4 ratio 等于 1，为什么仍有梯度

刚开始优化一批数据时，新旧策略相同，因此所有 $\rho_t=1$ 。但这是当前参数处的**函数值**，不表示它对参数是常数：

$$
\nabla_\theta\rho_t(\theta)
=\rho_t(\theta)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

$$
\left.\nabla_\theta\hat{L}^{\mathrm{PG}}(\theta)\right|_{\theta=\theta_{\mathrm{old}}}
=\frac{1}{B}\sum_t\hat{A}_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

第一步就是熟悉的策略梯度。在 $\epsilon>0$ 时，Clip 在这个位置也尚未起作用。若只做一次全批量梯度更新，PPO 的 Clip 机制还没有机会改变该步的梯度；多次更新后，它才会明显区别于普通 PG。

## 3. 自然梯度、TRPO、PPO 分别怎样控制更新

### 3.1 自然策略梯度：用 KL 的局部几何改变方向

在旧参数附近，令 $\Delta\theta=\theta-\theta_{\mathrm{old}}$ 。对 surrogate 做一阶近似，对平均 KL 做二阶近似：

$$
L(\theta_{\mathrm{old}}+\Delta\theta)
\approx L(\theta_{\mathrm{old}})+g^\top\Delta\theta
$$

$$
\bar{D}_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta_{\mathrm{old}}+\Delta\theta)
\approx\frac{1}{2}\Delta\theta^\top F\Delta\theta
$$

这里 $g$ 是旧参数处的 surrogate 梯度， $F$ 是旧策略、旧状态分布下的 Fisher 信息矩阵。在常见正则条件下，它也是此处平均 KL 的 Hessian。

令 score 向量为：

$$
z_\theta(s,a)=\nabla_\theta\log\pi_\theta(a\mid s)
$$

Fisher 就是它的外积期望，在旧参数处计算：

$$
F=\mathbb{E}_{s\sim d_{\pi_{\mathrm{old}}},a\sim\pi_{\mathrm{old}}(\cdot\mid s)}
\left[z_{\theta_{\mathrm{old}}}(s,a)z_{\theta_{\mathrm{old}}}(s,a)^\top\right]
$$

局部问题是最大化 $g^\top\Delta\theta$ ，同时满足：

$$
\frac{1}{2}\Delta\theta^\top F\Delta\theta\le\delta
$$

为什么会出现逆 Fisher？引入拉格朗日乘子 $\nu>0$ ：

$$
\mathcal{Q}(\Delta\theta,\nu)=g^\top\Delta\theta
-\nu\left(\frac{1}{2}\Delta\theta^\top F\Delta\theta-\delta\right)
$$

对更新量求导并令其为零：

$$
g-\nu F\Delta\theta=0
$$

于是方向必须是 $F^{-1}g$ 的倍数。把它代回恰好用满的 KL 预算，在 $F$ 正定且 $g\ne0$ 的理想情形下得到：

$$
\Delta\theta^*
=\sqrt{\frac{2\delta}{g^\top F^{-1}g}}F^{-1}g
$$

方向是 $F^{-1}g$ ，前面的系数决定满足局部 KL 预算的步长。它不仅改变步长大小，也会改变参数空间中的方向：对分布特别敏感的方向受到更强约束。

实际神经网络中 $F$ 可能奇异或病态，通常需要 damping。困难在于处理 Fisher 曲率及其近似，不是“自然梯度一定要显式求一个大矩阵的逆”。

**自然策略梯度伪代码。** 为与后面的 TRPO 对比，使用相同的 ratio surrogate 和固定 advantage。`KL_old(theta)` 表示在旧采样状态上，对旧、新完整动作分布的 KL 取平均；可保存旧 logits 或高斯均值 / 标准差来计算。Fisher 在新旧参数相同的位置计算。

```text
输入：局部 KL 预算 delta > 0、damping 系数 zeta >= 0
初始化独立的 actor theta 与 critic phi

repeat:
    theta_old = 复制并冻结 theta
    用旧策略采集 rollout，保存 old_log_prob 和旧动作分布参数
    用采样时的 critic 计算固定 A_raw、R_target（§6.1）

    定义 L(theta) = mean(exp(log_prob_theta - old_log_prob) * A_raw)
    g = 在 theta_old 处计算 L 对 theta 的梯度
    F = 在 theta_old 处计算 KL_old(theta) 的 Hessian
    H = F + zeta * I

    if g 非零:
        求解 H * x = g                  # 无需显式求逆；大网络可用共轭梯度
        q = x^T * H * x
        if 解有效且 q > 0:
            step = sqrt(2 * delta / q) * x
            theta = theta_old + step   # 直接接受局部模型给出的候选更新
    # 无有效方向时保持 theta_old

    用固定 R_target 拟合 critic，只更新 phi
    记录回报和更新后的平均 KL，收集下一批数据
```

`zeta=0` 且求解精确时就是上面的解析更新；加入 damping 后，方向和缩放使用的是 $H=F+\zeta I$ ，不是未经修改的 Fisher。该版本只记录更新后的 KL，不做回溯接受检查，所以局部预算不能保证实际 KL 不超限。用共轭梯度求方向也不自动使算法变成 TRPO。

### 3.2 TRPO：沿自然梯度方向，检查真正的候选更新

实践中的 TRPO 最大化：

$$
\hat{\mathbb{E}}_t[\rho_t(\theta)\hat{A}_t]
$$

并要求：

$$
\hat{\mathbb{E}}_t[D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t))]\le\delta
$$

它通常用 Fisher-vector product 与共轭梯度近似求出方向，然后回溯线搜索：缩短候选步长，检查采样估计的 KL 是否满足约束，以及 surrogate 是否改善。这比直接相信局部二阶近似多了一层检查。[TRPO 原论文 §4–5、Appendix C](https://arxiv.org/abs/1502.05477)

理论保证要分层看：原论文的性能下界涉及对状态取最大值的分布差异以及精确 advantage 等条件；实践中换成采样平均 KL、近似 advantage 和数值求解后，不能声称每次真实回报都保证上升。KL 足够小也不意味着任意方向的小步都有改进。

标准 TRPO 更新流程不直接采用普通 Adam 的无约束 minibatch 更新；不能把这理解成 Adam 与 KL 惩罚目标在数学上不兼容。

**TRPO 伪代码。** 在自然梯度候选步之后加入接受 / 拒绝逻辑。下面使用常见的阻尼求解版本；`Hvp(v)` 只返回矩阵与向量的乘积，不构造完整矩阵。`backtrack` 是步长缩减系数，避免与 PPO-Penalty 的 `beta` 混用。

```text
输入：KL 预算 delta、damping zeta、0 < backtrack < 1、最多尝试次数 K_ls
初始化独立的 actor theta 与 critic phi

repeat:
    theta_old = 复制并冻结 theta
    用旧策略采 rollout，保存 old_log_prob 和旧分布参数
    计算并冻结 A_raw、R_target（§6.1）

    定义 L(theta) = mean(exp(log_prob_theta - old_log_prob) * A_raw)
    定义 KL_old(theta) = 旧采样状态上 KL(旧动作分布 || 当前动作分布) 的均值
    L_old = L(theta_old)
    g = 在 theta_old 处计算 L 的梯度
    Hvp(v) = 在 theta_old 处计算 KL_old 的 Hessian-vector product + zeta * v
    用共轭梯度近似求解 H * x = g，只调用 Hvp(v)
    q = x^T * Hvp(x)
    accepted = False

    if 解有效且 q > 0 且 g^T * x > 0:
        full_step = sqrt(2 * delta / q) * x
        for j = 0 ... K_ls-1:
            candidate = theta_old + backtrack^j * full_step
            # 每次都从同一个 theta_old 出发，不能在上个 candidate 上累加
            计算 candidate 的 KL_old 和 L，仍使用同一批旧数据
            if 两者有限 且 KL_old(candidate) <= delta 且 L(candidate) > L_old:
                theta = candidate
                accepted = True
                break

    if not accepted:
        theta = theta_old              # 全部候选失败，恢复旧参数

    用固定 R_target 拟合 critic，只更新 phi
    记录接受步长、KL、surrogate 改善与回报，重新采样
```

接受条件比较的是 `L(candidate) - L_old`，不能假定采样估计的 `L_old` 总是精确等于 0。有些实现还要求实际 surrogate 改善达到线性预测改善的一定比例。这里的 KL 使用候选网络的完整分布计算，但状态仍是采样状态；没有对所有状态验证约束，也没有重新跑环境来验证真实回报。[求解与线搜索见 TRPO Appendix C](https://arxiv.org/abs/1502.05477)

### 3.3 PPO-Penalty：把约束放进目标

PPO-Penalty 使用一阶优化器最大化：

$$
L^{\mathrm{KLPEN}}(\theta)
=\hat{\mathbb{E}}_t\left[\rho_t(\theta)\hat{A}_t
-\beta D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t))\right]
$$

直觉是“改进动作选择”与“远离旧策略”一起计价。原论文根据更新后的平均 KL 调整下一轮的 $\beta$ ：高于目标的 1.5 倍就加倍，低于目标除以 1.5 就减半。这是软约束，单次更新仍可能超过目标 KL。[PPO 原论文 §4](https://arxiv.org/abs/1707.06347)

**PPO-Penalty 伪代码。** 下面用独立 actor / critic、固定原始 advantage 和不加 entropy bonus 的简化版本，突出 KL 惩罚的作用。`beta` 在一轮 rollout 的全部 epoch 内保持固定，结束后再为下一轮调整。

```text
输入：目标 KL d_target > 0、初始惩罚系数 beta > 0、epoch 数 K、minibatch 大小 M
初始化独立的 actor theta 与 critic phi

repeat:
    用当前策略采 rollout，保存 old_log_prob、old_value 和旧分布参数
    计算并冻结 A_raw、R_target（§6.1）
    把时间和环境维展平；本轮所有更新始终使用同一个旧策略参照

    for epoch = 1 ... K:
        打乱这批数据，划分成大小为 M 的 minibatch
        for 每个 minibatch:
            ratio = exp(log_prob_theta(a | s) - old_log_prob)
            kl = mean(KL(存储的旧动作分布 || 当前动作分布))
            actor_loss = -mean(ratio * A_raw) + beta * kl
            用一阶优化器最小化 actor_loss，更新 theta
            # KL 项参与反向传播；只有旧分布参数被 detach

            critic_loss = mean((V_phi(s) - R_target)^2)
            最小化 critic_loss，只更新 phi

    用最终 actor，在整批旧状态上重新计算平均 KL，记为 d
    if d > 1.5 * d_target:
        beta = 2 * beta
    else if d < d_target / 1.5:
        beta = beta / 2
    else:
        beta 保持不变

    记录 d、beta 与回报，换新数据开始下一轮
```

与 TRPO 不同，这个基本版本不会因超出目标 KL 而回滚参数，而是增大下一轮的惩罚。它也没有 PPO-Clip 的 `min / clip`；若同时加入两种机制，就应明确标成组合变体。

### 3.4 PPO-Clip：不求约束问题，改变单个样本的奖励方式

Clip 版本不要求显式把 KL 写进 loss，而是取消“对某个样本继续往有利方向大幅改变概率”的额外收益。它容易与 Adam、minibatch、多 epoch 配合。下面将它拆成可直接核对的分段函数。

PPO-Clip 的完整伪代码集中在 §8.2；先读 §4 的目标、§6 的 loss 与 §7 的边界处理，再对照完整循环。

## 4. PPO-Clip：公式、六种情形、梯度和边界

### 4.1 完整目标中，clip 和 min 缺一不可

$$
L^{\mathrm{CLIP}}(\theta)
=\hat{\mathbb{E}}_t\left[
\min\left(\rho_t(\theta)\hat{A}_t,
\mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)
\right]
$$

这里最大化 $L^{\mathrm{CLIP}}$ ；实现时通常最小化它的负值。先把 $\hat{A}_t$ 看作已固定的权重，暂时忽略 advantage 归一化。

单纯把 ratio clamp 后乘 advantage，会在区间两侧都产生平台。这样连“好动作概率降得太多”“差动作概率升得太多”也没有纠正信号。外面的 $\min$ 保留了这些不利变化的梯度，只取消有利变化过大的额外收益。[PPO 原论文 §3](https://arxiv.org/abs/1707.06347)

### 4.2 正 advantage：鼓励提高概率，但收益到上界为止

当 $\hat{A}>0$ 时，单样本目标等价于：

$$
\ell(\rho,\hat{A})=\hat{A}\min(\rho,1+\epsilon)
$$

- 在上界之前，增大 $\rho$ 会提高目标。
- 超过上界后，目标为常数，不再奖励这个样本进一步增大概率。
- 即使已经低于下界，仍保留提高概率的梯度，因为那是纠正不利变化。

### 4.3 负 advantage：鼓励降低概率，但收益到下界为止

当 $\hat{A}<0$ 时，乘负数会翻转大小关系，因此：

$$
\ell(\rho,\hat{A})=\hat{A}\max(\rho,1-\epsilon)
$$

- 在下界以上，减小 $\rho$ 会提高目标，注意目标可能是负数，“提高”表示变得没那么负。
- 低于下界后，目标为常数，不再奖励这个样本进一步减小概率。
- 高于上界时仍保留降低概率的梯度，纠正差动作概率被放大的变化。

### 4.4 六种情形：这里讨论的是最大化目标

| advantage | ratio 区间 | 单样本目标 | 对 $\rho$ 的导数 | 此样本提供的作用 |
|---|---|---|---|---|
| $\hat{A}>0$ | $\rho<1-\epsilon$ | $\rho\hat{A}$ | $\hat{A}>0$ | 鼓励提高概率，纠正降得太多 |
| $\hat{A}>0$ | $1-\epsilon<\rho<1+\epsilon$ | $\rho\hat{A}$ | $\hat{A}>0$ | 鼓励提高概率 |
| $\hat{A}>0$ | $\rho>1+\epsilon$ | $(1+\epsilon)\hat{A}$ | $0$ | 取消继续提高概率的收益 |
| $\hat{A}<0$ | $\rho<1-\epsilon$ | $(1-\epsilon)\hat{A}$ | $0$ | 取消继续降低概率的收益 |
| $\hat{A}<0$ | $1-\epsilon<\rho<1+\epsilon$ | $\rho\hat{A}$ | $\hat{A}<0$ | 鼓励降低概率 |
| $\hat{A}<0$ | $\rho>1+\epsilon$ | $\rho\hat{A}$ | $\hat{A}<0$ | 鼓励降低概率，纠正升得太多 |

分段边界处用自动微分的边界约定处理； $\hat{A}=0$ 时没有该样本的 policy 信号。表中的“鼓励”指该项的梯度贡献，实际多样本更新的最终结果还受到其他项影响。

### 4.5 代入数字，亲自检查 min 的方向

令 $\epsilon=0.2$ ，裁剪区间为 $[0.8,1.2]$ ：

| $\hat{A}$ | $\rho$ | 原始项 $\rho\hat{A}$ | 裁剪项 | min 后目标 | 梯度是否保留 |
|---|---|---|---|---|---|
| $+2$ | $1.4$ | $2.8$ | $2.4$ | $2.4$ | 否，已经向有利方向走过上界 |
| $+2$ | $0.6$ | $1.2$ | $1.6$ | $1.2$ | 是，好动作概率降得太多 |
| $-2$ | $0.6$ | $-1.2$ | $-1.6$ | $-1.6$ | 否，已经向有利方向走过下界 |
| $-2$ | $1.4$ | $-2.8$ | $-2.4$ | $-2.8$ | 是，差动作概率升得太多 |

第三行最容易想反：选择更小的 $-1.6$ ，才不会继续奖励把差动作的概率压得更低。

### 4.6 梯度为零，究竟是什么为零

在没有进入平台的区域：

$$
\nabla_\theta\ell_t
=\hat{A}_t\rho_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

在平台区域，该样本的 **clipped policy surrogate 对参数的梯度**为零。但这不表示它的动作概率随后固定不动：其他样本共享网络参数；entropy 项仍可能更新 actor；共享骨干时 critic loss 也会影响策略；Adam 的动量也可能延续更新。

因此要把下面三件事分清：

1. **ratio clipping** 改变 policy 目标的形状。
2. **gradient clipping** 限制一次反向传播后的梯度范数。
3. **action clipping** 把发送给环境的动作限制在合法范围内。

它们作用在不同对象上，不能互相替代。

### 4.7 PPO-Clip 保证什么，不保证什么

逐样本都有：

$$
\ell_t^{\mathrm{CLIP}}\le\rho_t\hat{A}_t
$$

这是对**未裁剪 surrogate** 的保守下界，不是对真实回报 $J$ 的通用下界。

Clip 也不是投影操作，不保证更新后所有 ratio 都在区间内；它不对所有状态、所有动作逐一施加硬约束，也不保证平均 KL 小于某个阈值。一次优化步仍可能跨过边界，其他梯度还会继续推动参数变化。所以实践中仍需监控 KL，有时配合 KL 早停。

## 5. old policy 与多 epoch：一批数据中的参照物不能移动

### 5.1 两层循环对应两种时间尺度

外层第 $k$ 轮：用 $\pi_{\theta_k}$ 收集一批数据。内层：在这批固定数据上把当前参数更新若干次。

```text
外层 k：
    用 theta_k 采 rollout
    固定：数据、old_log_prob、old_value、advantage、return_target
    当前 theta 从 theta_k 出发
        epoch 1 → 多个 minibatch 更新
        epoch 2 → 重新打乱同一批数据，继续更新
        ...
    得到 theta_{k+1}
    用新策略重新采下一批数据
```

在整段内层优化期间，ratio 的分母始终属于 $\theta_k$ 。不能在每个 epoch 或 minibatch 后用当前策略重算分母，否则参照物不断移动，PPO 就无法衡量相对**实际采样策略**已经走了多远。

例如采样时某动作的概率是 0.40，第一轮优化后变成 0.44，再下一轮变成 0.48。第二轮的 ratio 应为 $0.48/0.40=1.2$ ，而不是 $0.48/0.44\approx1.091$ 。后者只比较相邻更新，会掩盖相对采样策略的累计变化。

### 5.2 不一定要维护第二份 actor 网络

PPO-Clip 需要的是采样动作在旧策略下的 log-prob。采样时存下：

$$
\mathrm{old\_log\_prob}_t=\log\pi_{\theta_k}(a_t\mid s_t)
$$

更新时重新计算同一状态、同一动作的当前 log-prob：

$$
\rho_t=\exp\left(\log\pi_\theta(a_t\mid s_t)-\mathrm{old\_log\_prob}_t\right)
$$

这些旧 log-prob 是不带计算图的常量。普通 Clip 和采样 KL 诊断不必保留整个旧网络；若要计算完整分布的解析 KL，则还需旧分布参数或旧网络。

### 5.3 为什么能多次用数据，却仍叫 on-policy

PPO 的约定是：新采一批数据，在受控的策略变化范围内做有限次优化，然后替换数据。它不维护长期 replay buffer，也不认为久远数据仍与当前策略同样可靠。

ratio 使旧动作样本可以估计候选策略的 surrogate，Clip 减弱过大有利更新的激励；二者一起支持有限的数据复用。epoch 越多，拟合当前批次的机会越多，过拟合 advantage 噪声和偏离采样策略的机会也越多。因此“多 epoch”不是越多越好，更不是严格无偏的无限复用。

## 6. GAE、critic target、entropy：完整 loss 如何拼起来

### 6.1 先用采样时的 critic 构造固定 target

沿用奖励按动作编号的约定，暂时假设没有终止边界：

$$
\delta_t=r_t+\gamma V_{\phi_{\mathrm{old}}}(s_{t+1})-V_{\phi_{\mathrm{old}}}(s_t)
$$

$$
\hat{A}_t=\delta_t+\gamma\lambda\hat{A}_{t+1}
$$

$$
\hat{R}_t^\lambda=V_{\phi_{\mathrm{old}}}(s_t)+\hat{A}_t
$$

GAE 和 return target 在开始内层更新前算好并固定。critic 训练时改变的是预测 $V_\phi(s_t)$ ，不是每个 minibatch 都跟着重算标签。本篇的固定 target 流程与 SB3 对照；重算 advantage 的变体需要单独说明，不能悄悄混入。

**共用子过程：GAE 与固定标签。** 下列所有量使用采样时的 critic 计算，不记录梯度；采用 §7 的两个 mask，不额外向 reward 加 timeout bootstrap。

```text
子过程 GAE_固定标签(rollout, gamma, lambda):
    输入形状为 [T, N]，此时还没有展平或打乱
    old_value[t] = 采样时 V(s_t)
    next_value[t] = 采样时 V(该 transition 真正的下一状态)
    # next_value 不能误用自动 reset 后的新一局 observation
    b[t] = 0 若真正终止，否则为 1
    c[t] = 0 若真正终止或发生外部截断，否则为 1

    gae_next = 每个环境一个 0           # rollout 之外不递推，但允许 value bootstrap
    for t = T-1 ... 0:
        td = reward[t] + gamma * b[t] * next_value[t] - old_value[t]
        A_raw[t] = td + gamma * lambda * c[t] * gae_next
        gae_next = A_raw[t]

    R_target = old_value + A_raw
    返回 detach(A_raw)、detach(R_target)
```

`lambda=1` 是带尾部 bootstrap 的分段 n-step 估计；只有轨迹完整结束且没有需要 bootstrap 的尾巴时，才退化为完整 MC 回报。`lambda=0` 只保留一步 TD error。PPO 的 advantage 归一化在生成 `R_target` 之后做，不能改变这里的 critic 标签。

### 6.2 advantage 归一化只用于 policy 权重

保留原始 $\hat{A}_t$ 构造 critic target，再得到 policy 使用的标准化权重：

$$
\tilde{A}_t=\frac{\hat{A}_t-\mathrm{mean}(\hat{A})}{\mathrm{std}(\hat{A})+\eta}
$$

其中 $\eta$ 是防止除零的小常数，不是 PPO 的 $\epsilon$ 。归一化使 policy 权重的数值尺度较稳定，但减去批均值可能改变个别样本的符号，因此属于工程选择，不是有限批次下完全不改变目标的恒等变换。

可以按整个 rollout 或按 minibatch 归一化，但必须记录并保持对照一致。**本地 SB3 2.9.0 在每个 minibatch 内归一化**；内层迭代中原始 advantage 固定，重新分组后的均值、标准差可以不同。

不要把标准化后的 advantage 加到旧 value 上当 critic target：标准化已经改变了回报单位。

### 6.3 统一写成最小化 loss，避免符号错位

令 policy surrogate 中使用 $\tilde{A}$ ，最小化：

$$
\mathcal{L}_{\mathrm{policy}}=-L^{\mathrm{CLIP}}(\theta)
$$

$$
\mathcal{L}_{\mathrm{value}}=\frac{1}{B}\sum_t\left(V_\phi(s_t)-\hat{R}_t^\lambda\right)^2
$$

$$
\mathcal{L}_{\mathrm{total}}
=\mathcal{L}_{\mathrm{policy}}+c_v\mathcal{L}_{\mathrm{value}}
-c_e\hat{\mathbb{E}}_t[\mathcal{H}(\pi_\theta(\cdot\mid s_t))]
$$

policy 前面为负号，因为想最大化 surrogate；value 为正号，因为想最小化预测误差；entropy 前面为负号，因为想鼓励较大的熵。

本文 value loss 采用与本地 SB3 相同的 MSE，不额外乘 $1/2$ 。若参考实现把 MSE 定义成一半，相同的 $c_v$ 就不再代表相同权重。比较超参之前，要先比较 loss 的定义。

entropy 鼓励动作分布保留随机性，但不保证探索到有用状态。 $c_e=0$ 也不表示训练时动作确定：只要策略分布仍有非零随机性，采样仍然是随机的。

### 6.4 哪些量应该 detach

| 量 | 更新期间固定？ | 是否需要梯度？ |
|---|---|---|
| 采样的 observation、action、reward | 是 | 否 |
| old log-prob、old value | 是 | 否 |
| 原始 advantage、return target | 是 | 否 |
| 当前 log-prob | 否，每次重新前向 | 是，流向 actor |
| 当前 value | 否，每次重新前向 | 是，流向 critic |
| 当前策略的 entropy | 否，每次重新前向 | 是，流向 actor |

采样通常放在 `no_grad` 中。更新时也不要让采样动作通过 `rsample` 等路径连到当前参数；这里使用的是固定动作上的 score-function 梯度。

actor 和 critic 可以是两个独立网络，也可以共享骨干。共享时三个 loss 都可能影响骨干，需要留意 value loss 的尺度压过 policy 信号；这不是 PPO 要求必须共享网络。

### 6.5 value clipping 是可选细节，不是 PPO 定义

本地 SB3 默认 `clip_range_vf=None`，即普通 MSE。启用时先计算：

$$
V^{\mathrm{clip}}_\phi(s_t)=V_{\mathrm{old}}(s_t)
+\mathrm{clip}\left(V_\phi(s_t)-V_{\mathrm{old}}(s_t),-\epsilon_v,+\epsilon_v\right)
$$

然后用这个预测计算 MSE。其他实现可能取“原始预测误差”和“裁剪预测误差”的较大值，二者并不等价。value 的裁剪范围有回报单位，也不能直接把概率比的 $\epsilon$ 当成同一个概念。初版先保持普通 MSE，避免同时引入多种变体。

## 7. rollout 边界：terminated、truncated、采样停止不是同一件事

这是从“公式看懂”到“实现正确”最容易掉坑的部分。rollout 收集固定步数，不要求刚好包含完整 episode。

### 7.1 三种边界对应不同处理

| 边界 | 下一状态是否 bootstrap | GAE 是否跨过边界连接后续样本 |
|---|---|---|
| MDP 真正终止，`terminated=True` | 否，尾部 value 按 0 | 否 |
| 外部时间限制，`truncated=True` 且未真正终止 | 是，用截断前最终 observation 的 value | 否，不能连接 reset 后的新局 |
| rollout 满了，但 episode 仍继续 | 是，用 rollout 最后一步之后的 value | 当前批次递推到此结束，下一批继续采样 |

如果时间上限本来就是任务定义的一部分，应按有限时域 MDP 的终止处理，并让状态包含必要的剩余时间信息。不要仅凭“到时间了”就判断一定需要 bootstrap。[Gymnasium 时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

### 7.2 用两个 mask 分开思考

令 $b_t$ 表示是否允许 bootstrap， $c_t$ 表示是否允许把 GAE 递推连接到下一个存储样本。用 transition 真正的下一状态 $s_{t+1}^{*}$ ：

$$
\delta_t=r_t+\gamma b_tV_{\mathrm{old}}(s_{t+1}^{*})-V_{\mathrm{old}}(s_t)
$$

$$
\hat{A}_t=\delta_t+\gamma\lambda c_t\hat{A}_{t+1}
$$

普通连续 transition 中两者都为 1；真正终止时两者都为 0；外部截断时 $b_t=1$ 而 $c_t=0$ 。在 rollout 最后一项，令递推尾巴为 0，但正常计算 bootstrap 项。

例如外部截断前奖励为 1、最终 observation 的旧 value 为 100、 $\gamma=0.99$ 。只看这一项，bootstrap target 是 $1+0.99\times100=100$ ；若错误地当作真正终止，就会把它改成 1。重复这种错误会让 critic 在时间限制附近系统性低估。

### 7.3 自动 reset 的 observation 要确认含义

某些向量化封装会在 episode 结束时自动 reset，`step` 返回的 observation 可能已经属于下一局。此时 bootstrap 必须从封装保存的最终 observation 计算；字段名称和 autoreset 模式依 API 而异，不能机械地取返回值。

SB3 的 VecEnv 在 `info["terminal_observation"]` 保存最终 observation。对 timeout，它先把折扣后的最终 value 加进该 transition 的 reward，然后在 GAE 中切断 episode 连接。这是另一种等价实现；若手写时已显式用两个 mask，就不要再加一次 timeout reward 修正，否则会重复 bootstrap。

此外，必须先沿每个环境的时间维度计算 GAE，再展平并打乱样本；提前打乱会破坏递推关系。

## 8. 完整训练循环：每一步输入什么、哪些量会变化

### 8.1 rollout buffer 至少存哪些数据

设 $N$ 个环境各收集 $T$ 步，总样本数 $B=NT$ ：

| 字段 | 典型形状 | 用途 |
|---|---|---|
| observation | `[T, N, obs_dim]` | 更新时重新计算当前策略和 value |
| action | 离散 `[T, N]`，连续 `[T, N, act_dim]` | 评估当时实际采样动作的 log-prob |
| reward | `[T, N]` | TD error / GAE |
| old log-prob | `[T, N]` | 固定 ratio 分母 |
| old value | `[T, N]` | GAE、return target、可选 value clip |
| 终止 / 截断 / episode 边界信息 | `[T, N]` | 正确 bootstrap 与切断 GAE |
| advantage、return target | `[T, N]` | 采样完成后计算并存储 |

还需保存必要的尾部 value 和截断最终 observation，具体存法由 buffer 设计决定。神经网络输入一般是 observation；本篇为便于推导记成 $s$ ，不意味着部分可观测任务也自动满足 MDP 假设。

### 8.2 PPO-Clip：完整伪代码

这是本项目最终要手写的主体。采用普通 value MSE、可选 entropy bonus，以及与本地 SB3 相同的 minibatch advantage 归一化和可选 KL 早停。GAE 的采样边界处理见 §6.1 / §7。

```text
输入：N、T、epoch 数 K、minibatch 大小 M、gamma、lambda
      ratio 裁剪范围 epsilon、value 系数 c_v、entropy 系数 c_e
      可选 target_kl、全局梯度范数上限 max_grad_norm
初始化 actor theta、critic phi 和环境

repeat:
    清空本轮 rollout buffer

    # 采样阶段：参数固定，不记录梯度
    对每个时间步 t = 0 ... T-1：
        在当前 observation 上构造动作分布，采样 action
        保存 action、old_log_prob、old_value、observation
        与环境交互，保存 reward 和边界信息
        正确处理最终 observation 与 reset

    # 标签阶段：先沿时间维计算，再展平
    计算每个环境尾部与截断位置需要的 bootstrap value
    (A_raw, R_target) = GAE_固定标签(rollout, gamma, lambda)
    冻结 old_log_prob、old_value、A_raw、R_target
    把 [T, N, ...] 展平为 [B, ...]

    # 优化阶段：不与环境交互
    stop_update = False
    repeat K 个 epoch：
        重新打乱 B 个样本的索引
        遍历不重叠 minibatch：
            用当前网络评估存储 action 的 log-prob、value、entropy
            A_policy = (A_raw - mean(A_raw)) / (std(A_raw) + 1e-8)
            # 上式只对当前 minibatch 计算，minibatch 大小需大于 1
            log_ratio = current_log_prob - old_log_prob
            ratio = exp(log_ratio)
            unclipped = ratio * A_policy
            clipped = clip(ratio, 1-epsilon, 1+epsilon) * A_policy
            policy_loss = -mean(min(unclipped, clipped))
            value_loss = mean((current_value - R_target)^2)
            loss = policy_loss + c_v * value_loss - c_e * mean(entropy)

            approx_kl = mean(exp(log_ratio) - 1 - log_ratio)
            if target_kl 已设置 且 approx_kl > 1.5 * target_kl:
                stop_update = True
                break                  # 当前 minibatch 不执行 optimizer step

            清梯度 → 反向传播 → 全局梯度裁剪 → optimizer step
        if stop_update:
            break                      # 同时退出 epoch 循环

    记录回报、KL、clip fraction、value 诊断、实际更新次数
    用更新后的策略采下一轮；episode 未结束的环境无需强制 reset
```

`min` 和 `clip` 对每个样本逐元素计算，最后才求平均。KL 早停不回滚此前已经执行的更新；`old_log_prob` 在本轮全部 epoch 内不重算。为避免最后剩下 1 个样本，可以令 batch 大小整除 $B$ ，或像 SB3 一样对单样本 minibatch 跳过 advantage 归一化。

### 8.3 rollout、epoch、minibatch、环境步数怎么换算

例如 $N=8$ 、 $T=2048$ ，则每轮收集 $B=16384$ 个 transition。若 minibatch size 为 256，一个 epoch 有 64 次 optimizer step，10 个 epoch 共 640 次；每个样本原则上使用 10 次，但 KL 早停会减少实际使用次数。

这 640 次梯度更新没有新增环境交互。不同实现都叫“update”的量可能分别指一轮 rollout、一个 epoch 或一次 optimizer step，记录实验时必须写清。

并行环境的 `n_steps` 常指**每个环境**的步数。因此增加环境数而保持 `n_steps` 不变，会同时增大 batch；它不只是把相同实验跑得更快。`DummyVecEnv` 在一个进程中依次调用环境，是否换 `SubprocVecEnv` 应取决于环境开销和实际速度。

### 8.4 把六套伪代码放在一起看

| 算法 | actor 每批数据如何更新 | 主要新增机制 | 更新后是否接受检查 |
|---|---|---|---|
| REINFORCE | 完整 episode 的 MC policy gradient，一次更新 | reward-to-go | 无 |
| A2C | rollout 的 advantage policy gradient，一次更新 | critic 与 n-step bootstrap；可换 GAE | 无 |
| 自然策略梯度 | 沿 Fisher 校正方向做一次更新 | 分布几何与局部 KL 预算 | 本文版本直接接受候选步 |
| TRPO | 自然梯度候选步 + 回溯 | 高效求方向、KL 与 surrogate 检查 | 有，失败恢复旧 actor |
| PPO-Penalty | 同批数据多 epoch minibatch | ratio surrogate + 自适应 KL 惩罚 | 基本版本不回滚，下一轮调整 beta |
| PPO-Clip | 同批数据多 epoch minibatch | ratio surrogate + min / clip | 可选 KL 早停，不回滚已完成更新 |

GAE 是生成训练标签的子过程，可服务于不同策略更新算法。共轭梯度是求线性方程的数值子过程，也可用于自然梯度；算法差别应看整个采样、优化、接受与重新采样流程。

## 9. 从 CartPole 到 HalfCheetah：动作分布与形状检查

### 9.1 CartPole：Categorical 分布

actor 输出两个动作的 logits，用 Categorical 分布采样。保存被采样动作的 log-prob，更新时仍对这个**存储的动作**求 log-prob，不能重新采一个动作再算 ratio。

训练通常随机采样；确定性评估选择概率最大的动作。训练回报与确定性评估回报不是同一统计量，应分别记录。

### 9.2 HalfCheetah：对角高斯分布

常见策略输出均值，另有可训练的 log standard deviation。以 $D$ 维独立高斯为例：

$$
\pi_\theta(u\mid s)=\prod_{j=1}^{D}\mathcal{N}(u_j;\mu_{\theta,j}(s),\sigma_{\theta,j}^2)
$$

$$
\log\pi_\theta(u\mid s)=\sum_{j=1}^{D}\log\mathcal{N}(u_j;\mu_{\theta,j}(s),\sigma_{\theta,j}^2)
$$

一个完整动作向量对应一个 joint log-prob 和一个 ratio。漏掉对动作维度求和，会变成对每个关节分别 clip，已经改变算法。

本地 SB3 默认的普通对角高斯策略使用与状态无关的可训练 log-std 向量；这不是说所有 PPO 的标准差都必须与状态无关。

### 9.3 裁剪动作时，buffer 到底存什么

SB3 普通高斯策略先采样原始动作 $u$ ，发送给环境的是合法化后的 $a=\mathrm{clip}(u,a_{\min},a_{\max})$ ，buffer 保留原始 $u$ 与其高斯 log-prob。可把 clipping 看成从潜在动作到环境动作的固定映射。

如果把裁剪后的 $a$ 写入 buffer，却仍用未裁剪高斯的密度当作其真实执行动作密度，就混淆了两个随机变量。边界上的执行动作包含被压到边界的概率质量，不能简单套原高斯密度。

另一种方案是 tanh-squashed Gaussian；若计算变换后动作的 log-prob，需要一致处理变量变换的 Jacobian。先选择与对照实现相同的动作处理方式，再考虑替换分布。

### 9.4 防止广播产生“能跑但算错”的 loss

minibatch 大小为 $M$ 时，joint log-prob、ratio、advantage、value 和 return target 通常都应为 `[M]`。连续 action 可以是 `[M, D]`。

`[M, 1]` 的 value 减 `[M]` 的 target 可能广播成 `[M, M]`，不会必然报错。离散动作的整数类型、value 输出的 squeeze、连续 log-prob 的 sum 都应明确检查。不要用没有指定维度的 squeeze 意外删掉 batch size 为 1 时的 batch 维度。

## 10. 监控什么：loss 下降不等于策略变好

### 10.1 approximate KL：检查策略偏离采样策略多少

令：

$$
x_t=\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)
$$

旧策略样本上的 $-x_t$ 均值可估计旧到新 KL，但有限样本可能为负。本地 SB3 使用逐项非负的估计形式：

$$
\widehat{D}_{\mathrm{KL}}=\frac{1}{M}\sum_t\left[\exp(x_t)-1-x_t\right]
$$

因为固定状态下 $\mathbb{E}_{a\sim\pi_{\mathrm{old}}}[\exp(x)]=1$ ，它在支持等条件满足时与旧到新 KL 有相同的期望。这里仍是在旧样本上的估计，不是遍历整个状态空间的约束证明。

SB3 在设定 `target_kl` 时，若该 minibatch 的估计 KL 超过 `1.5 * target_kl`，就停止本轮后续优化，且不执行当前 minibatch 的 optimizer step。它不会回滚已经完成的更新；默认 `target_kl=None`，没有启用这一早停。

### 10.2 clip fraction 不等于“无梯度样本比例”

SB3 记录：

$$
\mathrm{clip\_fraction}=\frac{1}{M}\sum_t\mathbf{1}[|\rho_t-1|>\epsilon]
$$

它统计所有超出 ratio 区间的样本，不区分 advantage 符号。§4 已说明：越界但变化不利的样本仍有 policy 梯度。所以不能看到 clip fraction 为 0.3，就说 30% 样本被丢弃了。

### 10.3 critic 指标：explained variance

$$
\mathrm{EV}=1-\frac{\mathrm{Var}(\hat{R}^\lambda-V)}{\mathrm{Var}(\hat{R}^\lambda)}
$$

接近 1 表示残差方差较小；接近 0 表示在这个指标上与常数预测相近；负值表示残差方差比 target 方差更大。target 方差为 0 时该指标未定义。它不反映常量偏移造成的全部误差，也不能单独证明 critic 准确。

本地 SB3 用 buffer 中采样时保存的 old value 与 return target 计算这一日志，不是用内层最后一次更新后的 value 重新评估。对照日志时要比较同一时点。

### 10.4 现象与排查顺序

| 现象 | 优先核对的内容 |
|---|---|
| 第一轮更新前 ratio 明显不是 1 | old log-prob、存储动作、输入预处理、网络模式是否一致 |
| ratio 一直精确等于 1，训练没变化 | 分母是否错误地实时重算；是否没梯度或没执行 optimizer step |
| KL 突然升高，回报随后掉落 | 学习率、epoch 数、batch 大小、advantage 尺度、是否需要 KL 早停 |
| critic 在 episode 结束附近异常 | terminated / truncated、最终 observation、重复或缺失 bootstrap |
| critic loss 很大，actor 进展弱 | 回报尺度、value 系数、共享骨干、target 与预测形状 |
| 连续动作迅速变得近乎确定 | log-std 梯度与初始化、entropy 系数、action clipping、过大更新 |
| loss 数值正常但回报不涨 | 符号、动作与 log-prob 对齐、GAE 时间顺序、评估与归一化流程 |

这些是定位线索，不是一一对应的诊断结论。policy loss 尤其不能当作跨轮次统一的性能曲线：每轮数据、旧策略、advantage 基准都在变化；归一化后初始 policy loss 还可能接近 0，但梯度仍然非零。

## 11. SB3 对照表与手写路线

### 11.1 本地 SB3 2.9.0 的默认值

以下是构造函数默认值，不是对所有任务最优的推荐值；阶段 0 的脚本只显式设置 `MlpPolicy`、seed 等项，主要沿用这些默认值。

| 配置 | 默认值 | 控制什么 |
|---|---|---|
| learning rate | `3e-4` | 每次 optimizer step 的幅度 |
| `n_steps` | `2048` | 每个环境每轮收集的步数 |
| `batch_size` | `64` | 每次梯度更新的样本数 |
| `n_epochs` | `10` | 每批 rollout 遍历次数 |
| `gamma` | `0.99` | 折扣目标 |
| `gae_lambda` | `0.95` | advantage 估计的偏差 / 方差权衡 |
| `clip_range` | `0.2` | policy surrogate 平台开始的位置 |
| `clip_range_vf` | `None` | 默认不裁剪 value prediction |
| `normalize_advantage` | `True` | 每个 minibatch 内归一化 |
| `ent_coef` | `0.0` | 显式 entropy bonus 权重 |
| `vf_coef` | `0.5` | 普通 MSE 的权重 |
| `max_grad_norm` | `0.5` | 全部模型参数的全局梯度范数上限 |
| `target_kl` | `None` | 默认没有 KL 早停 |

其他对齐项包括：MLP 默认 actor / critic 各两层 64 单元、Tanh、正交初始化，以及默认 Adam 的 `eps=1e-5`。它们是实现配置，不是 PPO-Clip 数学定义的一部分。

`clip_range=0.2` 是常见起点，不是所有任务通用的定律。同理，没有“KL 大于 0.02 一律错误”的标准。学习率、epoch、minibatch、动作维数和任务都影响实际策略变化，应一起看回报与诊断指标。

### 11.2 归一化不要只记成一个开关

advantage 归一化、observation 归一化、reward / return 归一化解决的是不同尺度问题。普通 SB3 PPO **不会自动附加 observation / reward 归一化**；需要额外 wrapper，例如 `VecNormalize`。

若使用 running statistics，要保存到 checkpoint，评估时加载同一份并冻结更新；用于报告的回报应明确是否为原始 reward 尺度。更新阶段对存储 observation 的处理应与采样时一致，不应让归一化统计或 dropout 等因素破坏初始 ratio 检查。

### 11.3 读源码时按调用链走

以下路径相对于本地 `.venv/Lib/site-packages/stable_baselines3/`：

| 文件 / 方法 | 读它要回答的问题 |
|---|---|
| `common/on_policy_algorithm.py` → `collect_rollouts` | 怎么采样、保存旧量、处理 timeout、计算尾部 value |
| `common/buffers.py` → `RolloutBuffer.compute_returns_and_advantage` | GAE 在哪里递推，episode 边界如何切断 |
| `common/buffers.py` → `RolloutBuffer.get` | 什么时候展平、如何打乱与遍历 minibatch |
| `ppo/ppo.py` → `PPO.train` | 当前 log-prob、ratio、loss、KL 早停、梯度更新顺序 |
| `common/policies.py`、`common/distributions.py` | 网络输出如何变成分布，联合动作 log-prob 怎么算 |

完整 PPO 的正确性分布在这条调用链里，不能只对照 `PPO.train` 的十几行 loss。

### 11.4 按学习计划逐步加机制

1. **REINFORCE**：先让采样动作、log-prob、reward-to-go 和 policy loss 符号正确。
2. **baseline / actor-critic**：加入独立 critic，核对 policy 权重与 value target 的 detach。
3. **GAE**：处理固定长度 rollout、真正终止、外部截断、bootstrap；仍先保持简单更新。
4. **PPO**：保存 old log-prob，加 ratio、min / clip、多 epoch minibatch，再监控 KL。
5. **连续控制与对齐**：联合 log-prob、动作范围、归一化、网络初始化、固定预算对照。

每次加入机制后，应能解释它改变了哪个估计量、目标或数据流。不要一开始把全部工程开关一起打开，随后只能凭总回报猜哪里出了问题。

最小检查点：更新前 ratio 约为 1；四个数值例子的分支与梯度方向正确；真正终止不 bootstrap、外部截断正确 bootstrap；loss 向量形状正确；旧量和 target 没有梯度；模型确实在 optimizer step 后发生变化。

阶段验收仍按仓库计划：第 1 周把自己的 actor-critic 做到 CartPole 目标，第 2 周补完整 PPO，并在 HalfCheetah 1M 环境步预算下与 SB3 比较。比较时保存 seed、环境与依赖版本、网络、预处理、配置、对应 commit、原始回报曲线和评估方式；环境步预算相同不表示梯度计算预算相同。

## 12. 常见疑问与参考资料

### 为什么不用 log-prob 本身，而要用 ratio

普通 PG 的 log-prob loss 在旧参数处提供正确的一阶方向；ratio surrogate 则明确比较候选策略与固定采样策略，在同一批数据上多次更新时保留这个参照。两者在起点的梯度一致，离开起点后目标形状不同。

### Clip 是不是直接限制每次概率只改变 20%

不是。 $\epsilon=0.2$ 定义的是该批采样动作的 surrogate 截断阈值，参照对象是本轮采样策略；既不是每个 optimizer step 的硬限制，也不是对整个分布的 20% 距离约束。

### advantage 来自有误差的 critic，PPO 会自动修正吗

不会。Clip 控制更新激励，不辨别 advantage 的真假。若 critic / GAE / reward 有系统性错误，PPO 仍可能稳定地沿错误信号优化；多 epoch 还会反复拟合这些错误。

### PPO 是否一定比 REINFORCE 或 SAC 样本效率高

不能这样概括。PPO 的有限数据复用通常比每批只做一次 PG 更新更充分，但效果取决于任务和实现；它没有长期 replay 的复用方式，也不保证超过 off-policy 算法。选它做本项目主线，是因为训练流程清楚、便于从 actor-critic 逐步扩展并与 SB3 对照。

### 本篇与前两篇怎样衔接

- [01：策略梯度](01_policy_gradient.md) 给出更新方向及 baseline 的作用。
- [02：GAE](02_gae.md) 构造 advantage 与 critic target。
- 本篇解释固定旧策略参照下的有限多次更新，以及实现时必须保住的数据语义。

### 参考资料与适用范围

- [Schulman et al., 2017：Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：§3 Clip、§4 KL penalty、§5 训练框架。
- [Schulman et al., 2015：Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)：状态分布近似、性能下界、KL 约束与 Appendix C 的数值求解。
- [OpenAI Spinning Up：PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：Clip 的直观解释与 KL 早停；具体默认值与 SB3 不同。
- [Huang et al., 2022：The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)：复现与实现差异清单。它包含 Atari、连续控制等不同场景，不应把 37 项都当成 CartPole 初版的必需配置；其历史实现对 timeout 的处理也应与当前 Gymnasium / SB3 区分。
- [Gymnasium：Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：termination、truncation 与 bootstrap 语义。
- SB3 本地安装源码：`ppo/ppo.py`、`common/buffers.py`、`common/on_policy_algorithm.py`、`common/policies.py`、`common/distributions.py`。本篇工程细节已对照安装版本 2.9.0 核查。
- [知乎：PPO 算法理解——从策略梯度开始](https://zhuanlan.zhihu.com/p/614115887)：旧版笔记的演进叙事参考；本次推导与实现核查以原论文和源码为依据。
