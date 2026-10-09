# 强化学习方法框架与策略梯度

> 参考：[知乎：PPO 算法理解——从策略梯度开始](https://zhuanlan.zhihu.com/p/614115887) §1；[Berkeley CS285 Lecture 5: Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)（Levine）；S&B 第 13 章。

## 1. RL 方法地图：策略梯度在哪里

按"学什么"分三大类（都不依赖环境模型，属 model-free；先学模型再规划的是 model-based，另一条线）：

| | 学什么 | 代表算法 | 策略形态 | 特点 |
|---|---|---|---|---|
| **基于值函数** | $Q(s,a)$ 或 $V(s)$，贪心导出动作 | Q-Learning, SARSA, DQN | 隐式、确定性 | 样本效率高；难处理连续动作；策略不随机 |
| **基于策略** | 直接参数化 $\pi_\theta(a\mid s)$ | REINFORCE | 显式、随机 | 天然支持随机策略与连续动作；方差大、样本效率低 |
| **Actor-Critic** | 策略头 + 价值头一起学 | A2C, PPO, SAC, DDPG | 显式（随机或确定性） | 两者折中，现代主流 |

基于值函数的路线靠递归求解 Bellman 方程（A2 作业写的就是这个）；基于策略的路线**跳过价值函数、直接对期望回报做梯度上升**——这是策略梯度方法的出发点。

## 2. 目标函数与策略梯度定理

轨迹 $\tau = (s_0, a_0, \dots, s_T)$，目标：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \right] = \sum_\tau P(\tau;\theta)\, R(\tau)
$$

**策略梯度定理**（Sutton et al. 1999，只陈述不证明）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \left( \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) \cdot R(\tau) \right]
$$

推导只用了两步思想：

1. **log-derivative trick**：$\nabla P = P \cdot \nabla \log P$，把轨迹概率的梯度变成它自己分布下的期望；
2. **环境项消失**：$\log P(\tau;\theta)$ 中初始分布和转移概率 $p(s' \mid s, a)$ 与 $\theta$ 无关，求导后只剩 $\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$。

推论：**不需要知道环境模型**（免模型的来源），只需能采样。

## 3. 换个角度：PG 就是"加权极大似然"（CS285 的核心类比）

把策略梯度的 loss 和监督学习的极大似然放在一起看：

| | 权重 | 形式 |
|---|---|---|
| 监督学习极大似然 | 全 1（每个标签同等对待） | $L = -\mathrm{mean}\, \log p(y \mid x)$ |
| 策略梯度 | **回报**（好轨迹的动权重高，差的为负） | $L = -\mathrm{mean}\left[ \log \pi(a \mid s) \cdot R \right]$ |

CS285 的总结：PG **只是把 trial-and-error（试错学习）形式化**——"good stuff is made more likely, bad stuff is made less likely"。第 2 周手写时会发现代码就是极大似然训练改一下权重（slides 里有并排伪代码对照）。

## 4. 高方差问题的本质：credit assignment 的噪声

CS285 的国际象棋例子（终局奖励 $\pm 1$）：你想要的是"好棋着法加正权重、坏棋着法加负权重"，实际拿到的是——

- 运气好的开局拿到正权重，运气差的开局拿到负权重；
- 前面走了好棋、后面自己失误输了 → 好棋被记负权重；
- 走了坏棋但对手后来失误让你赢了 → 坏棋被记正权重。

这些问题在**期望意义上会平均掉（无偏），但收敛到期望需要非常大的样本量**——这就是"高方差"的确切含义，也是整条策略梯度方法线的头号敌人。

## 5. 方差缩减三板斧（本篇最重要的组织框架）

**① Causality / reward-to-go**：$t$ 时刻的动作影响不了 $t$ 之前的奖励，所以 $R(\tau)$ 换成

$$
G_t = r_{t+1} + \gamma r_{t+2} + \cdots \qquad \text{(reward to go)}
$$

直接删掉与动作无关的噪声项，无偏。

**② Baseline**：减去一个与动作无关的量 $b(s)$，期望不变（无偏）但方差下降：

$$
\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi(a_t \mid s_t) \cdot \big( G_t - b(s_t) \big) \right]
$$

$b$ 取平均回报已经"pretty good"；取 $V(s)$ 就得到 **advantage** $A = Q - V$——"这个动作比该状态的平均水平好多少"。

**③ 更好的 return 估计**：$G_t$ 是蒙特卡洛，换成 critic 的 bootstrap 估计（TD(λ)/GAE）——这一步引入偏差换取方差大幅下降，详见 [02_gae.md](02_gae.md)。

三板斧打完，剩下的一个"步长"问题（更新多大算安全）交给 [03_ppo.md](03_ppo.md)。

## 6. REINFORCE 与策略参数化

REINFORCE = 定理的直接落地（无三板斧版）：

```
1. 用 π_θ 跑完一整个 episode
2. loss = −Σ_t log π_θ(a_t|s_t) · G_t     ← 交给自动微分
3. 更新 θ，丢弃数据
```

| 动作空间 | 策略 | 说明 |
|---|---|---|
| 离散 | **Softmax**：$\pi(a \mid s) = \dfrac{e^{\phi(s,a)^\top \theta}}{\sum_{a'} e^{\phi(s,a')^\top \theta}}$ | log-grad = 观测特征 − 平均特征；好回报 + 偏离均值远 → 强更新 |
| 连续 | **高斯**：$\mu_\theta(s)$ 由网络给出，$\sigma$ 可学习 | $\nabla_\theta \log \pi(a \mid s) \propto \dfrac{(a - \mu_\theta(s))\, \phi(s)}{\sigma^2}$；好回报时远离均值的动作触发强更新 |

实现上不手推梯度：$L = -\log \pi_\theta(a \mid s) \cdot A_t$ 让框架自动求导——和极大似然训练只差一个加权。

## 7. 两个补充性质（CS285 slides 的边角要点）

- **部分可观测也能用**：PG 的推导从头到尾不需要 Markov 假设（不碰转移概率），观测换成任意 $o = \phi(s)$ 照样成立——这是它对 value-based 方法的一个实际优势（value-based 的收敛性论证依赖 Bellman/Markov）。
- **实践直觉**：梯度噪声极大 → 用**远大于监督学习的 batch**；学习率极难手调，Adam 勉强可用；专门的步长自适应方法（TRPO/PPO）正是为此而生 → [03_ppo.md](03_ppo.md)。

## 8. On-policy：贯穿始终的性质

策略梯度定理的期望取在 $\pi_\theta$ **自己诱导的分布**上——数据必须由当前策略产生，策略一更新数据即过期。后果：

- 样本利用率低（REINFORCE 每条轨迹只能更新一次）；
- 复用旧数据必须做修正 → 重要性采样比 $\pi_\theta / \pi_{\mathrm{old}}$，这正是 PPO 目标函数里那个 ratio 的来源。
