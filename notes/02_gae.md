# λ-Return 与 GAE(λ)：advantage 的估计器

> 详细推导见 S&B 第 12 章；这里只保留 PPO 视角需要的东西。

## 1. 问题：critic 的 target 怎么取

更新信号的两极（n-step return 在中间插值）。

TD(0)——一步 bootstrap，低方差、高偏差（ $V$ 的误差进 target）：

$$
r_{t+1} + \gamma V(s_{t+1})
$$

MC——完整蒙特卡洛回报，无偏、高方差，须等 episode 结束：

$$
r_{t+1} + \gamma r_{t+2} + \cdots
$$

## 2. λ-Return：几何加权混合所有 n

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \cdot G_t^{(n)}
$$

$\lambda = 0$ → TD(0)， $\lambda = 1$ → MC； $\lambda$ 是偏差-方差之间的连续旋钮。在线实现等价物是 eligibility traces（backward view），但 PPO 是批量算法，直接按定义算即可。

## 3. GAE：把同一套操作用到 advantage 上

用学到的 $V$ 定义 TD error：

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

n-step advantage 相邻 $\delta$ telescoping 相消，几何加权后有闭式解：

$$
\hat{A}_t^{\mathrm{GAE}(\lambda)} = \sum_{l \geq 0} (\gamma \lambda)^l \, \delta_{t+l}
$$

实现 = 存一串 $\delta$ 从后往前累乘 $\gamma\lambda$ （注意 done 掩码和 bootstrap 尾巴），十行以内。

## 4. 在 PPO 中的角色

- **policy 头**： $\hat{A}$ 进 clipped 目标当加权信号（零中心、尺度可控， $\epsilon = 0.2$ 这类全局超参才对不同任务通用）；
- **value 头**：critic 的回归目标就是 $G_t^\lambda = V(s_t) + \hat{A}^{\mathrm{GAE}}$ ——同一套 $\delta$ ，一体两面；
- **$\gamma$ vs $\lambda$ 分工**： $\lambda$ 只改估计量方差/偏差、不改优化目标（对真 $V^\pi$ 任何 $\lambda$ 期望都是 $A^\pi$ ），放心调； $\gamma$ 直接改目标本身，慎重调。

## 5. 默认值与诊断

- 起点： $\gamma = 0.99,\ \lambda = 0.95$ （绝大多数连续控制任务）。
- 曲线方差大、学一阵崩 → $\lambda$ 偏大 / advantage 没归一化；学得慢、对 critic 敏感 → $\lambda$ 偏小。
