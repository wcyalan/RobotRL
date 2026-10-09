# λ-Return 与 GAE(λ)，及其在 PPO 中的作用

## 1. 动机：更新目标的两极

```
TD(0):  G_t^{(1)} = r_{t+1} + γ V(s_{t+1})        ← 低方差、高偏差（V 的误差进 target）
MC:     G_t       = r_{t+1} + γ r_{t+2} + ...      ← 无偏、高方差，须等 episode 结束
n-step: G_t^{(n)} = r_{t+1} + ... + γ^{n−1} r_{t+n} + γ^n V(s_{t+n})
```

λ-return 的回答：不选 n，全都要，几何加权混合。

## 2. λ-Return 定义

```
G_t^λ = (1−λ) Σ_{n=1}^∞ λ^{n−1} · G_t^{(n)}
```

- 权重和 = 1（凸组合）；λ=0.9 时权重 0.1, 0.09, 0.081, …（以 one-step 为主）
- **λ=0 → TD(0)；λ=1 → MC**，λ 是两者间的连续插值
- episodic 截断修正（S&B 式 12.8）：终止前正常加权，剩余概率质量 λ^{T−t−1} 给完整回报 G_t

**Forward view**（定义本身）：需要未来所有 n-step return → 必须 offline。
**Backward view**（eligibility traces）：在线等价实现，每步 O(d)：

```
δ_t = r_{t+1} + γ V(s_{t+1}) − V(s_t)
z_t = γλ·z_{t−1} + ∇_w V(s_t)          ← 最近访问状态的"余温"
w  ← w + α δ_t z_t
```

直觉：TD error 按余温把 credit 分给最近一串状态，恰好等价于"当前 δ 是对 k 步前状态的 (γλ)^k 折扣修正"。tabular + offline 下两条路线逐项系数相同；**在线更新时需 dutch trace / true online TD(λ)（van Hasselt & Sutton 2015）才严格等价**。

## 3. GAE：advantage 上的 λ-return

用学到的 V 定义 TD error：

```
δ_t = r_{t+1} + γ V(s_{t+1}) − V(s_t)
```

n-step advantage 有 telescoping 恒等式（相邻 δ 相消）：

```
Σ_{l=0}^{n−1} γ^l δ_{t+l} = G_t^{(n)} − V(s_t)
```

对 n 几何加权后级数闭式解：

```
Â_t^GAE(λ) = Σ_{l≥0} (γλ)^l δ_{t+l}
```

端点：λ=0 → δ_t（一步 bootstrap，方差最小、偏差依赖 V）；λ=1 → G_t − V(s_t)（REINFORCE+baseline）。

## 4. GAE 在 PPO 中的作用

PPO 目标：

```
L^CLIP(θ) = E[ min( r_t(θ)·Â_t,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ],   r_t(θ) = π_θ/π_old
```

**分工**：clip 决定"怎么更新"（限制单步策略改动），GAE 决定"用什么信号更新"（Â 的估计器）。两者正交。

1. **尺度与零中心化**：advantage 大致以 0 为中心、幅度可控 → ε=0.2 这类全局超参对不同任务通用。若用原始 G_t，信号尺度随任务漂移几个数量级，clip 失去意义。配套操作：**批内 advantage 归一化** (Â−μ)/σ —— PPO 调参优先级最高的细节。
2. **样本复用放大估计误差代价**：PPO 一批数据多 epoch 复用，Â 方差大时信号正负乱跳，clip 可能把更新限制在错误方向。GAE 先压稳信号，clip 才有干净梯度可管。
3. **与 value loss 一体**：critic 的回归目标就是 G_t^λ = V(s_t) + Â^GAE——同一套 δ 和 (γλ) 权重，policy 头当梯度权重、value 头当回归标签。代码里就是一段从后往前的 (γλ) 累乘循环。

## 5. γ 与 λ 的分工

对**真实** V^π，任何 λ 的 GAE 期望都等于 A^π：

- **λ 只改估计量的偏差-方差结构，不改优化目标** → 纯估计器调参，放心动
- **γ 直接改变目标本身**（折扣回报定义）→ 改问题，慎重

默认起点：γ=0.99, λ=0.95（绝大多数连续控制任务）。

## 6. 诊断口诀

- 曲线方差大、学一阵崩 → 怀疑 λ 偏大 / advantage 没归一化
- 学得慢、对 critic 质量敏感 → 怀疑 λ 偏小（bootstrap 短视）
- GAE 实现只需存一串 δ 再从后往前累乘 γλ（注意 done 掩码与 bootstrap 尾部）
