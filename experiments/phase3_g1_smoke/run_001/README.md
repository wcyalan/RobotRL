# G1：官方 PPO 短训练验证

当前任务：`fee01531-a5ca-45bc-b2dd-810b7add6167`，2026-10-19提交。单张 RTX 6000 Ada；训练和独立加载均已通过。

前置条件：G1 4环境1100步预检通过、自动reset已观察、资产哈希核对通过。使用同一固定镜像和官方 `Isaac-Velocity-Flat-G1` / `newton_mjwarp`。

训练覆盖项：128环境、20轮、seed=0、本地资产位置；其余采用官方配置。每环境每轮24步，共61,440环境步。这个预算仅用于链路验证。

完成顺序：官方训练入口保存checkpoint → 检查首末checkpoint的actor和critic参数确有变化及20轮loss均有限 → 独立进程重新构建环境并加载最终checkpoint → 4环境首回合评估，检查有限动作/奖励、终止/reset并保存带回合边界的姿态轨迹。

代码：[训练](../../../phase3/training/train_smoke.sh)、[训练结果核验](../../../phase3/training/verify_training.py)、[独立加载评估](../../../phase3/evaluation/evaluate_g1.py)。短训练回报低或跌倒不等于保存加载失败；正式基线已在本次验证通过后启动，见 [seed=0](../../phase3_g1_baseline/run_001/README.md)。

## 已验证结果

- 共61,440环境步，20轮全部loss记录有限；actor有9个参数张量、critic有8个参数张量发生变化。
- 最终 `model_19.pt` 的SHA-256：`14cb9ae95b4bb7020a4627f9e4f5843c00a33dc13204309c46466427cf5ef374`。
- 独立进程加载后4个首回合均完整记录并提前跌倒，长度48–60策略步（0.96–1.20秒），平均回报约-3.904；有限动作/奖励检查和重置通过，尚未学会行走。
- 启停评估器验证任务：`b146c860-e424-4a24-a003-0de734f5058c`，使用该短训练模型和额外seed=19000；只核对4次指令切换与首回合统计，不计入正式行为结果。

证据：[训练参数与loss核验](training_check.json)、[独立加载](reload/summary.json)。正式协议在 [protocol.json](../../../phase3/evaluation/protocol.json)，训练前固定，不根据训练结果更改。

启停评估器已通过：任务 `e0c1fc05-b84d-48f2-adcd-ab1748ca590c` 成功退出，0/5/12/22秒四次指令切换完整。最初试跑的零指令整数/浮点比较错误已修复，未改模型和协议。该短模型行为0/4通过，仅用于验证评估代码。
