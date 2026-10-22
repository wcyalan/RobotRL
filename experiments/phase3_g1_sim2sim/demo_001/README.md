# G1 多指令演示与复现

本演示回放服务器 `phase3_g1_sim2sim/run_003` 已保存的原生MuJoCo评估状态。固定选择训练seed=0、1，评估seed=20000，lane=0，左右并排展示。没有挑选最佳回合，没有新增训练、策略推理或动力学步进。

## 看什么

|视频时间|内容|对应评估|
|---|---|---|
|0–20秒|站立|两模型各32/32通过窗口均值门槛|
|20–40秒|0.3 m/s前进|两模型各32/32|
|40–60秒|0.6 m/s前进|两模型各32/32|
|60–80秒|左转0.5 rad/s|两模型各32/32|
|80–100秒|右转−0.5 rad/s|两模型各32/32|
|100–130秒|站立→前进→停止→再启动|两模型各0/32通过连续稳定停车验收|

章节对应六个独立首回合，每章开始会重置，不是一条连续切换六种条件的闭环轨迹。最后30秒本身是连续启停回合：章内0–5秒站立、5–12秒前进、12–22秒停止、22–30秒重启。

画面显示保存的指令、平移速度误差和偏航速度误差，以及该组通过数和当前代表回合的结果。站立的窗口均值通过不代表完全静止。停车平均误差虽满足阈值，但没有达到连续1秒低速窗口，仍标为FAIL。机身坐标前进不等于世界坐标直线导航。

## 文件

- `g1_multicommand_replay.mp4`：1280×720、25 fps、130秒，3250帧；没有加速、插帧或循环补齐。
- `preview.png`：六个章节的总览。前五章选10秒，启停章选18秒（停车段）。
- `metadata.json`：渲染依赖版本、时间轴、来源和输出信息。
- `verification.json`：逐帧解码后的帧数、分辨率、时长与视频SHA-256。
- `inputs/manifest.json`：提取素材及原始输入的SHA-256；不包含凭据。
- `inputs/`：冻结模型和显示网格、12条指定代表轨迹、选择规则及协议。没有策略权重、训练checkpoint或整套实验数据。

完整实验复核见相邻的 `audit_001/README.md`。本演示只是视觉表达，不重测接触力，不证明两后端动力学完全等价，也不代表真机部署就绪。

棋盘地面材质、跟随相机和文字叠加仅用于显示；冻结模型文件与物理参数未改动。

## 已有素材的本地复现

在仓库根目录、具有Python 3.13的独立环境中安装渲染依赖：

```powershell
python -m pip install -r phase3/visualization/requirements_saved_demo.txt
python phase3/visualization/render_saved_demo.py --inputs experiments/phase3_g1_sim2sim/demo_001/inputs --output experiments/phase3_g1_sim2sim/demo_002
python phase3/visualization/verify_saved_demo.py --output experiments/phase3_g1_sim2sim/demo_002
```

已有输出目录会拒绝覆盖。先加 `--preview-only` 并使用另一个输出目录可检查六章画面。渲染需可用的OpenGL上下文；没有窗口不等于不需要图形驱动。脚本仅调用 `mj_forward` 设置保存姿态，没有调用 `mj_step`，不需要Isaac Lab、Newton、USD库或PyTorch。

本次在Windows已有Python 3.13运行时下，使用工作区独立安装的包运行；没有更改系统Python或LibreOffice自身的包目录。精确安装版本见requirements文件和metadata。

## 从原始评估提取最小素材

在已经拥有完整run_003的机器上执行，源目录保持只读：

```bash
python prepare_saved_demo.py --run /labdata2/congyuwang/RobotRL_kitless/experiments/phase3_g1_sim2sim/run_003 --output /path/to/new/demo_inputs
```

`prepare_saved_demo.py` 位于仓库 `phase3/visualization/`，需要NumPy；先验证冻结输入，再只提取所选lane和显示资产。必须使用新目录。本次通过SSH在内存中打包等价素材后下载，未向服务器写入演示文件或安装渲染依赖。

时间轴检查：

```powershell
python -m unittest discover -s phase3/visualization -p test_render_saved_demo.py -v
```

只下载本目录素材可以重放演示；不能据此重新运行完整768局评估或训练。后两项需要服务器的模型、策略和较新sim2sim代码。
