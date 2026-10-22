# 实验文件与复现数据

更新于2026-10-22。本轮按现有结果收尾，不追加训练、不批量同步服务器。实验结果见[实验索引](README.md)。

## 代码与材料位置

|位置|内容与边界|
|---|---|
|当前桌面 `C:/Users/Administrator/Desktop/RobotRL/`|Git克隆的代码和紧凑报告，加本次审计、演示与最小渲染素材；不是完整实验镜像|
|服务器 `/labdata2/congyuwang/RobotRL_kitless/`|此前核查的完整实验根目录；实验在 `experiments/`，较新的迁移源码在 `code/phase3/sim2sim/`；容器内实验根目录为 `/robotrl/experiments/`|
|旧记录中的 `C:/Users/wcyal/Desktop/RobotRL/`|历史工作机路径，不能据此推断当前电脑已经有对应权重或数据|
|Git记录|维护代码、笔记、实验README、配置与紧凑指标；本轮新增报告和脚本随仓库版本维护，原始大文件另行保存|

Go2维护代码在 `phase2_isaaclab/`，G1代码在 `phase3/`。本地sim2sim早期入口不能替代服务器run_003的完整分组评估入口。历史文档中的“本地保留”描述的是当时工作机，当前机器是否存在文件需单独检查。

## 当前可用范围

|用途|当前桌面状态|所需入口|
|---|---|---|
|阅读结论、配置与紧凑评估证据|可用|[实验索引](README.md)|
|观看130秒G1双模型回放|已生成并逐帧校验|[演示目录](phase3_g1_sim2sim/demo_001/README.md)|
|重新渲染同一演示|最小素材已就绪；需安装渲染依赖|[复现命令](phase3_g1_sim2sim/demo_001/README.md)|
|复算完整768局迁移保存结果|完整逐步数据未同步|[审计脚本与报告](phase3_g1_sim2sim/audit_001/README.md)|
|加载G1导出策略或重跑闭环|本机两个 `export/policy.pt` 均未下载|[模型记录](phase3_g1_deployment/README.md)、[推理接口](../phase3/deployment/README.md)|
|完整Go2部署复现|本机无 `reproduction_bundle.zip`|[部署包记录](phase2_go2_deployment/run_001/README.md)|

最小演示素材在 `phase3_g1_sim2sim/demo_001/inputs/`，只含模型、显示网格、两个训练seed各六条指定轨迹、协议和哈希清单。没有训练权重，也不能代替完整实验数据。`verification.json`记录视频逐帧解码结果和SHA-256。

## 完整实验复现

Go2历史复现包为 `phase2_go2_deployment/run_001/reproduction_bundle.zip`，包含两个父检查点、导出策略、配置、运行源码快照、动作比较、闭环结果、接口说明及清单。它记录的是10/12行为条件通过的结果，不是所有条件通过的模型。

G1采用原2%零指令配置的两个 `model_1499.pt`；导出策略分别对应 `phase3_g1_deployment/run_001/export/policy.pt` 与 `run_002/export/policy.pt`。完整推理与评估还需对应配置、源码、依赖及机器人资产。应按保存清单核对文件和版本，不能仅凭模型文件名判断一致。

具备服务器权限时，可按目标用途取得所需材料；本次没有新增下载或同步，也不以重新训练作为查看已有成果的前提。当前没有公共权重下载链接。服务器文件位置来自此前核查，本次文档整理没有再次检查服务器存储或启动远程任务。

## 保留与缺失的说明

模型、视频、轨迹、日志和大体积源码快照通常被Git忽略，仓库克隆并不包含它们。当前新增的演示视频及 `demo_*/inputs/` 同样被忽略；GitHub阅读者可查看紧凑报告和预览图，不能假定本地视频链接在网页端可播放。

G1的20%零指令对照此前已按要求清除模型、轨迹和训练日志，保留配置、源码和紧凑结果，不能再加载该组模型或重放该组轨迹；见[负结果记录](https://github.com/wcyalan/RobotRL/blob/4048eebbc349b73ddd7d1d9b3a115cfa29495f92/experiments/phase3_g1_standing/README.md)。原始基线未被该次清理替换。其余历史文件以各实验记录和存储实况为准，不统一声称所有原件仍在当前电脑。

## GitHub 精简规则

当前发布保留 8 个核心实验目录，以及必要的负结果摘要。历史实验的现有本地文件保持原路径，通过 `git rm --cached` 退出跟踪并加入 `.gitignore`；此次不删除本地数据、不更改阈值或模型。移出跟踪前后的文件 SHA-256 已逐一核对。

核心目录保留配置、紧凑评估证据与必要曲线/预览；重复源码快照、原始日志、逐帧拆图和临时完成标记留本地。HalfCheetah 的 SB3 配置与独立评估以原样副本保留在核心连续 PPO 的 `comparisons/` 下。

当前精简克隆不包含历史分析脚本所需的全部输入，不能保证直接重算所有旧实验。需先按原路径补齐对应材料；维护代码没有移除。目录映射与固定历史版本见[历史说明](HISTORY.md)。此次不改写 Git 历史，旧提交仍可访问。

整理前的根目录和实验索引另存于本机 `experiments/local_archive/`，该目录不会发布。
