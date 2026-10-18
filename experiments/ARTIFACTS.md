# 实验文件与复现数据

GitHub保存：维护中的phase2_isaaclab代码、notes说明、实验README、acceptance/comparison/diagnosis等紧凑JSON、模型哈希及配置。原始轨迹JSON（部分单文件约38MB）、日志、模型、视频、运行时第三方源码快照与ZIP不新增到Git。

完整原件的位置：

- 本地：`C:/Users/wcyal/Desktop/RobotRL/experiments/`。
- 服务器：`/labdata2/congyuwang/RobotRL_kitless/experiments/`，容器内为`/robotrl/experiments/`。
- 对应实验目录结构保持一致；两端历史维护ZIP各有自己的manifest，不能混用。

最新部署复现包：`phase2_go2_deployment/run_001/reproduction_bundle.zip`，包含两个父检查点、导出策略、配置、运行源码快照、官方动作比较、闭环结果、接口说明及bundle_manifest.json。下载后按manifest逐文件校验SHA256。该包记录的是10/12行为条件通过的真实结果，并非全部通过的发布模型。

有服务器权限时，在仓库根目录执行：

```powershell
scp login.lins.lab:/labdata2/congyuwang/RobotRL_kitless/experiments/phase2_go2_deployment/run_001/reproduction_bundle.zip .
```

没有服务器权限的读者可阅读GitHub摘要和源码，但不能仅凭Git克隆执行依赖原始轨迹的汇总或加载被忽略的权重；需要取得复现包或重新训练。当前没有公共模型下载链接。不能把缺文件当成训练或验收已失败。

视频引用用于本地工作区阅读，GitHub上未上传的视频链接可能无法访问。主README和实验摘要中的结论以已提交的验收报告为依据。原始文件未删除；失败案例也不因未上传而从分析中排除。
