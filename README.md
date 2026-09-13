# RTL RL GitOps

三节点 K3s 上的 RTL 模型训练、推理和调度基础设施。Argo CD 将受审查的 Git 配置同步到集群；Tekton 执行实验流程；现有 Operator 负责 pro6000D 昼夜推理/训练切换。

## 分工

| 层 | 职责 |
|---|---|
| Argo CD | 静态 Deployment、DaemonSet、CronJob、Service、RBAC、CRD、Pipeline/Task 与调度策略的持续部署 |
| Brain | 选择数据、提交有界实验、只读观测集群 |
| Tekton | 基线 → 训练 → 候选评测 → 独立比较与记录 |
| L20 worker / Executor | GPU 容器、阶段租约、检查点与独立证据校验 |
| 昼夜 Operator | pro6000D 在22:30业务空闲后训练，07:30恢复、08:00前就绪 |

`clusters/lab/` 是实际部署配置，`bootstrap/` 是 Argo CD 项目/根应用入口，`src/` 保存自建组件源码，`host/` 保存主机部署方法。内部地址属于实验室环境配置，可按环境替换。模型底座固定为官方 Qwen/Qwen3.8-27B，模型权重、数据集、评测结果不随仓库分发。

## 凭据边界

本仓库不包含任何 Kubernetes Secret、私钥、密码、模型服务令牌、kubeconfig 或生产运行日志。清单只引用已有 Secret。凭据通过独立渠道预置和轮换，不写入 Git，不把 base64 当作加密。参见 [凭据与恢复](docs/SECRETS.md)。

每个待发布文件必须进入 `public-files.json` 白名单，通过凭据扫描、全历史扫描和清单校验。CI 使用只读 GitHub token，无集群凭据；Argo CD 通过 HTTPS 拉取公开仓库，不需要 GitHub 写入凭据。

## 变更和恢复

修改清单 → PR → 校验/审查 → 合并 main → Argo CD 自动同步。启用 selfHeal；首轮自动 prune 保持关闭，Namespace/PVC/CRD额外禁止 prune/delete。删除资源需要显式运维操作，避免误删持久数据。不要把动态 PipelineRun、Job、Lease、路由状态或检查点提交为期望状态。

回退使用 Git revert，再由 Argo 同步。镜像版本变化是独立发布；当前已有自建镜像使用节点本地导入与 `imagePullPolicy: Never`，在更新镜像引用前必须确保目标节点有镜像。首次接管保留现有 Pod 模板，避免隐式滚动重启。

Argo 管理 Kubernetes 配置；主机 K3s 安装、GPU驱动、systemd执行器、模型/数据库恢复仍通过 `host/` 的受控入口处理，并非已经由 Argo 自动管理。当前控制面、数据库与入口仍有单点，GitOps 不等于高可用或备份。
