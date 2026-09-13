# GitOps 操作与所有权

## 首次接管

先恢复主机、持久数据、外部Secret和本地镜像；安装 `clusters/lab/argocd/resources.json`。再从 bootstrap 清单只应用 AppProject/根 Application，根应用会创建子应用。首次子应用保持手动，服务端 dry-run 和 Pod template 比对通过后逐组同步；每组确认健康再开启自动同步。

根应用为 `rtl-lab`，子应用为 `rtl-argocd`、`rtl-tekton-platform`、`rtl-pipelines`、`rtl-brain`、`rtl-system`、`rtl-egress`、`rtl-new-api`。Argo 使用 annotation 跟踪，不改应用已有的 instance selector。

```sh
ssh jump 'kubectl get applications -n argocd'
ssh jump 'kubectl get pipelineruns,taskruns -n rtl-pipelines'
```

Argo UI 仅ClusterIP。可在control上开启本地监听的 `kubectl -n argocd port-forward svc/argocd-server 8443:443 --address 127.0.0.1`，再通过受控SSH隧道访问；管理员凭据在独立渠道管理，不写在文档或URL中。

## 日常配置变更

main 是期望状态。PR必须通过 validate 检查和审查；初始化阶段由管理员提交并逐组验收，正常运维不要依赖管理员绕过。首次自动 prune 关闭，Git删除只会显示漂移，数据删除需要另行操作。Namespace、PVC、CRD禁止 Argo prune/delete。

自建镜像使用节点本地已导入版本；源码修改不会自动构建或发布镜像。按对应Dockerfile构建、在所有指定节点导入并验证镜像，再提交镜像版本变更。源代码和配置仓库不是模型/数据库/镜像仓库的替代品。

`gitops-reconciliation-proof` 是独立的无业务影响 ConfigMap，用来验收Git变更自动到达和现场漂移自愈，不参与应用运行。

## 动态字段

Argo不管理Job/Pod/ReplicaSet/PipelineRun/TaskRun/Lease、自动EndpointSlice、路由状态ConfigMap、Secret和数据。`glm-nightly.spec.jobId` 由Brain选取，Git保留首装所需字段但同步忽略这个字段；其他调度策略由Git负责。维护时要改变 suspend/mode 等策略，应先在Git提交；紧急现场操作应先暂停相关Application自动同步，处理后再回写Git恢复控制。

Tekton webhook CA bundle、聚合ClusterRole的rules、PVC绑定volumeName由对应Kubernetes控制器管理。忽略规则只针对这些字段，不能通过忽略整个spec隐藏漂移。

## 边界

Argo控制面、NewAPI入口与数据库、brain local-path存储仍存在单点。Git备份配置不备份数据库、训练产物或Secret。无cluster凭据的公开CI只能做静态验证；真实部署验收必须查看Application状态、实际Pod身份和服务健康。GitHub/网络故障期间已有工作负载继续运行，Argo无法取得新提交时不会把本地旧配置视为新发布。
