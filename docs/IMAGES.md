# GitOps 镜像冷启动边界

应用声明和部分构建源码齐备，但当前**尚不是独立可复现的完整镜像供应链**。以下依据2026-09-13的 `src/` 与 `clusters/lab/` 静态检查；未把源码存在当成构建成功。

## 六份 Dockerfile 检查

假设每份Dockerfile所在目录为build context，12个COPY源文件全部存在；未发现直接COPY缺文件。仓库根直接作为context则不成立，应明确每组件context。

| Dockerfile（src下） | 基础镜像 | COPY与缺口 |
|---|---|---|
| operator/Dockerfile | rtl-python-base:20260913 | controller.py存在；该base的Dockerfile/来源digest未见于src。apk安装openssh-client/tzdata未固定包版本，需要外网或固定离线APK仓库。 |
| research/brain/Dockerfile | rtl-operator:20260913-v3 | providers/state/workflow/loop/guardian/observer共6文件存在；依赖旧v3镜像，而部署operator为v4。必须保留v3制品或补其精确构建来源，不能假设当前operator源码生成相同v3。 |
| infrastructure/tekton/runtime/Dockerfile | rtl-operator:20260913-v3 | client/task/bridge均存在；同v3前置依赖。重新构建不会自动得到现有部署digest，必须验证并更新声明或分发原制品。 |
| infrastructure/router/Dockerfile | created-at-shim:latest | router.py存在；base没有公开构建配方，latest不可复现。v1/v2部署标签与单一当前源码的映射未完整提供。 |
| infrastructure/egress/Dockerfile | created-at-shim:latest | egress.py存在；wireguard-tools/iproute2/iptables/curl由未固定apk版本安装，base与包源均是前置。 |
| infrastructure/new-api/Dockerfile.ingress | created-at-shim:latest | 没有COPY，nginx配置由K8s ConfigMap挂载；base缺构建来源，apk nginx无版本锁，需单独验证配置兼容性。 |

`created-at-shim:latest`本身还作为NewAPI workload运行；src中没有其专用Dockerfile/应用源码。`calciumion/new-api:rtl-pinned-rc36`是本地保留的原应用镜像别名，公开src中没有NewAPI完整上游源码/固定commit构建过程。二者不能仅凭此仓库从零构建。

## 冷节点必须预置的运行镜像

以下来自声明文件，不代表公开registry确实存在对应本地tag：

- Never：docker.io/library/created-at-shim:latest、rtl-new-api-ingress:20260913、rtl-egress:20260913-nonpreempt；docker.io/calciumion/new-api:rtl-pinned-rc36；rtl-brain:20260913-v12。
- Never：docker.io/library/rtl-tekton-runtime@sha256:ed6625c4230fa46d33b78ac0f55da51f2a21f69dc7a511acd24799fcada682a2。
- Never：Argo CD（quay.io/argoproj/argocd）、Dex（ghcr.io/dexidp/dex）、Redis（public.ecr.aws/docker/library/redis），均固定digest，准确完整引用见clusters/lab/argocd/resources.json。即使registry可用，Never仍要求提前导入。
- IfNotPresent：rtl-router:20260913-v1、rtl-router:20260913-v2、rtl-operator:20260913-v4。它们是本地名称；缺缓存时将尝试默认registry，不能据此认为可下载。
- Tekton平台4个v1.16.0镜像（events/controller/webhook/resolvers）是官方ghcr.io digest pin、IfNotPresent。是否能从冷节点拉取没有在本次验证；离线恢复仍须预载。控制器配置可能还有动态任务辅助镜像，需独立提取配置镜像字段，不能只盘点PodSpec便声称全部依赖。

另需构建环境预置`rtl-python-base:20260913`、`rtl-operator:20260913-v3`、`created-at-shim:latest`。镜像导入必须是目标节点K3s containerd的k8s.io命名空间与准确引用；Docker image存在不等于K3s可用。不同节点架构与镜像platform也需要验证。

## 不属于本轮完整交付的供应链环节

- 尚未发现这6份配方的统一构建入口、每版本源码到digest映射、固定base digest、APK锁定快照、可验证SBOM/签名或公开镜像registry发布流程；这不否认现有节点已能运行，只限制冷重建声明。
- 宿主GPU训练/推理镜像、CUDA/torch/Transformers/bnb离线依赖、模型和checkpoint不由这6份Dockerfile提供。Argo接管不重建它们。
- Pod出网隧道不覆盖宿主containerd拉镜像。L20宿主没有公网时必须采用校验过的离线导入或另设受控registry，不可依赖普通Pod egress。
- 已运行Pod身份保持、Argo Synced与Never策略，证明声明接管没有重启现有服务；不证明空节点恢复或镜像可重复构建。

建议交付措辞：已提供Kubernetes GitOps声明、主机bootstrap白名单与核心服务源码，当前部署依赖预置镜像和外部持久数据；独立冷启动镜像构建/发布/导入链仍需补齐并验收。本次不修改现有image或pull policy，避免把接管变成意外升级。


## Ops and traffic identity (2026-09-14)

Locally built and imported on control/L20 before deployment; imagePullPolicy Never. These manifests reference OCI manifest digests, not Docker config IDs.

- router: c7c35c7fab40bc780eeb9c3d005cf4820c467dfe302229151586580587fa9ef3; includes timezone data from rtl-operator:20260913-v4.
- coordinator: 9546ecb754e60bf985508d6981677ce971b751072da46fc316672912a1aae12a.
- operator: e82e9a76e1f85625ea5bcd4e60341adcb0420ae358d662773d8836726f956fc8; copies current controller.py over rtl-operator:20260913-v4.
- ops agent: d23af1f0535de7a4456015c9e131cd7101a9f96f2dc6a864fd74ba73f5785ec7; BASE_IMAGE=rtl-brain:20260913-v12, with the same updated providers.py as coordinator.

Background router credentials are provisioned privately into the existing router and Brain Secrets; values are never in Git. Existing New API credentials stay protected. Ops CronJob is enabled after live read-only, admission-denial and isolated recovery acceptance. Production recovery allows only glm-router and rtl-operator; the canary is excluded from production configuration. The read-only and active acceptance jobs use separate state directories; production budgets are not reset for tests.

Production ops observations run every minute; LLM plans are spaced 30 minutes apart (at most 48/day), below the 60/day hard cap so routine polling cannot exhaust the model budget before night. Candidate recovery still requires continuous failure and all safety gates; its next eligible model plan may add up to 30 minutes latency.
