# 主机 bootstrap 与服务来源

这里是显式白名单的公开代码快照，**不是已实现的主机自动 GitOps**。Argo CD 只管理 Kubernetes 声明式资源；主机安装、systemd、Docker GPU、数据库和模型恢复仍是代码加人工受控入口。本轮未对任何主机执行变更。

## 节点和前置条件

- rtl-control / 172.18.5.188：单 K3s 控制面，New API 入口与既有 Docker PostgreSQL/Redis、DB私网bridge；不是控制面或存储HA。
- rtl-pro6000d / 172.18.4.199：8卡GPU，日间生产GLM、夜间训练、受限executor/judge。GPU Docker归宿主调度，K3s不能绕过资源门禁。
- rtl-l20 / 172.18.6.123：2卡L20，持久worker API与独立实验；模型与checkpoint在/mnt/data。不得依据GPU低利用率终止服务。

K3s固定 `v1.36.4+k3s1`，实测runtime containerd `2.3.4-k3s1.36`。样本配置保留实际node名/IP/接口、Pod10.42/16、Service10.43/16、Traefik/ServiceLB禁用、GPU节点external taint。部署前核实网卡与现有配置，不能把样本直接覆盖运行节点。现有Docker与K3s containerd相互独立；不安装GPU Operator来抢占Docker设备。

## 安全身份预置

join-token、server-token、kubeconfig、SSH私钥、worker/executor API key、New API DB/Redis/会话密钥、模型下载凭据均外部预置，root-only 0600，绝不写进本仓库。config样本只有token-file路径。Kubernetes Secret由安全入口预置，Argo仅引用。不能把`systemctl show Environment`、Docker env、job配置、完整ConfigMap/Secret导出当作恢复资料。

## 有界、幂等部署方法

1. 在运维窗口先做只读盘点与独立数据库备份：记录boot ID、GPU Docker ID、systemd启用状态、API健康及当前实验。仓库不包含这些动态记录。
2. 新节点bootstrap从官方K3s固定版本release获取二进制/airgap images并核对官方SHA256；人工获取并审查官方安装器，使用离线安装与明确server/agent角色。既有节点先比对版本；相同版本不重新安装、不重启。这里没有一键重建脚本，也不承诺bootstrap恢复已验收。
3. `python3 host/check_sources.py`验证来源完整性。使用`cmp -s`判断目标文件是否变化；无差异不写入。需要变更时先备份精确目标、安装单个文件到其unit声明路径，执行`systemd-analyze verify`；只有unit有变更才`daemon-reload`。安装文件不等于授权restart/enable。
4. `sources/infrastructure/k3s/guard-network.sh`以iptables -C判断后添加本项目链/规则，重复运行不复制规则、不清空其他链。它会修改防火墙，只能人工审阅后调用；不是通用网络收敛器，变更peer列表前需要独立规则迁移计划。
5. 任何API重启须按当前任务门禁操作，核验GPU Docker ID不变。训练单元的停止语义和checkpoint超时必须保留。不要批量`enable --now *.service`；night reconcile由现有operator/host策略决定启用，不能与另一个协调器双写。
6. 回退只还原本次改动文件；不自动恢复旧数据库、删除卷/模型/任务或重新启用旧NewAPI容器。所有会影响服务的动作单独审查。

## 来源白名单与依赖

`source-manifest.json`给出13个逐文件SHA256，来自RTL_RL本地受审查代码，未声称是当前主机全部drop-in的导出。`sources/`保持原相对路径方便审查：

- k3s三份配置样本、网络防护脚本及unit。
- NewAPI data_bridge.py及unit，固定现有Docker网络与容器名，私网15432/16379。数据库原卷/用户/密码/备份必须外部恢复完成，bridge不创建数据库、不做schema迁移。原入口hostPort3000属于K8s ingress，不能另起旧Docker new-api争用。
- executor、L20 worker、night training/reconcile及timer、judge unit。ExecStart所需完整Python包由仓库`src/`另行提供，部署者必须按unit固定路径安装并检查依赖；此目录不是完整运行包。
- judge unit内bootstrap-v2 registry是原样来源默认值，实际registry由受控job/drop-in绑定；本仓库不含训练数据/registry。不得直接启用缺少真实来源验证的judge。

未复制历史standalone scheduler、backup GLM自动激活/下载timer：它们与当前调度角色可能冲突，不能以恢复名义重新启用。未复制job.json、config.json、api-key、日志、下载/评测输出、数据、模型、数据库或主机秘密。现有bind/drop-in和运行策略必须在部署前单独对照，不通过整目录scp覆盖。

## 持久资源恢复顺序

先主机网络/磁盘挂载与Docker/NVIDIA runtime，再恢复既有数据库与私网bridge、模型来源验证和checkpoint目录，再预置最小权限身份，最后逐项接回worker/executor/Kubernetes调用端。官方Qwen来源与派生NF4 SHA链由模型验证程序检查；`MODEL_READY`不能人工伪造。模型不在Git中，恢复路径必须真实可读且完整。K3s local-path PVC及其节点本地目录独立备份，不能当跨节点存储。L20普通Pod出网隧道不覆盖宿主containerd image pull，离线镜像仍需独立预载。
