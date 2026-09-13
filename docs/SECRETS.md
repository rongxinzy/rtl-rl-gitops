# 凭据与恢复

公开仓库只保存引用；实际值必须从组织的独立凭据保管渠道恢复。不得将从线上导出的完整 YAML/JSON、base64 内容、last-applied 注解、请求日志放进 Git。

| Namespace | 外部 Secret |
|---|---|
| new-api | new-api-runtime |
| rtl-brain | rtl-brain-secrets |
| rtl-egress | wg-primary、wg-client、wg-backup |
| rtl-system | rtl-router-admin、rtl-operator-ssh、rtl-brain-provider |
| rtl-pipelines | rtl-worker-client、rtl-executor-client |

保留原 Secret 的 key 名及协议含义；清单中的 secretKeyRef/volume items 是调用契约。不要重生成数据库密码、SESSION_SECRET 或 WireGuard 身份再覆盖正在运行的应用。轮换应分别验证消费者，必要时安排滚动发布。

Tekton webhook 证书、Argo CD 运行密钥与初始管理员凭据由平台管理，不导入 Git。Argo 控制台保持 ClusterIP，使用受控 SSH/端口转发访问，管理员凭据单独取用和轮换，不回显到对话或 CI 日志。

恢复顺序：基础主机与K3s → 数据/模型及既有主机服务 → 命名空间与外部Secret → 本地镜像导入 → Argo bootstrap → 各 Application。PVC声明不是数据备份，恢复 brain 状态需单独备份数据目录；数据库和训练产物同理。

源码中的路径变量、Secret名称和示例占位符不是凭据。敏感扫描是发布门禁的一部分，仍需人工审查新增配置和编码后的数据。
