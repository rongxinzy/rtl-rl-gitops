# L20 SwanLab 实时指标转发

## 实验信息

实验名称采用 `RTL SFT | Qwen3.8-27B | <job_id>`。顶部包含训练目标描述及 RTL、SFT、QLoRA、NF4、模型、L20、单卡等标签；Tekton 任务另有 Tekton 标签。任务类型明确是监督微调，不把 SFT 实验标为 GRPO。

Config 保存任务身份、官方模型 revision、数据/配方/镜像摘要、冻结评测版本、rank、学习率、seed、序列长度、数据预检统计及可训练参数/量化模块信息。训练设备字段来自 L20 的 querygpu、CPU 与内存观测，明确标记观测时间和历史核验边界；不启用上报宿主机的自动硬件探针，避免误报 pro6000D 为训练设备。

Logs 保存从已验证指标文件重建的结构化事件，包括任务接入、每步 loss/梯度/显存/耗时/检查点名称及训练终态。仅代理该进程的受控 stdout，不采集原始训练 stdout、环境变量、样本、提示词或凭据。新增学习率、进度百分比和会话耗时曲线；恢复后的会话耗时不声称是总 GPU 用时。

SDK 0.10 的 resume 不更新旧 run 顶部字段。`enrich_existing.py` 使用与 SwanLab 网页相同的名称/描述与增量标签接口补齐，并回读确认；不改变实验状态。旧实验的任务类型可在 Config 的 task_type/stage 字段查看，新实验还在 init 中设置 job_type/group。这些接口的兼容性变化必须通过云端回读验收，不能仅凭 HTTP 成功判定字段已生效。

L20 训练容器保持离线。pro6000D 的 systemd timer 每30秒通过专用、受限 SSH 身份读取 L20 的标量快照，独立 CPU 容器持续上传到 SwanLab。训练不依赖上传服务成功与否。

`source.py` 只允许读取任务身份、运行状态及训练指标文件；校验官方模型 revision、连续步数和终态绑定，不读取训练数据、模型权重或凭据。SSH authorized_keys 必须使用 `restrict,from="172.18.4.199",command="/usr/bin/python3 /opt/rtl-swanlab/source.py"`，不能授权交互 shell。主机公钥需经已有可信 SSH 通道核对，不能禁用 host key 检查。

pro6000D 路径：

- 程序：`/root/rtl-rl/infrastructure/swanlab/`
- 快照：`/root/rtl-rl/swanlab-relay/snapshot.json`
- 状态和 SDK 持久队列：`/root/rtl-rl/swanlab-relay-state/`
- SwanLab 凭据：沿用 `/root/rtl-rl/secrets/swanlab.env`，不写入仓库。
- 专用 SSH 私钥与已核对的主机公钥：`secrets/metrics_ssh`、`secrets/metrics_known_hosts`，独立配置，不发布值。

从任务 ID 与身份摘要生成固定 run ID，使用 `resume=allow`。同一进程只上报新步数；进程恢复时交由 SDK 按远端步数续传，避免仅依赖本地乐观游标丢失尚未送达的数据。SFT 完成且两个终态文件通过校验后才结束 run；失败标记 crashed，采集过期或上报中断标记 aborted 并重试。这里的完成只指训练阶段，不代表后续候选评测通过或模型能力提高。

每个任务的状态 JSON 保存 run ID、链接、已提交给 SDK 的步数和 SDK finish 返回情况。它不替代云端独立验收。SDK 标准输出不转发到 systemd 日志，日志中不输出原始 API 异常。

启动 `rtl-swanlab-collect.timer` 和 `rtl-swanlab-relay.service`；二者均不访问 GPU、不停止训练。当前采用已有宿主机 Docker/systemd 边界，因此这些 unit 是可审查的主机配置，不是 Argo 自动管理的 Kubernetes 工作负载。
