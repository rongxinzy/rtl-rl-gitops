# L20 SwanLab 实时指标转发

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
