# 推理请求身份与夜间门控

已有 `/secrets/key` 保持 protected 身份，新 `/secrets/background-key` 是可选的独立自动任务凭据。两值必须不同，未配置时保持旧行为；已存在但空值使启动失败。未知凭据返回 401，客户端的 User-Agent、请求频率、身份标签和正文均不能把 protected 降级。

后台凭据仅可使用 `/v1/`，不能访问 `/admin/`。默认北京时间 22:30（含）至次日 07:30（不含），新的 background POST 推理返回 503、错误类型 `background_paused`，调用方应使用已配置 fallback。通过 `BACKGROUND_NIGHT_START`、`BACKGROUND_NIGHT_END`、`SCHEDULE_TIMEZONE` 配置窗口。只在路由容器受信挂载中配置凭据，不在公开配置或日志记录值。

这代表显式的工作负载身份，不代表通过流量猜测真人。共享 key 的现有 new-api 请求仍全部受保护；受控自动调用方必须迁移至独立后台 key 后才能消除它们对夜间空闲的影响。既有真人/Agent 共享 token 无法可靠自动区分。

后台已在进行的请求继续排空，不中断流式输出。`primary_inflight` 与 `backup_inflight` 仍统计所有已接受请求，Operator 必须继续检查总在途为零。`last_business_at` 和 `business_idle_seconds` 保持兼容字段名，现定义为 protected 最后开始/完成时间及对应空闲时长。新增 protected/background 在途计数、各自最后活动时间及后台拒绝计数。health 与 GET 模型列表不刷新业务空闲；拒绝的后台请求也不刷新。滚动部署时旧副本缺少身份字段会保守回退至旧业务时间。

模块接口为 `authenticate(Authorization)`、`admit(identity,method,path)`、`begin/end(ticket)` 和 `snapshot()`。router 保留单独的后端排空计数。没有 LLM 分类或任意自报可信身份，状态输出不含密钥、请求正文或客户端标识。

测试：`python3 -B -m unittest discover -s infrastructure/router -p 'test_*.py'`。覆盖跨午夜边界、白昼窗口、凭据碰撞、HTTP 权限、自报伪装、流式完成计时、后台排空与敏感值不回显。
