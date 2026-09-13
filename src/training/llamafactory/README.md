# LLaMA-Factory 后端适配

此目录为既有单卡 worker 的替换配方。`build_recipe.py --output <新目录>` 复制原版评测和来源校验代码，并安装新的训练入口和 checkpoint 校验器。不要修改已有任务的 recipe；新任务才选择此目录。镜像必须安装固定上游 commit `100e9a42c6c09f8f7849b70d60f3da445fb2024b`，并将其记录在 `/opt/llamafactory/COMMIT`。

训练实际调用 `llamafactory.train.tuner.run_exp`，使用 SFT、官方 qwen3_5_nothink 模板、ShareGPT、rank 8、alpha 16、NF4 双重量化、BF16、1024 token、最多 20 步。只接受 train split 的 Q2 或已校验 K1-grounded 文本，过滤 tokenizer 与 LF 模板任一种编码超长的样本，不导入评测提示或其他元数据。官方模型来源和权重完整性沿用现有验证器。

每步 HF 原生保存后，桥接器校验并原子发布六位数 checkpoint；完整文件摘要绑定不可变任务身份。恢复直接传 HF `resume_from_checkpoint`，恢复 optimizer、scheduler、RNG、Trainer global step 和数据跳过语义。只保留两个完整桥接 checkpoint；HF 自己也保留两个。暂停在步末请求保存，退出后从最后已验证 checkpoint 发布 adapter 和 worker 所需结束文件。崩溃恢复时丢弃超过已提交 checkpoint 的指标，避免重复步数。旧后端检查点校验分支保留，但新后端拒绝从旧训练器状态恢复。

`python3 training/llamafactory/test_bridge.py` 为纯 CPU 契约测试，不能作为真实 GPU 兼容性或能力提高的证据。部署还需隔离 GPU 冒烟、真实暂停/恢复与流水线验收。

镜像构建使用固定源码生成的 `llamafactory-0.9.6.dev0-py3-none-any.whl` 和 `requirements-runtime.txt` 中的依赖 wheel，放入构建上下文的 `wheels/`，随后执行 `docker build --network=none`。底层 CUDA/PyTorch 镜像是独立前置条件；本目录不能从空主机独立恢复训练环境。构建阶段执行 `pip check`，不绕过依赖冲突。

CPU 处理器检查：在新镜像内，以只读方式挂载实际模型目录，运行 `python check_processor.py <模型目录>`。它加载 tokenizer/processor 并执行真实文本 collator，不加载 27B 权重。通过后仍需 GPU 验收。发布的 adapter/checkpoint 依赖已锁定的底座及其 processor，并非自包含模型。

崩溃恢复补充：恢复前校验输出目录中所有正式六位数 checkpoint，选择同一任务最高完整步数，再 fsync 并原子修复 `latest`。这覆盖目录已 rename、指针尚未更新的窗口；第一步崩溃导致指针尚不存在时，也按同一不可变任务身份自动恢复。任一正式 checkpoint 损坏、属于其他任务、目录步数与 Trainer 不符则失败，不静默退回旧优化器。显式恢复较旧 checkpoint 被拒绝，避免重复执行已发布步数；`.tmp` 未发布目录不被采纳。

最高已验证 checkpoint 达到 `max_steps` 时，恢复入口只重新生成 adapter 和最终状态文件，直接返回，不调用 LF Trainer、不加载 GPU 模型、不再发布同一步 checkpoint。CPU helper 测试覆盖完成、未完成、损坏与重复收尾场景。
