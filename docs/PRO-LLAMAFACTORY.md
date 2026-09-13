# Pro nighttime LLaMA-Factory migration

This adds bounded NF4 QLoRA **SFT**, not GRPO, to the existing nighttime runner. Existing GRPO experiments and checkpoints remain immutable. The official Qwen/Qwen3.8-27B revision and LF commit are pinned in code; the deployment profile must pin the local image digest and every recipe file.

## Ownership and activation

The Operator retains GPU and inference lifecycle ownership. The runner does not stop GLM, change routing, restart hosts or claim GPU reclamation. Static Kubernetes policy remains Argo-managed; jobId is a dynamic Operator/Brain field. Do not temporarily edit `spec.suspend` without managing Argo reconciliation: self-heal can revert it.

Host source installation remains an explicit deployment step after reviewed Git merge; merely merging this source does not install Python files on a host. Preserve existing local hotfixes and unrelated changes. Install only the reviewed changed modules and their imports, then restart the CPU Executor if needed. No blanket workspace copy.

For a never-started queued job, hold the host scheduler and block Operator dispatch before installation. A temporary jobId that cannot match the host queue blocks dispatch before route changes; record both identities, verify `BlockedJobMismatch`, and restore the new admitted identity at activation. This is a temporary migration guard, not an experiment or completion record. Do not use it to interrupt an active job.

Place a host-owned `scheduling/llamafactory-profile.json` containing exactly backend, image_id, recipe_path, recipe_sha256, model_path, model_revision and llamafactory_commit. Paths are relative to the host root. Model/data/credentials and live profile are not committed to Git.

`research.executor.lf_profile.migrate_queued(root, expected_job_id)` requires the hold, scheduler lock, exact old queue identity, no old run directory, inactive training service and no training container. It revalidates the same factory dataset SFT export against the frozen evaluation, archives the old queued configuration and returns a new content-addressed job id. It does not patch Kubernetes or manufacture old-job completion. Patch the Operator jobId to that returned identity only after verification, remove the hold, and observe the existing business-idle/drain gates before actual training.

## Baseline and training

The runner first creates a new baseline under the pinned LF image and original BF16 model, using the existing frozen tasks and judge. A prior GRPO image baseline cannot qualify. The baseline hash is saved into the immutable job config. Training then runs on GPU0 with an NF4 model, rank8, length1024 and 1–20 steps. The container has no network or credentials. Candidate evaluation uses the same image/freeze as the new baseline.

Native checkpoints and optimizer/scheduler/RNG state live under `train-run/`; the outer job identity remains separate. Published adapter and final state are verified against the committed checkpoint before candidate evaluation. Baseline interruptions signal only the owned CLI, allowing its finally block to remove its exact evaluation container. SIGKILL or power loss cannot guarantee Python cleanup; existing host lifecycle recovery remains required.

## Telemetry and acceptance

The Pro collector reads only validated identities and scalar metrics from the local run directory, merged independently with L20 telemetry. L20 outage marks cached entries stale without inventing progress or ending runs. Pro has separate run IDs and per-job device metadata; old L20 IDs remain unchanged. No training text, raw logs or credentials are uploaded by this collector.

CPU tests cover admission, checkpoint binding, stop cleanup, evaluation runtime identity and telemetry. A target-image tokenizer/template preflight is separate from GPU acceptance. A successful prior L20 two-step resume test does not prove the different Pro runtime or full nightly baseline/train/candidate chain. Actual Pro GPU acceptance must wait for the authorized idle window. No capability improvement is claimed by this migration.
