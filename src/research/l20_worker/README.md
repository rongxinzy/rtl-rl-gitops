# Fixed L20 worker (prepared, not enabled)

This host API runs one experiment at a time: baseline generation, bounded 1–20-step training (default 20 at the caller), candidate generation, then external CPU-judge comparison acknowledgment. Every container uses GPU 0, network none, a pinned image digest and copied recipe hashes. It never stops unrelated GPU processes or containers. Training is not enabled merely by starting the API: the deployment must explicitly create `<ROOT>/enabled`.

## Install after review

On 6.123 create `/mnt/data/rtl-l20-training/worker/{source,jobs}` and copy only this module's Python files into `source/`. Copy the reviewed unit to `/etc/systemd/system/rtl-l20-worker.service`. Configure `<ROOT>/config.json` with root-owned values:

- `image_id`: verified `sha256:` image digest, already imported on this host.
- `model_revision`: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- `recipe_path`: directory containing the reviewed `training/l20/*.py` recipe, including `state.py` and `provenance.py`.
- `model_ready`: path to the model readiness JSON. Its `path` field supplies the model directory. A legacy `model_path` config value is ignored.

Generate a strong token directly on the target into `<ROOT>/api-key` with mode 0600; do not put it in this repository. The service initially binds localhost:18766. Root may install a drop-in setting `L20_WORK_BIND=172.18.6.123` only with its reviewed cluster ingress firewall. Review configuration and run tests first, then `systemctl daemon-reload` and `systemctl start rtl-l20-worker`; enable boot startup only when authorized. No installation or enabling was performed by this subtask.

The model marker is not trusted on its own. Each launch runs the pinned `provenance.verify_model` against its resolved path, accepting only the official source or the verified NF4 derivative. Hashing may take time. During a manager critical section, the API returns bounded `503 worker_busy` rather than racing state. Missing model readiness and insufficient GPU memory wait without spending retry attempts. Real start failures have five-minute backoff and fail after three attempts. An uncertain Docker response is adopted by job/image/GPU/network identity on the next pass.

## API and lifecycle

All endpoints require `Authorization: Bearer <token>`. `GET /status` reports sanitized job state; `POST /jobs` accepts the fixed typed dataset/prompt body; `GET /jobs/<id>/artifacts` returns results only after actual checkpoint/adapter/evaluation identity verification. `POST /jobs/<id>/evaluation` requires a 64-lowercase-hex comparison ID and one outcome. An identical acknowledgment is idempotent; changing an already committed comparison is rejected.

HTTP reads time out after 10 seconds, bodies are limited to 5 MiB and chunked requests are rejected. API and manager share one reentrant lock. CPU API resource limits do not constrain the Docker container's separate explicit limits.

To pause, create `<ROOT>/pause` or remove `<ROOT>/enabled`. The manager writes the owned job's `pause.request`, letting training finish its current step and publish a complete optimizer checkpoint. Baseline/candidate generation is not force-killed; it completes its current bounded batch. Remove pause and create enabled to resume from the verified `latest` checkpoint. Controller restarts adopt an owned running container instead of launching a duplicate. The latest pointer, checkpoint hashes, final step/metrics, adapter files and evaluation prompt/model identities are checked before advancing. Container exit code zero alone is never completion evidence.

Tests: `python3 -m research.l20_worker.test_worker`. Also `python3 -m research.teacher.test_teacher` verifies compatibility with the teacher's no-redirect opener.

Stopping/restarting the control API does not terminate the Docker experiment; request pause first when intentional training suspension is required. If the container has disappeared but fully verified output already exists, the manager advances from those artifacts without rerunning the finished phase. Disappearance without valid output consumes the bounded retry budget. Model and artifact verifiers execute only copied recipe helpers whose hashes still match the immutable job manifest.

## Optional trusted training profiles

The existing default `image_id` and `recipe_path` remain unchanged. An operator may add
`training_profiles` to the local worker configuration. Each ID (lowercase letter then
at most 63 lowercase letters, digits or hyphens) maps to exactly `image_id` (immutable
`sha256:` Docker image ID), `recipe_path` (absolute local directory) and
`recipe_sha256` (complete filename-to-SHA256 map of admitted non-test Python files).
The deployed directory must match every pinned hash; extra Python files, missing files,
symlinks and mutable image tags are rejected. Profile configuration and recipes require
normal PR/CI review and controlled host deployment before use; this is not a remote
code-upload API. Model revision and GPU ownership stay in the existing worker policy.

A new `l20.admit` request may specify `params.profile_id`; the executor forwards only
that identifier. Neither API accepts an arbitrary image or recipe path. Admission
records the profile ID, actual image ID and recipe hashes in the new immutable job,
then snapshots the recipe. Requests without a profile retain their previous identity
format and default behavior. Existing tasks never re-read the selected profile for
their pinned image/recipe. Tekton still receives only `job-id`. This does not bypass
training-data validation, semantic deduplication, frozen evaluation, or the single
active experiment limit.
