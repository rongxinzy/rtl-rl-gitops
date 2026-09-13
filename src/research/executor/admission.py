"""Training admission has no GPU lifecycle commands or arbitrary recipe options."""
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import urllib.request
from .core import atomic, digest


def resource_gate(root, rows):
    if len({row.get('task_id') for row in rows if row.get('task_id')}) < 8:
        raise ValueError('Training requires at least 8 distinct trusted train tasks')
    if shutil.disk_usage(root).free < 50 * 1024 ** 3:
        raise ValueError('Training requires at least 50 GiB free disk')


def admit(executor, params):
    if params.get('max_steps', 20) > 200:
        raise ValueError('Admission permits at most 200 steps')
    executor.propose(params)
    gates = executor.state / 'verified'
    dataset = json.loads((gates / ('dataset-' + params['dataset_id'] + '.json')).read_text())
    relative = dataset['data']
    path = (executor.root / relative).resolve()
    allowed = (executor.root / 'research/data_factory').resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError('Dataset path outside immutable factory artifacts')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != dataset['data_sha256']:
        raise ValueError('Dataset content changed')
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows or not all(row.get('task_id') for row in rows):
        raise ValueError('Dataset missing registered tasks')
    resource_gate(executor.root, rows)
    scheduler = executor.root / 'state'
    with (scheduler / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job_path = executor.root / 'scheduling/job.json'
        current = json.loads(job_path.read_text())
        job_id = 'brain-' + digest(params)[:24]
        if current['job_id'] == job_id:
            return {'status': 'admitted', 'job_id': job_id, 'operator_patch_required': True}
        if not (executor.root / 'runs' / current['job_id'] / 'job-complete.json').is_file():
            raise ValueError('Current job has not completed')
        from .iteration import gate
        last_outcome = gate(executor, current, params)
        service = subprocess.run(['systemctl', 'show', 'rtl-night-training.service', '-p', 'ActiveState', '--value'], capture_output=True, text=True, timeout=10)
        if service.returncode or service.stdout.strip() not in ('inactive', 'failed'):
            raise ValueError('Training service is not inactive')
        running = subprocess.run(['docker', 'ps', '--filter', 'name=^/rtl-night-training$', '--format', '{{.ID}}'], capture_output=True, text=True, timeout=10)
        if running.returncode or running.stdout.strip():
            raise ValueError('Training container may be active')
        # Candidate contamination check uses the exact frozen baseline, never latest.
        evaluation = json.loads((gates / ('evaluation-' + params['evaluation_id'] + '.json')).read_text())
        freeze_path = executor.root / 'research/evaluation/artifacts/frozen' / (evaluation['freeze_id'] + '.json')
        check = subprocess.run(['/usr/bin/python3', str(executor.root / 'research/evaluation/cli.py'), 'check-training', '--root', str(executor.root), '--freeze-file', str(freeze_path), '--training', str(path), '--training', str(executor.root / dataset['registry'])], capture_output=True, text=True, timeout=60)
        if check.returncode or json.loads(check.stdout).get('status') != 'clean':
            raise ValueError('Candidate overlaps frozen evaluation')
        from .registry import upgrade
        prior_state = (scheduler / 'scheduler.json').read_bytes() if (scheduler / 'scheduler.json').exists() else None
        rollback_registry = upgrade(executor, dataset)
        try:
            job = {'job_id': job_id, 'data': str(path.relative_to(executor.root)), 'data_sha256': dataset['data_sha256'], 'max_steps': params.get('max_steps', 20), 'max_completion_length': 512, 'save_steps': 10, 'purpose': 'Autonomous validated single-GPU GRPO admission', 'dataset_id': params['dataset_id'], 'baseline_evaluation_id': params['evaluation_id'], 'evaluation_freeze_id': evaluation['freeze_id']}
            # Pending journal is durable before either file changes. Recovery repeats the
            # exact admission; old completed job stays harmless if interrupted before job commit.
            journal = {'previous_job_id': current['job_id'], 'job': job, 'evaluation_id': params['evaluation_id']}
            atomic(scheduler / 'admission-pending.json', journal)
            state_path = scheduler / 'scheduler.json'
            old_state = json.loads(state_path.read_text()) if state_path.exists() else {}
            clear = {'started_window', 'training_boot_id', 'resume_window', 'failed_window', 'idle_after_completion', 'stop_requested', 'training_pids', 'glm_stop_window', 'glm_stop_started', 'glm_pids'}
            state = {key: value for key, value in old_state.items() if key not in clear}
            state.update(admitted_job_id=job_id, phase='queued', last_iteration_outcome=last_outcome)
            atomic(state_path, state)
            atomic(job_path, job)
            atomic(scheduler / 'admission-complete.json', journal)
            (scheduler / 'admission-pending.json').unlink(missing_ok=True)
            return {'status': 'admitted', 'job_id': job_id, 'operator_patch_required': True}
        except Exception:
            atomic(job_path, current)
            if prior_state is not None:
                (scheduler / 'scheduler.json').write_bytes(prior_state)
            rollback_registry()
            raise
