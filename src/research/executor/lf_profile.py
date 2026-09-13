"""Host-owned LF profiles and explicit migration of never-started queued jobs."""
import fcntl
import hashlib
import json
import re
import subprocess
from pathlib import Path
from .core import atomic, digest

REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
COMMIT = '100e9a42c6c09f8f7849b70d60f3da445fb2024b'
CLEAR = {'started_window', 'training_boot_id', 'resume_window', 'failed_window', 'idle_after_completion', 'stop_requested', 'training_pids', 'glm_stop_window', 'glm_stop_started', 'glm_pids'}

def safe_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts:
        raise ValueError('profile path must remain relative to root')
    current = root
    for part in p.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('symlink forbidden')
    if not current.resolve().is_relative_to(root):
        raise ValueError('path escapes root')
    return current

def load_profile(root):
    path = safe_path(root, 'scheduling/llamafactory-profile.json')
    if not path.exists():
        return None
    p = json.loads(path.read_text())
    required = {'backend', 'image_id', 'recipe_path', 'recipe_sha256', 'model_path', 'model_revision', 'llamafactory_commit'}
    if set(p) != required or p['backend'] != 'llamafactory' or p['model_revision'] != REVISION or p['llamafactory_commit'] != COMMIT:
        raise ValueError('invalid trusted LF profile')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', p['image_id']):
        raise ValueError('immutable image required')
    recipe = safe_path(root, p['recipe_path'])
    model = safe_path(root, p['model_path'])
    if not recipe.is_dir() or not model.is_dir():
        raise ValueError('missing profile directory')
    hashes = p['recipe_sha256']
    if not isinstance(hashes, dict) or not hashes or set(hashes) != {x.name for x in recipe.iterdir()}:
        raise ValueError('complete recipe file set required')
    for name, expected in hashes.items():
        f = recipe / name
        if not re.fullmatch(r'[A-Za-z0-9_]+\.py', name) or f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != expected:
            raise ValueError('recipe integrity failed')
    return p

def config(root, old, profile=None, runner=None):
    """Revalidate the same factory dataset and frozen evaluation for SFT."""
    from research.data_factory.cli import read_dataset
    root = Path(root).resolve()
    profile = load_profile(root) if profile is None else profile
    if profile is None:
        raise ValueError('LF profile unavailable')
    if type(old['max_steps']) is not int or not 1 <= old['max_steps'] <= 20:
        raise ValueError('LF permits 1..20 steps')
    data = safe_path(root, old['data'])
    allowed = root / 'research/data_factory/state/datasets'
    if not data.is_relative_to(allowed) or data.parent.name != old['dataset_id']:
        raise ValueError('dataset identity mismatch')
    if hashlib.sha256(data.read_bytes()).hexdigest() != old['data_sha256']:
        raise ValueError('original training data changed')
    manifest = read_dataset(data.parent)
    sft = safe_path(root, str((data.parent / 'sft_train.jsonl').relative_to(root)))
    content = sft.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if manifest['dataset_id'] != old['dataset_id'] or sha != manifest['files']['sft_train.jsonl']:
        raise ValueError('SFT integrity mismatch')
    freeze = old['evaluation_freeze_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', freeze):
        raise ValueError('invalid freeze identity')
    freeze_path = safe_path(root, 'research/evaluation/artifacts/frozen/' + freeze + '.json')
    check = (runner or subprocess.run)(['/usr/bin/python3', str(root/'research/evaluation/cli.py'), 'check-training', '--root', str(root), '--freeze-file', str(freeze_path), '--training', str(sft), '--training', str(data.parent/'registry_train.jsonl')], capture_output=True, text=True, timeout=60)
    if check.returncode or json.loads(check.stdout).get('status') != 'clean':
        raise ValueError('SFT overlaps frozen evaluation')
    result = dict(profile, data=str(sft.relative_to(root)), data_sha256=sha, dataset_id=old['dataset_id'], max_steps=old['max_steps'], max_length=1024, lora_rank=8, save_steps=1, purpose='Validated single-GPU LLaMA-Factory SFT', source_baseline_evaluation_id=old.get('source_baseline_evaluation_id', old.get('baseline_evaluation_id')), evaluation_freeze_id=freeze)
    result['job_id'] = 'brain-' + digest(result)[:24]
    return result

def inactive(runner=None):
    runner = runner or subprocess.run
    service = runner(['systemctl', 'show', 'rtl-night-training.service', '-p', 'ActiveState', '--value'], capture_output=True, text=True, timeout=10)
    container = runner(['docker', 'ps', '--filter', 'name=^/rtl-night-training$', '--format', '{{.ID}}'], capture_output=True, text=True, timeout=10)
    if service.returncode or service.stdout.strip() not in ('inactive', 'failed') or container.returncode or container.stdout.strip():
        raise ValueError('training service or container active')

def migrate_queued(root, expected_job_id, checkrunner=None):
    root = Path(root).resolve()
    state = root/'state'
    if not (state/'scheduler-hold').is_file():
        raise ValueError('scheduler hold required')
    with (state/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (state/'scheduler-hold').is_file():
            raise ValueError('scheduler hold removed')
        path = root/'scheduling/job.json'
        old = json.loads(path.read_text())
        if old['job_id'] != expected_job_id or not re.fullmatch(r'brain-[0-9a-f]{24}', expected_job_id):
            raise ValueError('queued job CAS mismatch')
        run = root/'runs'/expected_job_id
        if run.exists() or run.is_symlink():
            raise ValueError('started job cannot migrate')
        inactive(checkrunner)
        new = config(root, old, runner=checkrunner)
        if new['job_id'] == old['job_id']:
            return {'status': 'unchanged', 'job_id': new['job_id']}
        if (root/'runs'/new['job_id']).exists():
            raise ValueError('target experiment already exists')
        journal = state/'superseded'
        journal.mkdir(exist_ok=True)
        record = {'previous_job': old, 'job': new, 'reason': 'explicit queued LF migration'}
        dest = journal/(expected_job_id + '.json')
        if dest.exists() and json.loads(dest.read_text()) != record:
            raise ValueError('conflicting migration journal')
        atomic(dest, record)
        atomic(state/'admission-pending.json', record)
        state_path = state/'scheduler.json'
        prior = json.loads(state_path.read_text()) if state_path.exists() else {}
        next_state = {k:v for k,v in prior.items() if k not in CLEAR}
        next_state.update(admitted_job_id=new['job_id'], phase='queued')
        atomic(state_path, next_state)
        atomic(path, new)
        atomic(state/'admission-complete.json', record)
        (state/'admission-pending.json').unlink(missing_ok=True)
        return {'status':'admitted', 'job_id':new['job_id'], 'previous_job_id':expected_job_id, 'operator_patch_required':True}
