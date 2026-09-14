"""Read only whitelisted L20 identity and scalar metric files; emit no raw errors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import platform
from datetime import datetime, timezone

REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
ROOT = Path('/mnt/data/rtl-l20-training/worker/jobs')
PHASES = {'queued', 'baseline', 'training', 'candidate', 'complete', 'failed',
          'cancelled', 'queued_baseline', 'queued_training', 'queued_candidate',
          'awaiting_baseline_authorization', 'awaiting_training_authorization',
          'awaiting_candidate_authorization', 'awaiting_evaluation'}
ALLOWED = {'run/job.json', 'state.json', 'run/metrics.jsonl',
           'run/training_metrics.json', 'run/trainer_state_final.json',
           'job.json', 'run/data-preflight.json', 'run/model-report.json'}


def read(folder, relative):
    if relative not in ALLOWED:
        raise ValueError('invalid_path')
    path = folder / relative
    if folder.is_symlink() or (folder / 'run').is_symlink():
        raise ValueError('unsafe_path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
            raise ValueError('invalid_file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            return stream.read(8 * 1024 * 1024 + 1)
    finally:
        os.close(fd)


def optional_json(folder, name, reader=read):
    try:
        value = json.loads(reader(folder, name))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def token(value, pattern=r'[A-Za-z0-9_.-]{1,160}'):
    return value if isinstance(value, str) and re.fullmatch(pattern, value) else None


def metadata(folder, identity, reader=read):
    result = {'model_repo': 'Qwen/Qwen3.8-27B'}
    if identity.get('backend') == 'llamafactory':
        result['training_backend'] = 'LLaMA-Factory'
        if token(identity.get('llamafactory_commit'), r'[a-f0-9]{40}'):
            result['llamafactory_commit'] = identity['llamafactory_commit']
        cfg = identity.get('training_config', {})
        if isinstance(cfg, dict):
            result['backend_config'] = {k: cfg[k] for k in
                ('template', 'optim', 'lr_scheduler_type', 'per_device_train_batch_size',
                 'gradient_accumulation_steps', 'weight_decay', 'max_grad_norm', 'save_steps',
                 'bf16', 'gradient_checkpointing', 'train_on_prompt', 'mask_history')
                if k in cfg and type(cfg[k]) in (str, int, float, bool)}
    for key in ('max_length', 'rank', 'seed'):
        value = identity.get(key)
        if type(value) is int and 0 <= value <= 10000000:
            result[key] = value
    lr = identity.get('learning_rate')
    if type(lr) in (int, float) and math.isfinite(lr) and 0 < lr <= 1:
        result['learning_rate'] = lr
    for key in ('model_manifest_sha256', 'data_sha256'):
        if token(identity.get(key), r'[a-f0-9]{64}'):
            result[key] = identity[key]
    recipes = identity.get('recipe_sha256', {})
    if isinstance(recipes, dict):
        result['recipe_sha256'] = {key: recipes[key] for key in
            ('train.py', 'model.py', 'state.py', 'provenance.py', 'knowledge_data.py')
            if token(recipes.get(key), r'[a-f0-9]{64}')}
    job = optional_json(folder, 'job.json', reader)
    for key in ('image_id', 'freeze_id', 'knowledge_freeze_id'):
        pattern = r'sha256:[a-f0-9]{64}' if key == 'image_id' else r'[A-Za-z0-9_.-]{1,160}'
        if token(job.get(key), pattern):
            result[key] = job[key]
    if job.get('orchestrator') in ('tekton', 'brain', 'manual'):
        result['orchestrator'] = job['orchestrator']
    if job.get('telemetry') == 'swanlab-native-v1':
        result['telemetry'] = 'swanlab-native-v1'
    preflight = optional_json(folder, 'run/data-preflight.json', reader)
    data = {}
    for key in ('examples', 'dropped_overlength'):
        value = preflight.get(key)
        if type(value) is int and 0 <= value <= 1000000000:
            data[key] = value
    families = preflight.get('families')
    if isinstance(families, list):
        data['families'] = sorted({x for x in families[:1000] if token(x, r'[a-z][a-z0-9_]{0,63}')})
    result['data_preflight'] = data
    report = optional_json(folder, 'run/model-report.json', reader)
    model = {}
    for key in ('architecture', 'bitsandbytes'):
        if token(report.get(key)):
            model[key] = report[key]
    for key in ('trainable_parameters', 'quantized_modules'):
        value = report.get(key)
        if type(value) is int and 0 <= value <= 1000000000000:
            model[key] = value
    targets = report.get('target_modules')
    if isinstance(targets, list):
        # Publish module types and count, not arbitrary report text or full paths.
        permitted = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
                     'out_proj', 'gate_proj', 'up_proj', 'down_proj'}
        safe = [x for x in targets if isinstance(x, str) and
                re.fullmatch(r'[A-Za-z0-9_.]{1,250}', x) and x.rsplit('.', 1)[-1] in permitted]
        model['target_module_count'] = len(safe)
        model['target_module_types'] = sorted({x.rsplit('.', 1)[-1] for x in safe})
    result['model_report'] = model
    return result


def observe_device():
    """Current host observation, explicitly not historical run hardware evidence."""
    result = {'observed_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'current_source_host_observation',
              'historical_training_hardware_verified': False,
              'cpu_logical_count': os.cpu_count(), 'architecture': platform.machine(),
              'training_gpu_index': 0, 'host_gpu_count': None, 'training_gpu': None}
    try:
        match = re.search(r'^MemTotal:\s+(\d+) kB$', Path('/proc/meminfo').read_text(), re.M)
        if match:
            result['host_memory_bytes'] = int(match[1]) * 1024
    except OSError:
        pass
    try:
        output = subprocess.run(['/usr/bin/querygpu',
            '--query-gpu=index,name,memory.total,driver_version',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=5, check=True).stdout
        rows = [line.split(',') for line in output.splitlines() if line.strip()]
        parsed = []
        for row in rows:
            if len(row) != 4:
                raise ValueError('invalid_gpu_observation')
            index, name, memory, driver = [x.strip() for x in row]
            if not (index.isdigit() and token(name, r'[A-Za-z0-9 ()_.-]{1,100}') and
                    token(driver, r'[0-9.]{1,30}') and memory.isdigit()):
                raise ValueError('invalid_gpu_observation')
            parsed.append({'index': int(index), 'name': name,
                           'memory_total_mib': int(memory), 'driver_version': driver})
        result['host_gpu_count'] = len(parsed)
        result['training_gpu'] = next((x for x in parsed if x['index'] == 0), None)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def collect(folder, reader=read):
    read = reader
    raw = read(folder, 'run/job.json')
    identity = json.loads(raw)
    binding = hashlib.sha256(raw).hexdigest()
    if identity.get('model_revision') != REVISION:
        raise ValueError('invalid_revision')
    max_steps = identity.get('max_steps')
    if type(max_steps) is not int or not 1 <= max_steps <= 100000:
        raise ValueError('invalid_max_steps')
    dataset = identity.get('dataset_id')
    if not isinstance(dataset, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', dataset):
        raise ValueError('invalid_dataset_id')
    phase = json.loads(read(folder, 'state.json')).get('phase')
    if phase not in PHASES:
        raise ValueError('invalid_phase')
    try:
        metrics = read(folder, 'run/metrics.jsonl')
    except FileNotFoundError:
        metrics = b''
    # The writer flushes a whole JSON line. Never interpret a partial trailing line.
    lines = metrics.split(b'\n')[:-1]
    steps = []
    events = []
    for expected, line in enumerate(lines, 1):
        row = json.loads(line)
        if type(row.get('step')) is not int or row['step'] != expected or expected > max_steps:
            raise ValueError('invalid_step_sequence')
        values = {key: row.get(key) for key in ('loss', 'gradient_norm', 'peak_allocated_bytes')}
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise ValueError('invalid_scalar')
        if values['gradient_norm'] <= 0 or values['peak_allocated_bytes'] < 0:
            raise ValueError('invalid_scalar')
        event = {'event': 'training_step', 'step': expected, **values}
        seconds = row.get('seconds')
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            values['seconds'] = seconds
            event['seconds'] = seconds
        checkpoint = row.get('checkpoint')
        if token(checkpoint, r'checkpoint-[0-9]{1,9}'):
            event['checkpoint'] = checkpoint
        steps.append({'step': expected, **values})
        events.append(event)
    final_records = []
    for name in ('training_metrics.json', 'trainer_state_final.json'):
        try:
            final = json.loads(read(folder, 'run/' + name))
        except FileNotFoundError:
            continue
        if final.get('job_sha256') != binding:
            raise ValueError('invalid_final_binding')
        step = final.get('global_step')
        if type(step) is not int or not 0 <= step <= len(steps):
            raise ValueError('invalid_final_step')
        final_records.append(final)
    complete = len(final_records) == 2 and all(r['global_step'] == max_steps for r in final_records) and len(steps) == max_steps
    if phase == 'complete' and not complete:
        raise ValueError('incomplete_terminal_metrics')
    if raw != read(folder, 'run/job.json'):
        raise ValueError('identity_changed')
    return {'job_id': folder.name, 'model_revision': REVISION, 'dataset_id': dataset,
            'job_sha256': binding, 'max_steps': max_steps, 'phase': phase,
            'training_complete': complete, 'steps': steps, 'events': events,
            'metadata_version': 2, 'metadata': metadata(folder, identity, reader)}


def snapshot(root=ROOT):
    result = {'schema_version': 1, 'metadata_version': 2,
              'device_observation': observe_device(), 'jobs': [], 'rejected': []}
    for folder in sorted(root.glob('l20-*')):
        if not re.fullmatch(r'l20-[a-f0-9]{24}', folder.name):
            continue
        try:
            result['jobs'].append(collect(folder))
        except FileNotFoundError:
            # Identity is not created until training starts; no cloud run yet.
            continue
        except (ValueError, KeyError, TypeError, OSError, UnicodeError):
            # Never emit exception text: parser errors can contain source text.
            result['rejected'].append({'job_id': folder.name, 'reason': 'invalid_snapshot'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), allow_nan=False, separators=(',', ':')))
