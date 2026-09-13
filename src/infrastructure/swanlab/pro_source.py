"""Pro LF metrics only: no training data, raw logs, credentials or arbitrary paths."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import source

ROOT = Path('/root/rtl-rl/runs')
ALLOWED = {'job-config.json', 'train-run/job.json', 'train-run/metrics.jsonl',
           'train-run/training_metrics.json', 'train-run/trainer_state_final.json',
           'train-run/data-preflight.json', 'train-run/model-report.json',
           'train-run/status.json', 'last-start.json', 'last-exit.json'}


def read(folder, relative):
    if relative not in ALLOWED:
        raise ValueError('invalid_path')
    if folder.is_symlink() or (folder / 'train-run').is_symlink():
        raise ValueError('unsafe_path')
    fd = os.open(folder / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
            raise ValueError('invalid_file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            return stream.read(8 * 1024 * 1024 + 1)
    finally:
        os.close(fd)


def lifecycle(folder, binding, steps, complete):
    records = {}
    for name in ('last-start.json', 'last-exit.json'):
        try:
            record = json.loads(read(folder, name))
        except FileNotFoundError:
            continue
        if not isinstance(record, dict):
            raise ValueError('invalid_lifecycle_record')
        stamp = record.get('time')
        if record.get('backend') != 'llamafactory' or type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0 or stamp > time.time() + 60:
            raise ValueError('invalid_lifecycle_record')
        records[name] = record
    start = records.get('last-start.json')
    exit_record = records.get('last-exit.json')
    # No current start means an old exit cannot describe an active attempt.
    result = {'phase': 'complete' if complete else 'queued_training'}
    if start is None:
        return result
    result.update(phase='complete' if complete else 'training', attempt_started_at=start['time'])
    if exit_record is None or exit_record['time'] < start['time']:
        return result
    code = exit_record.get('returncode')
    exit_step = exit_record.get('step')
    if type(code) is not int or not -255 <= code <= 255:
        raise ValueError('invalid_exit_code')
    if (code == 0 or exit_step is not None) and (type(exit_step) is not int or not 0 <= exit_step <= steps):
        raise ValueError('invalid_exit_step')
    result['attempt_finished_at'] = exit_record['time']
    result['returncode'] = code
    if code != 0:
        result['phase'] = 'failed'
        return result
    status = json.loads(read(folder, 'train-run/status.json'))
    if status.get('job_sha256') != binding or type(status.get('step')) is not int or status['step'] != exit_step or status.get('status') not in ('paused', 'complete'):
        raise ValueError('invalid_exit_status_binding')
    if complete and status['status'] == 'complete':
        result['phase'] = 'complete'
    elif not complete and status['status'] == 'paused':
        result['phase'] = 'paused'
    else:
        raise ValueError('inconsistent_exit_status')
    return result


def collect(folder, device=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', folder.name):
        raise ValueError('invalid_job_id')
    cfg_raw = read(folder, 'job-config.json')
    cfg = json.loads(cfg_raw)
    identity = json.loads(read(folder, 'train-run/job.json'))
    if cfg.get('job_id') != folder.name or cfg.get('backend') != 'llamafactory' or identity.get('backend') != 'llamafactory':
        raise ValueError('invalid_backend_binding')
    for key in ('model_revision', 'dataset_id', 'max_steps', 'data_sha256'):
        if key not in cfg or cfg[key] != identity.get(key):
            raise ValueError('invalid_job_binding')
    commit = '100e9a42c6c09f8f7849b70d60f3da445fb2024b'
    if cfg.get('llamafactory_commit') != commit or identity.get('llamafactory_commit') != commit or cfg.get('max_length') != 1024 or cfg.get('lora_rank') != 8 or identity.get('max_length') != 1024 or identity.get('rank') != 8:
        raise ValueError('invalid_recipe_version')
    expected = cfg.get('recipe_sha256', {})
    required = {'train.py', 'model.py', 'state.py', 'provenance.py', 'knowledge_data.py'}
    if not isinstance(expected, dict) or set(expected) != required | {'generate_eval.py'} or any(not source.token(v, r'[a-f0-9]{64}') for v in expected.values()) or identity.get('recipe_sha256') != {k: expected[k] for k in required}:
        raise ValueError('invalid_recipe_binding')
    if not source.token(cfg.get('image_id'), r'sha256:[a-f0-9]{64}'):
        raise ValueError('invalid_runtime_binding')
    def mapped_read(_, relative):
        if relative == 'job.json':
            return cfg_raw
        if relative == 'state.json':
            # Completion is derived exclusively from matching final metrics below.
            return b'{"phase":"training"}'
        if relative.startswith('run/'):
            return read(folder, 'train-run/' + relative.removeprefix('run/'))
        raise ValueError('invalid_path')
    result = source.collect(folder, reader=mapped_read)
    if cfg_raw != read(folder, 'job-config.json'):
        raise ValueError('identity_changed')
    result.update(source='pro6000d', source_stale=False, source_observed_at=time.time(),
                  device_observation=device or {})
    state = lifecycle(folder, result['job_sha256'], len(result['steps']), result['training_complete'])
    result.update(state)
    if result['phase'] == 'failed':
        result['training_complete'] = False
    result['metadata'].update(training_host='pro6000D',
                              job_config_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                              training_backend='LLaMA-Factory')
    for key in ('evaluation_freeze_id', 'baseline_evaluation_id'):
        if source.token(cfg.get(key), r'[a-f0-9]{64}'):
            result['metadata'][key] = cfg[key]
    return result


def snapshot(root=ROOT):
    device = source.observe_device()
    result = {'schema_version': 1, 'jobs': [], 'rejected': []}
    for folder in sorted(root.iterdir()) if root.exists() else []:
        if not folder.is_dir() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', folder.name):
            continue
        try:
            cfg = json.loads(read(folder, 'job-config.json'))
            if cfg.get('backend') != 'llamafactory':
                continue
            result['jobs'].append(collect(folder, device))
        except FileNotFoundError:
            continue  # No cloud run until train-run identity exists.
        except (ValueError, KeyError, TypeError, OSError, UnicodeError):
            result['rejected'].append({'job_id': folder.name, 'reason': 'invalid_pro_snapshot'})
    return result
