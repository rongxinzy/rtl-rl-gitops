"""Read only whitelisted L20 identity and scalar metric files; emit no raw errors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
ROOT = Path('/mnt/data/rtl-l20-training/worker/jobs')
PHASES = {'queued', 'baseline', 'training', 'candidate', 'complete', 'failed',
          'cancelled', 'queued_baseline', 'queued_training', 'queued_candidate',
          'awaiting_baseline_authorization', 'awaiting_training_authorization',
          'awaiting_candidate_authorization', 'awaiting_evaluation'}
ALLOWED = {'run/job.json', 'state.json', 'run/metrics.jsonl',
           'run/training_metrics.json', 'run/trainer_state_final.json'}


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


def collect(folder):
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
    for expected, line in enumerate(lines, 1):
        row = json.loads(line)
        if type(row.get('step')) is not int or row['step'] != expected or expected > max_steps:
            raise ValueError('invalid_step_sequence')
        values = {key: row.get(key) for key in ('loss', 'gradient_norm', 'peak_allocated_bytes')}
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise ValueError('invalid_scalar')
        if values['gradient_norm'] <= 0 or values['peak_allocated_bytes'] < 0:
            raise ValueError('invalid_scalar')
        steps.append({'step': expected, **values})
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
            'training_complete': complete, 'steps': steps}


def snapshot(root=ROOT):
    result = {'schema_version': 1, 'jobs': [], 'rejected': []}
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
