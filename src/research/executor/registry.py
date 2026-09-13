"""Transactional upgrade of the single fixed CPU judge service."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from .core import atomic


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Fixed judge service operation failed')
    return result.stdout.strip()


def registry_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get('split') != 'train' for row in rows):
        raise ValueError('Only nonempty train registry is admissible')
    return rows


def upgrade(executor, dataset):
    directory = executor.state / 'registries'
    directory.mkdir(exist_ok=True)
    pointer = directory / 'active.json'
    old_pointer = pointer.read_bytes() if pointer.exists() else None
    prior = json.loads(pointer.read_text())['path'] if pointer.exists() else str(executor.root / 'processed/bootstrap-v2/registry_train.jsonl')
    new_path = (executor.root / dataset['registry']).resolve()
    if not new_path.is_relative_to(executor.root / 'research/data_factory') or hashlib.sha256(new_path.read_bytes()).hexdigest() != dataset['registry_sha256']:
        raise ValueError('Candidate registry integrity failure')
    merged = {}
    specs = {}
    for row in registry_rows(Path(prior)) + registry_rows(new_path):
        ident, spec = row['task_id'], row['spec']
        if ident in merged and merged[ident] != row:
            raise ValueError('Task ID collision')
        if spec in specs and specs[spec] != ident:
            raise ValueError('Spec collision')
        merged[ident], specs[spec] = row, ident
    data = ''.join(json.dumps(merged[k], sort_keys=True) + '\n' for k in sorted(merged))
    sha = hashlib.sha256(data.encode()).hexdigest()
    registry = directory / (sha + '.jsonl')
    if registry.exists() and registry.read_text() != data:
        raise ValueError('Immutable registry mismatch')
    registry.write_text(data)
    dropin = Path('/etc/systemd/system/rtl-judge.service.d/30-research-registry.conf')
    dropin.parent.mkdir(parents=True, exist_ok=True)
    backup = dropin.read_bytes() if dropin.exists() else None
    image = command(['docker', 'image', 'inspect', 'rtl-judge:local', '--format', '{{.Id}}'])
    if not image.startswith('sha256:') or len(image) != 71:
        raise ValueError('Cannot pin judge image')
    content = '[Service]\nExecStart=\nExecStart=/usr/bin/python3 -u ' + str(executor.root / 'scripts/judge_service.py') + ' --registry ' + str(registry) + ' --cache ' + str(executor.root / 'reports/judge-cache') + ' --image ' + image + '\n'
    (directory / (sha + '.previous-unit')).write_bytes(backup or b'')
    atomic(directory / 'pending.json', {'registry': str(registry), 'prior': prior, 'had_dropin': backup is not None, 'backup': str(directory / (sha + '.previous-unit'))})
    def restore():
        if old_pointer is None:
            pointer.unlink(missing_ok=True)
        else:
            pointer.write_bytes(old_pointer)
        if backup is None:
            dropin.unlink(missing_ok=True)
        else:
            dropin.write_bytes(backup)
        command(['systemctl', 'daemon-reload'])
        command(['systemctl', 'restart', 'rtl-judge.service'])
    try:
        temp = dropin.with_suffix('.tmp');temp.write_text(content);os.replace(temp, dropin)
        command(['systemctl', 'daemon-reload'])
        command(['systemctl', 'restart', 'rtl-judge.service'])
        good = False
        for _ in range(10):
            try:
                with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2) as response:
                    health = json.load(response)
                with urllib.request.urlopen('http://127.0.0.1:8765/tasks', timeout=2) as response:
                    actual = {row['task_id']: row for row in json.load(response)['tasks']}
                good = health['status'] == 'ok' and health['image'] == image and set(actual) == set(merged)
                good = good and all(actual[k]['split'] == 'train' and actual[k]['spec_sha256'] == hashlib.sha256(v['spec'].encode()).hexdigest() for k, v in merged.items())
                if good:
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1)
        if not good:
            raise RuntimeError('Judge registry health verification failed')
        atomic(pointer, {'path': str(registry), 'sha256': sha})
        (directory / 'pending.json').unlink(missing_ok=True)
        return restore
    except Exception:
        restore()
        raise
