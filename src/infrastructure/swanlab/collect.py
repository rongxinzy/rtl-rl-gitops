"""Merge independent scalar-only sources; one unavailable host cannot block another."""
import copy
import json
from pathlib import Path
import subprocess
import time
import pro_source

ROOT = Path('/root/rtl-rl/swanlab-relay')


def pull_l20():
    result = subprocess.run(['ssh', '-T', '-i', '/root/rtl-rl/secrets/metrics_ssh',
        '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=/root/rtl-rl/secrets/metrics_known_hosts',
        'root@172.18.5.123'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=25)
    if result.returncode or len(result.stdout) > 8 * 1024 * 1024:
        raise ValueError('scalar_source_unavailable')
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('jobs'), list):
        raise ValueError('scalar_schema_invalid')
    return data


def merge(previous, l20, pro):
    now = time.time()
    result = {'schema_version': 1, 'metadata_version': 2, 'jobs': [], 'rejected': [], 'sources': {}}
    for name, fresh in (('l20', l20), ('pro6000d', pro)):
        if fresh is None:
            jobs = [copy.deepcopy(j) for j in previous.get('jobs', []) if j.get('source', 'l20') == name]
            for job in jobs:
                job['source_stale'] = True
                job['source'] = name
            result['sources'][name] = {'stale': True}
        else:
            jobs = copy.deepcopy(fresh.get('jobs', []))
            for job in jobs:
                job.update(source=name, source_stale=False, source_observed_at=now)
                if 'device_observation' not in job:
                    job['device_observation'] = copy.deepcopy(fresh.get('device_observation', {}))
            result['rejected'].extend(fresh.get('rejected', []))
            result['sources'][name] = {'stale': False, 'observed_at': now}
        result['jobs'].extend(jobs)
    return result


def main(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads((root / 'snapshot.json').read_text())
    except (OSError, ValueError):
        previous = {}
    try:
        pro = pro_source.snapshot()
    except (OSError, ValueError, TypeError):
        pro = None
    try:
        l20 = pull_l20()
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        l20 = None
    data = merge(previous, l20, pro)
    tmp = root / 'snapshot.tmp'
    tmp.write_text(json.dumps(data, allow_nan=False))
    tmp.replace(root / 'snapshot.json')


if __name__ == '__main__':
    main()
