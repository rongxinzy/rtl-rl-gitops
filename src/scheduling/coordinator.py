#!/usr/bin/env python3
"""External observer/reconciler; target also has an independent local timer."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/opt/rtl-scheduler')
def main():
    result = {'updated': dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        p = subprocess.run(['ssh', '-i', str(ROOT / 'id_ed25519'),
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(ROOT / 'known_hosts'),
            'root@172.18.4.199', 'reconcile'], text=True, capture_output=True, timeout=80)
        result['returncode'] = p.returncode
        if p.returncode == 0:
            result['target'] = json.loads(p.stdout)
        else:
            result['status'] = 'target_unreachable_or_restarting'
            result['error'] = p.stderr[-1000:]
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        result.update(status='coordinator_error', error=str(exc))
    temp = ROOT / 'status.json.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temp, ROOT / 'status.json')
    print(json.dumps(result))
if __name__ == '__main__':
    main()
