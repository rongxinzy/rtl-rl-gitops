"""Pull a forced-command, scalar-only snapshot; never execute arbitrary SSH input."""
import json
from pathlib import Path
import subprocess

root = Path('/root/rtl-rl/swanlab-relay')
result = subprocess.run(['ssh', '-T', '-i', '/root/rtl-rl/secrets/metrics_ssh',
    '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=/root/rtl-rl/secrets/metrics_known_hosts',
    'root@172.18.6.123'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=25)
if result.returncode or len(result.stdout) > 8 * 1024 * 1024:
    raise SystemExit('scalar_source_unavailable')
data = json.loads(result.stdout)
if data.get('schema_version') != 1:
    raise SystemExit('scalar_schema_invalid')
root.mkdir(parents=True, exist_ok=True)
tmp = root / 'snapshot.tmp'
tmp.write_text(json.dumps(data))
tmp.replace(root / 'snapshot.json')
