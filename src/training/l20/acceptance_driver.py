"""Continue the fixed GPU acceptance sequence; never enable the training worker."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/data/rtl-l20-training')
OUT = ROOT / 'acceptance'
IMAGE = 'sha256:6436ae9c5ced1ee5b6053b5516f343a437e270219904a45a9ccfdd1db8bb8cc3'

def record(value):
    temp = OUT / 'driver-status.tmp'
    temp.write_text(json.dumps({'time': time.time(), **value}, indent=2))
    temp.replace(OUT / 'driver-status.json')

def inspect(name):
    p = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=20)
    return json.loads(p.stdout)[0] if p.returncode == 0 else None

def run_phase(name, command=None):
    record({'phase': name, 'status': 'running'})
    instance = inspect(name)
    if instance is None:
        if command is None:
            raise RuntimeError('initial acceptance container missing')
        model = json.loads((ROOT / 'nf4-cache/MODEL_READY.json').read_text())['path']
        args = ['docker', 'run', '-d', '--name', name, '--network', 'none', '--gpus', 'device=0',
                '--cpus', '8', '--memory', '24g', '--memory-swap', '24g', '--shm-size', '2g',
                '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
                '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                '--mount', 'type=bind,src='+model+',dst=/model,readonly',
                '--mount', 'type=bind,src='+str(OUT)+',dst=/acceptance', IMAGE, 'python3'] + command
        subprocess.run(args, check=True, timeout=60)
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        instance = inspect(name)
        if instance is None or instance['Image'] != IMAGE:
            raise RuntimeError('acceptance container identity changed')
        if not instance['State']['Running']:
            logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True, timeout=30)
            (OUT / (name+'.log')).write_text(logs.stdout + logs.stderr)
            if instance['State']['ExitCode'] or instance['State']['OOMKilled']:
                raise RuntimeError(name+' failed; inspect saved log')
            return
        time.sleep(10)
    raise RuntimeError(name+' exceeded acceptance time budget; left for explicit inspection')

def main():
    recipe = OUT / 'qlora-recipe-v1/train.py'
    if hashlib.sha256(recipe.read_bytes()).hexdigest() != 'b16eb9acaec7d3c5b52eaac4751689246a0aa281466d6a36e1dbe0e67ec9fc1b':
        raise ValueError('frozen acceptance recipe changed')
    if (ROOT / 'worker/enabled').exists():
        raise ValueError('acceptance requires worker disabled')
    train = ['/acceptance/qlora-recipe-v1/train.py', '--model', '/model', '--data',
             '/acceptance/data/bootstrap-v2-sft.jsonl', '--dataset-id',
             '9c9764fb8ba2a7af69bccee8fb94ae2241eb5f1658ac600d5b513a65610a0b46', '--max-steps', '20']
    run_phase('rtl-l20-accept-step3')
    run_phase('rtl-l20-accept-step4', train + ['--output', '/acceptance/cuda-512',
              '--max-length', '512', '--stop-after', '4', '--resume-from', 'latest'])
    run_phase('rtl-l20-accept-reload', ['/acceptance/qlora-recipe-v1/verify_reload.py',
              '--model', '/model', '--output', '/acceptance/cuda-512'])
    reload = json.loads((OUT / 'cuda-512/reload-verification.json').read_text())
    metrics = json.loads((OUT / 'cuda-512/training_metrics.json').read_text())
    if not reload['passed'] or reload['step'] != 4 or metrics['resumed_from_step'] != 3:
        raise ValueError('resume/reload acceptance mismatch')
    run_phase('rtl-l20-accept-1024', train + ['--output', '/acceptance/cuda-1024',
              '--max-length', '1024', '--stop-after', '1'])
    rows = [json.loads(x) for x in (OUT / 'cuda-512/metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != [1, 2, 3, 4] or any(r['gradient_norm'] <= 0 for r in rows):
        raise ValueError('step/gradient sequence mismatch')
    record({'status': 'passed', 'resume_from': 3, 'resume_to': 4, 'reload': reload,
            'max_length_1024_config': True, 'actual_1024_token_stress': False,
            'worker_enabled': False})

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record({'status': 'failed', 'reason': str(error), 'error_type': type(error).__name__})
        raise
