"""SwanLab scalar relay. Training never depends on this process or its network."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STATE = Path(os.environ.get('RELAY_STATE', '/state'))
SNAPSHOT = Path(os.environ.get('RELAY_SNAPSHOT', '/input/snapshot.json'))


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True))
    tmp.replace(path)


def run_id(job):
    return 'l20-' + hashlib.sha256((job['job_id'] + ':' + job['job_sha256']).encode()).hexdigest()[:24]


def snapshot():
    if time.time() - SNAPSHOT.stat().st_mtime > 180:
        raise ValueError('source_stale')
    data = json.loads(SNAPSHOT.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('snapshot_schema')
    return data


def worker(ident):
    import swanlab
    run = None
    sent = 0
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    record = STATE / (ident + '.json')
    info = {'job_id': ident, 'status': 'starting'}
    try:
        while not stopping[0]:
            data = snapshot()
            job = next(j for j in data['jobs'] if j['job_id'] == ident)
            if run is None:
                config = {k: job[k] for k in ('job_id', 'job_sha256', 'model_revision', 'dataset_id', 'max_steps')}
                config['stage'] = 'l20_qlora_sft'
                config['scope'] = 'scalar telemetry; training loss is not capability evaluation'
                run = swanlab.init(project=os.environ.get('RTL_SWANLAB_PROJECT', 'RTL-RL'), public=False,
                    name=ident, id=run_id(job), resume='allow', mode='online', config=config,
                    log_dir=str(STATE / 'sdk'), settings=swanlab.Settings(interactive=False,
                        terminal={'proxy_type': 'none'}, probe={'hardware': False, 'runtime': False,
                        'requirements': False, 'git': False, 'swanlab': False, 'monitor': False}))
                info.update(run_id=run.id, url=run.url, job_sha256=job['job_sha256'])
            if info['job_sha256'] != job['job_sha256']:
                raise ValueError('identity_changed')
            # Replay on process restart: SDK resume reconciles remote step history.
            # Never skip unsent data based solely on an optimistic local cursor.
            for row in job['steps']:
                if row['step'] > sent:
                    swanlab.log({k: row[k] for k in ('loss', 'gradient_norm', 'peak_allocated_bytes')}, step=row['step'])
                    sent = row['step']
            info.update(status='running', step=sent, observed_at=time.time(), phase=job['phase'])
            atomic(record, info)
            if job['training_complete'] or job['phase'] == 'failed':
                ok = job['training_complete']
                swanlab.finish(state='success' if ok else 'crashed')
                run = None
                info.update(status='completed' if ok else 'failed', sdk_finish_returned=True, finished_at=time.time())
                atomic(record, info)
                return 0
            time.sleep(10)
        if run is not None:
            swanlab.finish(state='aborted')
            run = None
        info.update(status='interrupted', observed_at=time.time())
        atomic(record, info)
        return 1
    except Exception as exc:
        # No exception strings, provider bodies, environment or training data in output.
        info.update(status='retry', error_type=type(exc).__name__, observed_at=time.time())
        atomic(record, info)
        if run is not None:
            try:
                swanlab.finish(state='aborted')
            except Exception:
                pass
        return 1


def supervise():
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'relay.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children, retry = {}, {}
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    while not stopping[0]:
        for ident, proc in list(children.items()):
            if proc.poll() is not None:
                del children[ident]
                retry[ident] = time.time() + 60
        health = {'observed_at': time.time(), 'children': sorted(children)}
        try:
            data = snapshot()
            jobs = sorted(data['jobs'], key=lambda j: j['training_complete'])
            for job in jobs:
                ident = job['job_id']
                if ident in children or time.time() < retry.get(ident, 0) or len(children) >= 2:
                    continue
                path = STATE / (ident + '.json')
                receipt = json.loads(path.read_text()) if path.exists() else {}
                if receipt.get('status') in ('completed', 'failed') and receipt.get('job_sha256') == job['job_sha256']:
                    continue
                children[ident] = subprocess.Popen([sys.executable, __file__, '--worker', ident],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            health.update(status='healthy', jobs=len(jobs), rejected=data.get('rejected', []))
        except Exception as exc:
            health.update(status='source_unavailable', error_type=type(exc).__name__)
        atomic(STATE / 'heartbeat.json', health)
        time.sleep(5)
    for proc in children.values():
        proc.terminate()
    for proc in children.values():
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        sys.exit(worker(sys.argv[2]))
    supervise()
