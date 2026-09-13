#!/usr/bin/env python3
"""Idempotent local reconciliation; all scheduling times are Asia/Shanghai."""
import datetime as dt
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path('/root/rtl-rl')
STATE = ROOT / 'state'
GLM = 'sglang-glm53'
TRAIN = 'rtl-night-training'

def command(args, timeout=15):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return 124, ''

def schedule(now):
    local = now.astimezone(ZoneInfo('Asia/Shanghai'))
    minute = local.hour * 60 + local.minute
    night = minute >= 1350 or minute < 450
    day = local.date() if minute >= 1350 else local.date() - dt.timedelta(days=1)
    return ('night' if night else 'day'), str(day)

def read_state():
    try:
        return json.loads((STATE / 'scheduler.json').read_text())
    except FileNotFoundError:
        return {}

def write_state(state):
    tmp = STATE / 'scheduler.json.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(tmp, STATE / 'scheduler.json')

def gpu():
    rc, out = command(['/usr/bin/querygpu', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    if rc:
        raise RuntimeError('GPU query failed; refusing transition')
    used = [int(x.strip()) for x in out.splitlines()]
    if len(used) != 8:
        raise RuntimeError('Expected eight GPUs; refusing transition')
    rc, out = command(['/usr/bin/querygpu', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if rc:
        raise RuntimeError('GPU process query failed; refusing transition')
    return used, set(int(x.strip()) for x in out.splitlines() if x.strip())

def running(name):
    rc, out = command(['docker', 'inspect', '-f', '{{.State.Running}}', name])
    return rc == 0 and out == 'true'

def healthy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def other_gpu_containers(allowed=GLM):
    rc, out = command(['docker', 'ps', '-q'])
    if rc:
        raise RuntimeError('Cannot inspect running containers')
    for cid in out.splitlines():
        rc, raw = command(['docker', 'inspect', '--format', '{{.Name}} {{json .HostConfig.DeviceRequests}}', cid])
        if rc:
            raise RuntimeError('Cannot inspect running container')
        name, requests = raw.split(' ', 1)
        if name.lstrip('/') != allowed and json.loads(requests):
            return True
    return False

def reboot_glm(s, window, pids):
    if s.get('glm_stop_window') != window or not pids.issubset(set(s.get('glm_pids', []))) or other_gpu_containers():
        raise RuntimeError('Unowned GPU allocation; refusing reboot or training')
    if s.get('reboot_window') == window:
        raise RuntimeError('Recovery reboot already attempted this night')
    s.update(reboot_window=window, phase='rebooting_after_glm_stop')
    write_state(s)
    rc, _ = command(['systemctl', 'reboot', '--no-block'])
    if rc:
        raise RuntimeError('Recovery reboot request failed')
    return s

def reboot_training(s, window, pids):
    if 'stop_requested' not in s or not s.get('training_pids') or not pids.issubset(set(s.get('training_pids', []))) or other_gpu_containers(TRAIN):
        raise RuntimeError('Unowned GPU allocation; refusing training recovery reboot')
    if s.get('training_reboot_window') == window:
        raise RuntimeError('Training recovery reboot already attempted this night')
    s.update(training_reboot_window=window, phase='rebooting_after_training_stop')
    write_state(s)
    rc, _ = command(['systemctl', 'reboot', '--no-block'])
    if rc:
        raise RuntimeError('Recovery reboot request failed')
    return s

def container_pids(name):
    rc, out = command(['docker', 'top', name, '-eo', 'pid'])
    if rc:
        return set()
    return {int(x.strip()) for x in out.splitlines()[1:] if x.strip().isdigit()}

def reconcile(s, now, requested_mode=None):
    mode, window = schedule(now)
    if requested_mode is not None:
        if requested_mode not in ("night", "day"): raise ValueError("Invalid requested mode")
        mode = requested_mode
    stamp = now.timestamp()
    s.update(desired=mode, window=window, updated=now.isoformat())
    if (STATE / 'scheduler-hold').exists():
        s['phase'] = 'held'
        return s
    # Runner atomically publishes this only when the configured job is complete.
    job = json.loads((ROOT / 'scheduling/job.json').read_text())
    job_id = job['job_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', job_id):
        raise RuntimeError('Invalid configured job identifier')
    if (ROOT / 'runs' / job_id / 'job-complete.json').exists() and not (ROOT / 'runs' / job_id / 'evaluation-pending.json').exists():
        mode = 'day'
        s['idle_after_completion'] = True
    else:
        s.pop('idle_after_completion', None)
    if s.get('failed_window') == window:
        mode = 'day'
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else 'test'
    if mode == 'night':
        if running(GLM):
            used, pids = gpu()
            owned = container_pids(GLM)
            if s.get('glm_stop_window') == window and stamp - s.get('glm_stop_started', stamp) >= 120:
                if max(used) > 512:
                    return reboot_glm(s, window, pids)
                raise RuntimeError('GLM container stop stalled but GPUs are free; reboot unnecessary')
            if not pids or not pids.issubset(owned):
                raise RuntimeError('Cannot attribute every GPU process to GLM; refusing stop')
            if s.get('glm_stop_window') != window:
                s['glm_stop_started'] = stamp
            s.update(phase='stopping_glm', glm_stop_window=window, glm_pids=sorted(pids))
            write_state(s)  # Must precede potentially stalled stop or reboot.
            command(['docker', 'stop', '--time', '30', GLM], timeout=40)
            return s
        if running(TRAIN):
            _, pids = gpu()
            owned = container_pids(TRAIN)
            if pids.issubset(owned):
                s['training_pids'] = sorted(pids)
            rc, active = command(['systemctl', 'show', 'rtl-night-training.service', '-p', 'ActiveState', '--value'])
            if rc == 0 and active in ('failed', 'inactive'):
                s.update(failed_window=window, phase='job_failed_restoring_glm')
                s.setdefault('stop_requested', stamp)
                return s
            s['phase'] = 'training'
            return s
        if s.get('started_window') == window and s.get('training_boot_id') != boot_id:
            if s.get('resume_window') == window:
                raise RuntimeError('Automatic resume already attempted this night')
            s['resume_window'] = window
            s.pop('started_window', None)
        if s.get('started_window') == window:
            rc, status = command(['systemctl', 'show', 'rtl-night-training.service', '-p', 'ActiveState', '--value'])
            s['phase'] = 'training_service_' + status if rc == 0 else 'training_service_unknown'
            if rc == 0 and status in ('failed', 'inactive'):
                s.update(failed_window=window, phase='job_failed_restoring_glm')
                s.setdefault('stop_requested', stamp)
            return s
        used, pids = gpu()
        if max(used) > 512:
            # Only recover the known GLM stop. Never reboot unowned GPU jobs.
            return reboot_glm(s, window, pids)
        if pids:
            raise RuntimeError('GPU processes present; refusing training')
        (STATE / 'night-stop').unlink(missing_ok=True)
        s.pop('stop_requested', None)
        s.pop('training_pids', None)
        s.update(started_window=window, training_boot_id=boot_id, phase='starting_training')
        write_state(s)  # At most once, even if coordinator dies after systemctl start.
        rc, _ = command(['systemctl', 'start', '--no-block', 'rtl-night-training.service'])
        if rc:
            raise RuntimeError('Training service start failed; manual inspection required')
        return s
    # Morning: request checkpoint before stopping the training service.
    rc, active = command(['systemctl', 'show', 'rtl-night-training.service', '-p', 'ActiveState', '--value'])
    if running(TRAIN) or active in ('active', 'activating', 'deactivating'):
        if 'stop_requested' not in s:
            _, pids = gpu()
            owned = container_pids(TRAIN)
            if pids and not pids.issubset(owned):
                raise RuntimeError('Cannot attribute GPU processes to training; refusing recovery')
            s['training_pids'] = sorted(pids)
        (STATE / 'night-stop').touch()
        s.setdefault('stop_requested', stamp)
        s['phase'] = 'checkpointing'
        if stamp - s['stop_requested'] >= 720:
            used, pids = gpu()
            if max(used) > 512:
                return reboot_training(s, window, pids)
            raise RuntimeError('Training service stop stalled but GPUs are free; reboot unnecessary')
        if stamp - s['stop_requested'] >= 600:
            command(['docker', 'stop', '--time', '30', TRAIN], timeout=40)
            command(['systemctl', 'stop', '--no-block', 'rtl-night-training.service'])
            s['phase'] = 'training_stop_timeout'
        return s
    if running(GLM):
        if healthy():
            s['phase'] = 'glm_ready'
            s.pop('deadline_missed', None)
            s.pop('glm_loading_started', None)
            s.pop('stop_requested', None)
        else:
            s.setdefault('glm_loading_started', stamp)
            s['phase'] = 'glm_loading_timeout' if stamp - s['glm_loading_started'] > 900 else 'glm_loading'
            local = now.astimezone(ZoneInfo('Asia/Shanghai'))
            if 8 <= local.hour < 22:
                s['deadline_missed'] = True
        return s
    used, pids = gpu()
    if max(used) > 512 or pids:
        if s.get('stop_requested') is not None and stamp - s['stop_requested'] >= 600:
            return reboot_training(s, window, pids)
        raise RuntimeError('GPU occupancy blocks GLM restore; manual inspection required')
    rc, _ = command(['docker', 'start', GLM], timeout=20)
    if rc:
        raise RuntimeError('GLM start failed')
    s['phase'] = 'glm_loading'
    s['glm_loading_started'] = stamp
    return s

def main():
    action = os.environ.get('SSH_ORIGINAL_COMMAND', ' '.join(sys.argv[1:]))
    if action not in ('status', 'reconcile'):
        raise SystemExit('Only status or reconcile is permitted')
    if action == 'status':
        print(json.dumps(read_state()))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'phase': 'reconcile_busy'}))
            return
        s = read_state()
        s.pop('error', None)
        try:
            now = dt.datetime.now(dt.timezone.utc)
            override = None
            if (STATE / 'operator-enabled').exists():
                try:
                    authority = json.loads((STATE / 'operator-authority.json').read_text())
                except (OSError, ValueError):
                    authority = {}
                live = authority.get('valid_until', 0) > now.timestamp()
                expired = authority.get('mode') == 'training' and authority.get('deadline', 0) <= now.timestamp()
                if live and not expired:
                    print(json.dumps({**s, 'authority': 'operator', 'failsafe': 'armed'}))
                    return
                override = 'day'  # Control-plane loss may stop training; never starts it.
                s['failsafe_reason'] = 'operator_deadline' if expired else 'operator_heartbeat_expired'
            s = reconcile(s, now, requested_mode=override)
        except Exception as exc:
            s.update(phase='error', error=str(exc))
        write_state(s)
        print(json.dumps(s))

if __name__ == '__main__':
    main()
