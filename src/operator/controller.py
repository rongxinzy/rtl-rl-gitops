#!/usr/bin/env python3
"""Small level-triggered controller. No host action occurs without a live Lease."""
import datetime as dt
import json
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc
PROBE_CACHE = {}

def stamp(now):
    return now.astimezone(UTC).isoformat().replace('+00:00', 'Z')

def parse(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))

def desired(spec, now):
    local = now.astimezone(ZoneInfo(spec.get('timezone', 'Asia/Shanghai')))
    def minute(value):
        h, m = map(int, value.split(':'))
        if not 0 <= h <= 23 or not 0 <= m <= 59:
            raise ValueError('Invalid schedule time')
        return h * 60 + m
    start, stop = minute(spec.get('trainingStart', '22:30')), minute(spec.get('trainingStop', '07:30'))
    current = local.hour * 60 + local.minute
    within = (current >= start or current < stop) if start > stop else start <= current < stop
    mode = spec.get('mode', 'Auto')
    # The stop boundary is hard even for a daytime/manual Training override.
    deadline = spec.get('trainingUntil')
    if deadline and now >= parse(deadline):
        return 'Inference'
    if mode == 'Training' and not deadline:
        raise ValueError('Training override requires trainingUntil')
    return ('Training' if within else 'Inference') if mode == 'Auto' else mode


def force_training_due(spec, now):
    """Deadline is relative to the current window, including its next-day tail."""
    if not spec.get('forceTrainingAt') or spec.get('mode', 'Auto') != 'Auto':
        return False
    if desired(spec, now) != 'Training':
        return False
    def minute(value):
        h, m = map(int, value.split(':'))
        if not 0 <= h <= 23 or not 0 <= m <= 59:
            raise ValueError('Invalid schedule time')
        return h * 60 + m
    start = minute(spec.get('trainingStart', '22:30'))
    stop = minute(spec.get('trainingStop', '07:30'))
    cutoff = minute(spec['forceTrainingAt'])
    duration = (stop - start) % 1440
    offset = (cutoff - start) % 1440
    if duration == 0 or offset >= duration:
        raise ValueError('Force deadline must be inside the training window')
    local = now.astimezone(ZoneInfo(spec.get('timezone', 'Asia/Shanghai')))
    elapsed = (local.hour * 60 + local.minute - start) % 1440
    return elapsed >= offset


def backup_window(spec, now):
    """Preparation and overnight coverage share a midnight-safe window."""
    return desired({**spec, 'mode': 'Auto', 'trainingUntil': None,
                    'trainingStart': spec.get('backupPrepareStart', '22:00')}, now) == 'Training'


def training_deadline(spec, now):
    if spec.get('mode', 'Auto') == 'Training':
        end = parse(spec['trainingUntil'])
    else:
        local = now.astimezone(ZoneInfo(spec.get('timezone', 'Asia/Shanghai')))
        hour, minute = map(int, spec.get('trainingStop', '07:30').split(':'))
        end = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if end <= local:
            end += dt.timedelta(days=1)
    delta = (end - now).total_seconds()
    if not 0 < delta <= 86400:
        raise ValueError('Training deadline must be in the next 24 hours')
    return int(end.timestamp())


def request(url, method='GET', body=None, headers=None, timeout=20, context=None, decode_json=True):
    data = None if body is None else json.dumps(body).encode()
    hdr = {'Accept': 'application/json', **(headers or {})}
    if data is not None:
        hdr['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, method=method, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
        raw = response.read(2 * 1024 * 1024)
        return json.loads(raw) if raw and decode_json else {}


class Kubernetes:
    def __init__(self):
        base = Path('/var/run/secrets/kubernetes.io/serviceaccount')
        self.base = 'https://' + os.environ['KUBERNETES_SERVICE_HOST'] + ':' + os.environ.get('KUBERNETES_SERVICE_PORT', '443')
        self.token_file = base / 'token'
        self.context = ssl.create_default_context(cafile=str(base / 'ca.crt'))
        self.namespace = os.environ.get('NAMESPACE', 'rtl-system')
        self.identity = os.environ['POD_NAME']
        self.name = os.environ.get('SCHEDULE_NAME', 'glm-nightly')
        self.resource = f'/apis/rtl.rongxin.ai/v1alpha1/namespaces/{self.namespace}/inferenceschedules/{self.name}'
        self.lease_path = f'/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/rtl-inference-controller'
    def call(self, path, method='GET', body=None):
        return request(self.base + path, method, body, {'Authorization': 'Bearer ' + self.token_file.read_text().strip()}, context=self.context)
    def acquire(self):
        now = dt.datetime.now(UTC)
        lease = self.call(self.lease_path)
        spec = lease.get('spec', {})
        holder = spec.get('holderIdentity')
        renewal = spec.get('renewTime') or spec.get('acquireTime')
        if holder and holder != self.identity and renewal and parse(renewal) + dt.timedelta(seconds=spec.get('leaseDurationSeconds', 120)) > now:
            return False
        lease['spec'] = {**spec, 'holderIdentity': self.identity, 'leaseDurationSeconds': 120, 'renewTime': stamp(now)}
        if holder != self.identity:
            lease['spec'].update(acquireTime=stamp(now), leaseTransitions=spec.get('leaseTransitions', 0) + 1)
        try:
            self.call(self.lease_path, 'PUT', lease)  # resourceVersion is mandatory CAS.
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                return False
            raise
    def status(self, obj, value):
        obj['status'] = value
        self.call(self.resource + '/status', 'PUT', obj)


class Effects:
    def __init__(self, spec, leader):
        self.spec, self.leader = spec, leader
    def guard(self):
        if not self.leader():
            raise RuntimeError('Leadership lost; refusing mutation')
    def router(self, method='GET', body=None):
        token = Path(os.environ.get('ROUTER_TOKEN_FILE', '/secrets/router/token')).read_text().strip()
        return request(self.spec['routerURL'].rstrip('/') + '/admin/' + ('status' if method == 'GET' else 'backend'), method, body, {'Authorization': 'Bearer ' + token})
    def switch(self, backend, force=False):
        self.guard()
        return self.router('PUT', {'backend': backend, **({'force': True} if force else {})})
    def probe(self, backend):
        url = self.spec[backend + 'URL'].rstrip('/')
        key = Path('/secrets/router/' + backend + '-key')
        headers = {'Authorization': 'Bearer ' + key.read_text().strip()} if key.exists() else {}
        try:
            request(url + '/health', headers=headers, timeout=5, decode_json=False)
            if time.monotonic() - PROBE_CACHE.get(url, float('-inf')) < 60:
                return True
            answer = request(url + '/v1/chat/completions', 'POST', {'model': self.spec['model'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}], 'max_tokens': 8, 'stream': False}, headers=headers, timeout=35)
            valid = bool(answer.get('choices')) and bool(answer['choices'][0].get('message'))
            if valid:
                PROBE_CACHE[url] = time.monotonic()
            return valid
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
            PROBE_CACHE.pop(url, None)
            return False
    def l20(self, role=None):
        if role is not None:
            if role not in ('inference', 'training'):
                raise ValueError('Unsupported L20 role')
            self.guard()
        token = Path('/secrets/l20/token').read_text().strip()
        return request('http://172.18.6.123:18766/rotation' + ('/status' if role is None else ''),
                       'GET' if role is None else 'POST', None if role is None else {'role': role},
                       {'Authorization': 'Bearer ' + token}, timeout=8)
    def host(self, action, deadline=None):
        if action != 'status':
            self.guard()
        allowed = {'status', 'training', 'inference'}
        if action not in allowed:
            raise ValueError('Unsupported host action')
        key = Path('/tmp/operator-key')
        if not key.exists():
            key.write_bytes(Path('/secrets/ssh/id_ed25519').read_bytes())
            key.chmod(0o600)
        args = ['ssh', '-i', str(key), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=/secrets/ssh/known_hosts', 'root@172.18.4.199', 'operator-' + action]
        if action == 'training':
            if not isinstance(deadline, int) or not 0 < deadline - time.time() <= 86400:
                raise ValueError('Invalid host training deadline')
            args[-1] += ' ' + str(deadline)
        result = subprocess.run(args, capture_output=True, text=True, timeout=65)
        if result.returncode:
            raise RuntimeError('Host action failed with code ' + str(result.returncode))
        value = json.loads(result.stdout)
        if value.get('error'):
            raise RuntimeError('Host reports: ' + value['error'])
        return value


def reconcile(obj, effects, now):
    spec = obj['spec']
    result = {'observedGeneration': obj['metadata']['generation'], 'updatedAt': stamp(now), 'desiredMode': desired(spec, now), 'phase': 'Observing'}
    host = effects.host('status')
    result['host'] = host
    if spec.get('suspend', False):
        return {**result, 'phase': 'Suspended'}
    if spec.get('controlMode', 'Observe') != 'Active':
        return result
    route = effects.router()
    result['traffic'] = {k:route[k] for k in ('primary_inflight','backup_inflight','business_idle_seconds','last_background_at','protected_inflight','background_inflight','background_rejected_total','healthy_backends') if k in route}
    rotate = spec.get('rotateL20', False)
    night_backup = rotate and spec.get('mode', 'Auto') == 'Auto' and backup_window(spec, now)
    l20 = {}
    if rotate:
        try:
            l20 = effects.l20()
            if night_backup and l20.get('requested_role') != 'inference':
                l20 = effects.l20('inference')
        except (OSError, ValueError, TimeoutError):
            l20 = {'phase': 'unavailable', 'ready': False}
        result['l20'] = {k: l20[k] for k in ('phase', 'ready', 'training_allowed', 'requested_role') if k in l20}
    target = result['desiredMode']
    if host.get('idle_after_completion'):
        target = 'Inference'
    if target == 'Training':
        deadline = training_deadline(spec, now)
        if host.get('job_id') != spec.get('jobId'):
            return {**result, 'phase': 'BlockedJobMismatch'}
        already_running = host.get('phase') in ('training', 'starting_training', 'stopping_glm', 'training_service_active', 'training_service_activating')
        if spec.get('mode', 'Auto') == 'Auto':
            local = now.astimezone(ZoneInfo(spec.get('timezone', 'Asia/Shanghai')))
            hour, minute = map(int, spec.get('trainingStart', '22:30').split(':'))
            window_start = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if window_start > local:
                window_start -= dt.timedelta(days=1)
            # host status may report glm_ready while docker stop is pending.
            # Only this night's durable stop marker commits the transition.
            started = host.get('glm_stop_started')
            committed = (host.get('glm_stop_window') == str(window_start.date())
                         and isinstance(started, (int, float))
                         and window_start.timestamp() <= started <= now.timestamp())
            already_running = already_running or committed
        if not already_running and deadline - now.timestamp() < spec.get('minStartRemainingSeconds', 3600):
            result['host'] = effects.host('inference')
            if route.get('active_backend') == 'maintenance' and result['host'].get('phase') == 'glm_ready':
                effects.switch('primary')
            return {**result, 'phase': 'TooLateToStartTraining'}
        forced = force_training_due(spec, now)
        if rotate:
            # Only a live worker plus an actual model request establishes readiness.
            backup_ready = l20.get('requested_role') == 'inference' and l20.get('ready') is True and effects.probe('backup')
            if backup_ready:
                idle = route.get('business_idle_seconds', -1) >= spec.get('businessIdleSeconds', 300)
                if not forced and not already_running and not idle:
                    return {**result, 'phase': 'WaitingForBusinessIdle'}
                if route.get('active_backend') != 'backup':
                    effects.switch('backup')
                    return {**result, 'forceTraining': forced, 'phase': 'DrainingPrimary'}
                if route.get('converged') is not True or (not forced and route.get('primary_inflight') != 0):
                    return {**result, 'forceTraining': forced, 'phase': 'DrainingPrimary'}
                result['host'] = effects.host('training', deadline)
                return {**result, 'forceTraining': forced, 'phase': 'Training' if result['host'].get('phase') == 'training' else 'StartingTraining'}
            result['fallbackUnavailable'] = True
            if not forced:
                return {**result, 'phase': 'BlockedBackupNotReady'}
        if forced:
            # Re-observe the shared router on every reconcile; persisted controller
            # status is not authority to stop inference after a restart.
            result['forceTraining'] = True
            if route.get('active_backend') != 'maintenance':
                effects.switch('maintenance', force=True)
                return {**result, 'phase': 'ForcingMaintenance'}
            if route.get('converged') is not True:
                return {**result, 'phase': 'ForcingMaintenance'}
            # The deadline explicitly permits interrupting outstanding requests.
            # Ownership and all host-side admission checks remain authoritative.
            result['host'] = effects.host('training', deadline)
            return {**result, 'nightWithoutBackup': True, 'phase': 'Training' if result['host'].get('phase') == 'training' else 'StartingTraining'}
        night = desired({**spec, 'mode': 'Auto', 'trainingUntil': None}, now) == 'Training'
        if spec.get('allowNightWithoutBackup', False) and night:
            idle = route.get('business_idle_seconds', -1) >= spec.get('businessIdleSeconds', 300)
            drained = route.get('converged') is True and route.get('primary_inflight') == 0
            if not already_running and (not idle or not drained):
                if route.get('active_backend') == 'maintenance' and not already_running and host.get('phase') == 'glm_ready':
                    effects.switch('primary')
                return {**result, 'phase': 'WaitingForBusinessIdle'}
            if route.get('active_backend') != 'maintenance':
                effects.switch('maintenance')
                return {**result, 'phase': 'DrainingPrimary'}
            if route.get('converged') is not True:
                return {**result, 'phase': 'DrainingPrimary'}
            result['host'] = effects.host('training', deadline)
            return {**result, 'nightWithoutBackup': True, 'phase': 'Training' if result['host'].get('phase') == 'training' else 'StartingTraining'}
        # Never trust persisted Ready status after controller or backend restarts.
        if not effects.probe('backup'):
            return {**result, 'phase': 'BlockedBackupNotReady'}
        if route.get('active_backend') != 'backup':
            effects.switch('backup')
            return {**result, 'phase': 'DrainingPrimary'}
        # Missing drain counters are unsafe; never interpret missing as zero.
        if route.get('converged') is not True or route.get('primary_inflight') != 0:
            return {**result, 'phase': 'DrainingPrimary'}
        result['host'] = effects.host('training', deadline)
        return {**result, 'phase': 'Training' if result['host'].get('phase') == 'training' else 'StartingTraining'}
    # Establish fallback before restoration, which may involve a recovery reboot.
    # Real probe covers semantic HTTP errors even when /health remains green.
    primary_ready = host.get('phase') == 'glm_ready' and effects.probe('primary')
    if not primary_ready and route.get('active_backend') != 'backup':
        if (not rotate or (l20.get('requested_role') == 'inference' and l20.get('ready') is True)) and effects.probe('backup'):
            effects.switch('backup')
            route = {**route, 'active_backend': 'backup'}
        else:
            result['fallbackUnavailable'] = True
    result['host'] = effects.host('inference')
    if not primary_ready:
        phase = 'RestoringPrimary' if result['host'].get('phase') != 'glm_ready' else 'CheckingPrimary'
        return {**result, 'phase': phase}
    if route.get('active_backend') != 'primary':
        effects.switch('primary')
        if rotate:
            return {**result, 'phase': 'DrainingBackup'}
    if rotate and not night_backup and spec.get('mode', 'Auto') == 'Auto':
        if route.get('converged') is not True or route.get('backup_inflight') != 0:
            return {**result, 'phase': 'DrainingBackup'}
        if l20.get('requested_role') != 'training':
            effects.l20('training')
    return {**result, 'phase': 'PrimaryReady'}


def main():
    api = Kubernetes()
    while True:
        try:
            if api.acquire():
                obj = api.call(api.resource)
                try:
                    status = reconcile(obj, Effects(obj['spec'], api.acquire), dt.datetime.now(UTC))
                except Exception as exc:
                    # Exception details intentionally excluded: external errors may include credentials.
                    status = {'observedGeneration': obj['metadata']['generation'], 'phase': 'ReconcileError', 'reason': type(exc).__name__, 'updatedAt': stamp(dt.datetime.now(UTC))}
                if api.acquire():
                    api.status(obj, status)
                print(json.dumps(status), flush=True)
        except Exception as exc:
            print(json.dumps({'phase': 'ControllerError', 'reason': type(exc).__name__}), flush=True)
        time.sleep(15)

if __name__ == '__main__':
    main()
