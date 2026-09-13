#!/usr/bin/env python3
"""Forced-command bridge: Operator owns intent; host retains bounded recovery."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
import target

STATE=target.STATE

def status():
    s=target.read_state()
    job=json.loads((target.ROOT/'scheduling/job.json').read_text())['job_id']
    s['job_id']=job
    s['idle_after_completion']=(target.ROOT/'runs'/job/'job-complete.json').exists() and not (target.ROOT/'runs'/job/'evaluation-pending.json').exists()
    if target.running(target.GLM):s['phase']='glm_ready' if target.healthy() else 'glm_loading'
    elif target.running(target.TRAIN):s['phase']='training'
    s['operator_enabled']=(STATE/'operator-enabled').exists()
    return s

def main():
    action=os.environ.get('SSH_ORIGINAL_COMMAND',' '.join(sys.argv[1:]))
    if action=='operator-status':
        print(json.dumps(status()));return
    match=re.fullmatch(r'operator-(inference|training)(?: ([0-9]{10}))?',action)
    if not match:raise SystemExit('Unsupported operator command')
    mode,raw=match.groups()
    deadline=int(raw) if raw else None
    if mode=='training' and (deadline is None or not 0<deadline-time.time()<=86400):raise SystemExit('Invalid training deadline')
    if mode=='inference' and raw:raise SystemExit('Unexpected argument')
    if not (STATE/'operator-enabled').exists():raise SystemExit('Operator authority not enabled')
    with (STATE/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        now=time.time()
        value={'mode':mode,'deadline':deadline,'updated':now,'valid_until':now+180}
        tmp=STATE/'operator-authority.json.tmp';tmp.write_text(json.dumps(value));os.replace(tmp,STATE/'operator-authority.json')
        s=target.read_state()
        s.pop('error',None);s.pop('failsafe_reason',None)
        try:s=target.reconcile(s,dt.datetime.now(dt.timezone.utc),requested_mode='night' if mode=='training' else 'day')
        except Exception as exc:s.update(phase='error',error=str(exc))
        target.write_state(s)
    print(json.dumps(status()))

if __name__=='__main__':main()
