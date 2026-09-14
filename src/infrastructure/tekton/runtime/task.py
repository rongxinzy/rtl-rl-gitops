"""TaskRun retries renew one bound phase, never create a second GPU experiment."""
import argparse
import hashlib
import json
import time
import urllib.error
from client import worker, executor, identity, check_status, result

def retryable(exc):
    return isinstance(exc, (TimeoutError, OSError)) and (not isinstance(exc, urllib.error.HTTPError) or exc.code in (409,429,500,502,503,504))

MAX_ROTATION_PAUSE = 14 * 3600

def phase(job, uid, name, deadline):
    # Only host-verified scheduled rotation pauses suspend the active-work budget.
    # The wall limit remains bounded if inference cannot hand the GPUs back.
    wall_deadline = deadline + MAX_ROTATION_PAUSE
    previous = time.monotonic()
    was_paused = False
    paused_seconds = 0.0
    while time.monotonic() < wall_deadline:
        now = time.monotonic()
        if was_paused:
            extra = min(max(0.0, now - previous), MAX_ROTATION_PAUSE - paused_seconds)
            deadline += extra
            paused_seconds += extra
        previous = now
        if now >= deadline:
            break
        try:
            state = check_status(worker('/jobs/'+job+'/status'), job, uid, unbound=True)
            was_paused = state.get('rotation_paused') is True
            if state['phase']=='failed':
                raise ValueError('worker phase exhausted its bounded retry budget')
            # A retry may arrive after this phase completed; retain the original run binding.
            if state.get('verified_complete',{}).get(name):
                check_status(state,job,uid)
                return {'job_id':job,'run_uid':uid,'phase':name,'job_sha256':state['job_sha256'],'verified_complete':True}
            worker('/jobs/'+job+'/phase', {'phase':name,'run_uid':uid})
        except Exception as exc:
            if not retryable(exc):
                raise
        time.sleep(10)
    raise TimeoutError('bounded phase deadline; worker lease will request checkpoint pause')

def compare(job, uid, deadline):
    state=check_status(worker('/jobs/'+job+'/status'),job,uid)
    if not state.get('verified_complete',{}).get('candidate'):
        raise ValueError('candidate evidence is not complete')
    base='tekton-'+hashlib.sha256((job+uid).encode()).hexdigest()[:32]+'-compare-'
    attempt=0
    while time.monotonic()<deadline:
        aid=base+str(attempt)
        try:
            try:
                action=executor('/v1/actions/'+aid)
            except urllib.error.HTTPError as exc:
                if exc.code!=404:
                    raise
                action={'state':'absent'}
            if action['state'] in ('absent','staged'):
                body={'action_id':aid,'action':'l20.compare','params':{'job_id':job,'run_uid':uid},'stage':'stage'}
                staged=executor('/v1/actions',body)
                if staged.get('state')=='staged':
                    executor('/v1/actions',{**body,'stage':'apply','plan_hash':staged['plan_hash']})
            elif action['state']=='completed':
                value=action.get('result') or {}
                if not action.get('evidence_verified') or not value.get('comparison_complete') or value.get('job_id')!=job:
                    raise ValueError('comparison result is not verified for this job')
                current=check_status(worker('/jobs/'+job+'/status'),job,uid)
                if current['phase']!='complete' or current.get('comparison',{}).get('comparison_id')!=value.get('comparison_id'):
                    raise ValueError('worker comparison acknowledgement mismatch')
                return {k:value[k] for k in ('job_id','comparison_id','outcome')}
            elif action['state'] in ('interrupted','failed') and attempt<2:
                current=check_status(worker('/jobs/'+job+'/status'),job,uid)
                transient=action['state']=='interrupted' or action.get('reason') in ('timeout','TimeoutError','URLError','ConnectionError','HTTPError')
                if not transient and current['phase']!='complete':
                    raise ValueError('nonretryable independent comparison failure')
                # Re-run the executor's full cached verification after an uncertain commit.
                attempt+=1
            elif action['state'] in ('failed','rejected','interrupted'):
                raise ValueError('independent comparison failed')
        except Exception as exc:
            if not retryable(exc):
                raise
        time.sleep(5)
    raise TimeoutError('comparison deadline')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['baseline','training','candidate','compare','record'])
    p.add_argument('--job-id',required=True);p.add_argument('--run-uid',required=True)
    a=p.parse_args();identity(a.job_id,a.run_uid)
    if a.action=='record':
        state=check_status(worker('/jobs/'+a.job_id+'/status'),a.job_id,a.run_uid,unbound=True)
        value={k:state.get(k) for k in ('job_id','run_uid','job_sha256','phase','comparison')}
        value['scope']='observed worker state; finalization does not bypass phase leases'
    elif a.action=='compare':
        value=compare(a.job_id,a.run_uid,time.monotonic()+900)
        result('outcome',value['outcome'])
    else:
        value=phase(a.job_id,a.run_uid,a.action,time.monotonic()+(3600 if a.action=='training' else 900))
    result('proof',value)
    print(json.dumps({'event':'task_verified','action':a.action,'job_id':a.job_id}))

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'event':'task_failed','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
