#!/usr/bin/env python3
"""Run a pinned, bounded job; resume only atomically committed checkpoints."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/root/rtl-rl')

def atomic(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)

def main():
    cfg=json.loads((ROOT/'scheduling/job.json').read_text())
    job=cfg['job_id']
    if not job.replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid job id')
    now=dt.datetime.now(ZoneInfo('Asia/Shanghai'))
    authority_path=ROOT/'state/operator-authority.json'
    authority=json.loads(authority_path.read_text()) if authority_path.exists() else {}
    if (ROOT/'state/operator-enabled').exists():
        if authority.get('mode')!='training' or authority.get('valid_until',0)<=time.time():
            raise SystemExit('Operator training authority absent or expired')
        deadline=authority['deadline']
    else:
        if not (now.hour>=22 or (now.hour,now.minute)<(7,30)):
            raise SystemExit('Outside training window')
        morning=now.date()+dt.timedelta(days=1) if now.hour>=22 else now.date()
        deadline=dt.datetime.combine(morning,dt.time(7,30),ZoneInfo('Asia/Shanghai')).timestamp()
    if deadline-time.time()<1800:
        raise SystemExit('Too close to morning deadline for a new model load')
    out=ROOT/'runs'/job
    out.mkdir(parents=True,exist_ok=True)
    if (out/'job-complete.json').exists():
        if (out/'evaluation-pending.json').exists():
            import post_eval
            post_eval.run(ROOT,cfg,out,deadline)
            return
        print('Configured job already complete; awaiting a new reviewed job.',flush=True)
        return
    data=ROOT/cfg['data']
    if hashlib.sha256(data.read_bytes()).hexdigest()!=cfg['data_sha256']:
        raise ValueError('Pinned data hash mismatch')
    identity={k:cfg[k] for k in ('job_id','data','data_sha256','max_steps','max_completion_length','save_steps')}
    source=json.loads((ROOT/'manifests/qwen-model-source.json').read_text())
    image=subprocess.check_output(['docker','image','inspect','rtl-training:20260912-swanlab','--format','{{.Id}}'],text=True).strip()
    identity.update(model_revision=source['sha'], image_id=image, num_generations=4,
                    recipe_sha256={name:hashlib.sha256((ROOT/'training'/name).read_bytes()).hexdigest()
                                   for name in ('grpo.py','rewards.py','launch.sh')})
    pin=out/'job-config.json'
    if pin.exists() and json.loads(pin.read_text())!=identity:
        raise ValueError('Existing job configuration differs; use a new job id')
    atomic(pin,identity)
    sys.path.insert(0, str(ROOT/'training'))
    from nightly import latest_complete_checkpoint
    latest=latest_complete_checkpoint(out)
    candidates=[]
    if latest:
        folder=Path(latest)
        step=int(json.loads((folder/'trainer_state.json').read_text())['global_step'])
        candidates.append((step,folder))
    if candidates and max(candidates)[0]>=cfg['max_steps']:
        import post_eval
        post_eval.run(ROOT,cfg,out,deadline)
        atomic(out/'job-complete.json',{'step':max(candidates)[0],'source':'committed_checkpoint'})
        return
    command=['bash',str(ROOT/'training/launch.sh'),'grpo','--data','/workspace/'+cfg['data'],
             '--output','/workspace/runs/'+job,'--max-steps',str(cfg['max_steps']),
             '--max-completion-length',str(cfg['max_completion_length']),
             '--save-steps',str(cfg['save_steps']),'--num-generations','4','--swanlab']
    if candidates:
        command+=['--resume-from','/workspace/runs/'+job+'/'+max(candidates)[1].name]
    env=dict(os.environ,RTL_GPU_DEVICES='0',RTL_CONTAINER_NAME='rtl-night-training',RTL_DEADLINE_EPOCH=str(deadline))
    atomic(out/'last-start.json',{'time':time.time(),'deadline':deadline,'resume_step':max(candidates)[0] if candidates else None})
    result=subprocess.run(command,env=env)
    state=out/'trainer_state_final.json'
    step=int(json.loads(state.read_text())['global_step']) if state.exists() else None
    atomic(out/'last-exit.json',{'time':time.time(),'returncode':result.returncode,'step':step})
    if result.returncode==0 and step is not None and step>=cfg['max_steps']:
        import post_eval
        post_eval.run(ROOT,cfg,out,deadline)
        atomic(out/'job-complete.json',{'step':step,'time':time.time()})
    raise SystemExit(result.returncode)

if __name__=='__main__': main()
