"""Evaluate autonomous jobs before releasing GPUs; defer safely to a later window."""
import json,os,subprocess,time
from pathlib import Path

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def run(root,cfg,out,deadline):
    pending=out/'evaluation-pending.json'
    if not cfg['job_id'].startswith('brain-') and not pending.exists():return True
    request=json.loads(pending.read_text()) if pending.exists() else {'job_id':cfg['job_id'],'requested':time.time()}
    if request.get('job_id')!=cfg['job_id']:raise ValueError('Pending evaluation job mismatch')
    request['deadline']=deadline
    atomic(pending,request)
    if deadline-time.time()<1800:
        atomic(out/'post-eval-status.json',{'status':'deferred','reason':'insufficient_window'});return False
    command=['/usr/bin/python3',str(root/'research/evaluation/cli.py'),'candidate-local','--deadline',str(deadline)]
    freeze_id=cfg.get('evaluation_freeze_id') or request.get('freeze_id')
    if freeze_id:
        if not isinstance(freeze_id,str) or len(freeze_id)!=64 or any(c not in '0123456789abcdef' for c in freeze_id):raise ValueError('Invalid freeze ID')
        command+=['--freeze-file',str(root/'research/evaluation/artifacts/frozen'/(freeze_id+'.json'))]
    try:
        if cfg.get('backend')=='llamafactory':
            from llamafactory_runner import evaluation_process
            command+=['--root',str(root),'--job-id',cfg['job_id'],'--runtime-image-id',cfg['image_id']]
            data=evaluation_process(root,out,command,deadline,phase='candidate',budget=1740)
            returncode=0
        else:
            result=subprocess.run(command,capture_output=True,text=True,timeout=1740)
            data=json.loads(result.stdout)
            returncode=result.returncode
        verified=returncode==0 and data.get('comparison_complete') is True and data.get('job_id')==cfg['job_id']
        atomic(out/'post-eval-status.json',{'status':'complete' if verified else 'deferred','returncode':returncode,
            'comparison_path':data.get('comparison_path'),'outcome':data.get('outcome'),'error_type':data.get('error_type')})
        if verified:pending.unlink(missing_ok=True)
        return verified
    except Exception as exc:
        atomic(out/'post-eval-status.json',{'status':'deferred','error_type':type(exc).__name__});return False
