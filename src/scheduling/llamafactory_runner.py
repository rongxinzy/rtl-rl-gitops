"""Pinned night SFT runner; never owns inference shutdown or GPU reclamation."""
import hashlib,importlib.util,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
FILES={'train.py','state.py','model.py','provenance.py','knowledge_data.py','generate_eval.py'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True)+'\n');tmp.replace(path)
def relative(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or not p.parts or any(x in name for x in (',','\n','\r')):raise ValueError('unsafe relative path')
 result=root/p
 if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):raise ValueError('path escape')
 return result

def validate(root,cfg):
 if cfg.get('backend')!='llamafactory' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',cfg['job_id']):raise ValueError('invalid backend/job')
 if type(cfg['max_steps']) is not int or not 1<=cfg['max_steps']<=20:raise ValueError('max_steps outside bound')
 if cfg.get('max_length')!=1024 or cfg.get('lora_rank')!=8 or cfg.get('model_revision')!=REVISION or cfg.get('llamafactory_commit')!='100e9a42c6c09f8f7849b70d60f3da445fb2024b':raise ValueError('training recipe version mismatch')
 if not re.fullmatch(r'sha256:[0-9a-f]{64}',cfg['image_id']):raise ValueError('immutable image ID required')
 for key in ('data_sha256','evaluation_freeze_id','source_baseline_evaluation_id'):
  if not re.fullmatch(r'[0-9a-f]{64}',cfg[key]):raise ValueError('invalid identity hash')
 if not isinstance(cfg['dataset_id'],str) or not cfg['dataset_id']:raise ValueError('dataset ID required')
 data=relative(root,cfg['data']);model=relative(root,cfg['model_path']);recipe=relative(root,cfg['recipe_path'])
 if sha(data)!=cfg['data_sha256']:raise ValueError('data hash mismatch')
 if set(cfg['recipe_sha256'])!=FILES:raise ValueError('complete recipe hashes required')
 for name,value in cfg['recipe_sha256'].items():
  if (recipe/name).is_symlink() or sha(recipe/name)!=value:raise ValueError('recipe hash mismatch')
 if not model.is_dir():raise ValueError('model missing')
 return data,model,recipe

def stop_reason(root,deadline):
 if time.time()>=deadline-720:return 'deadline_save_margin'
 if (root/'state/scheduler-hold').exists():return 'scheduler_hold'
 if (root/'state/night-stop').exists():return 'operator_stop'
 if (root/'state/operator-enabled').exists():
  path=root/'state/operator-authority.json'
  try:authority=json.loads(path.read_text())
  except (OSError,ValueError):return 'authority_unavailable'
  if authority.get('mode')!='training' or authority.get('valid_until',0)<=time.time() or authority.get('deadline',0)<deadline:return 'authority_expired'
 return None

def evaluation_process(root,out,command,deadline,phase='baseline',budget=1800):
 """Interrupt only the owned CLI; its finally removes its exact Docker name."""
 if phase not in ('baseline','candidate') or not 1<=budget<=1800:raise ValueError('invalid evaluation phase/budget')
 if stop_reason(root,deadline):raise RuntimeError(phase+' stop already requested')
 process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 started=time.monotonic();requested=None;warned=False
 def interrupt(reason):
  nonlocal requested
  if requested is None and process.poll() is None:
   requested=time.monotonic()
   atomic(out/(phase+'-stop-request.json'),{'reason':reason,'time':time.time(),'cli_pid':process.pid})
   # No process-group signal: Docker CLI must not kill the cleanup subprocess.
   try:process.send_signal(signal.SIGINT)
   except ProcessLookupError:pass
 def terminate(signum,frame):raise InterruptedError(phase+' runner terminated')
 previous=signal.signal(signal.SIGTERM,terminate)
 try:
  while True:
   reason=stop_reason(root,deadline)
   if time.monotonic()-started>=budget:reason=reason or phase+'_runtime_budget'
   if reason:interrupt(reason)
   if requested is not None and not warned and time.monotonic()-requested>=120:
    atomic(out/(phase+'-cleanup-pending.json'),{'time':time.time(),'cli_pid':process.pid,'status':'awaiting_owned_cli_cleanup'});warned=True
   try:stdout,stderr=process.communicate(timeout=1);break
   except subprocess.TimeoutExpired:continue
  if requested is not None:raise RuntimeError(phase+' interrupted; cleanup awaited')
  if process.returncode:raise RuntimeError(phase+' evaluation failed')
  return json.loads(stdout)
 finally:
  # SIGTERM/KeyboardInterrupt of this runner still give the child its cleanup path.
  try:
   if process.poll() is None:
    interrupt('runner_interrupted')
    process.communicate()  # never timeout-kill a CLI while its finally owns cleanup
  finally:signal.signal(signal.SIGTERM,previous)

def baseline_process(root,out,command,deadline):
 return evaluation_process(root,out,command,deadline,phase='baseline',budget=1800)

def baseline(root,cfg,out,deadline):
 cache=out/'runtime-baseline.json'
 if cache.exists():result=json.loads(cache.read_text())
 else:
  command=['/usr/bin/python3',str(root/'research/evaluation/cli.py'),'baseline-local','--root',str(root),'--runtime-image-id',cfg['image_id'],'--freeze-file',str(root/'research/evaluation/artifacts/frozen'/(cfg['evaluation_freeze_id']+'.json')),'--deadline',str(deadline)]
  result=baseline_process(root,out,command,deadline)
 if result.get('status')!='complete' or result.get('freeze_id')!=cfg['evaluation_freeze_id']:raise ValueError('baseline incomplete/freeze mismatch')
 path=Path(result['result_path'])
 if not path.resolve().is_relative_to((root/'research/evaluation/artifacts/runs').resolve()) or path.is_symlink() or sha(path)!=result.get('evaluation_id'):raise ValueError('baseline artifact hash/path mismatch')
 report=json.loads(path.read_text())
 frozen=json.loads((root/'research/evaluation/artifacts/frozen'/(cfg['evaluation_freeze_id']+'.json')).read_text())
 tasks=[r['task_id'] for r in report.get('results',[])]
 if len(tasks)!=len(set(tasks)) or set(tasks)!={r['task_id'] for r in frozen['tasks']} or report.get('judge_runner_sha256')!=frozen['judge_runner_sha256']:raise ValueError('baseline task/judge mismatch')
 if report.get('kind')!='qwen_base_baseline' or report.get('model_revision')!=REVISION or report.get('freeze_id')!=cfg['evaluation_freeze_id'] or report.get('metadata',{}).get('training_image_id')!=cfg['image_id']:raise ValueError('baseline identity mismatch')
 if report.get('counts',{}).get('infrastructure',1) or report.get('counts',{}).get('unknown',1):raise ValueError('baseline infrastructure incomplete')
 if not cache.exists():atomic(cache,result)
 return result['evaluation_id']

def command(root,cfg,out,data,model,recipe):
 cmd=['docker','run','--rm','--name','rtl-night-training','--network','none','--gpus','device=0','--cpus','8','--memory','48g','--shm-size','2g','--entrypoint','python3','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1']
 for source,target,ro in ((data,'/data/train.jsonl',True),(model,'/model',True),(recipe,'/recipe',True),(out,'/job',False),(root/'state','/controls',True)):
  cmd+=['--mount',f'type=bind,src={source},dst={target}'+(',readonly' if ro else '')]
 cmd += [cfg['image_id'],'/recipe/train.py','--model','/model','--data','/data/train.jsonl','--output','/job/train-run','--dataset-id',cfg['dataset_id'],'--max-steps',str(cfg['max_steps']),'--max-length','1024','--stop-file','/controls/night-stop']
 if (out/'train-run/latest').exists():cmd+=['--resume-from','latest']
 return cmd

def publish_outputs(out,recipe,cfg):
 run=out/'train-run';spec=importlib.util.spec_from_file_location('lf_night_state',recipe/'state.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 identity=json.loads((run/'job.json').read_text())
 for key in ('dataset_id','data_sha256','max_steps','model_revision','recipe_sha256'):
  expected=cfg[key]
  if key=='recipe_sha256':expected={k:v for k,v in expected.items() if k!='generate_eval.py'}
  if identity.get(key)!=expected:raise ValueError('native training identity mismatch')
 if identity.get('backend')!='llamafactory' or identity.get('max_length')!=1024 or identity.get('rank')!=8:raise ValueError('native training config mismatch')
 cp=helper.recover_latest(run,helper.digest(run/'job.json'));manifest=helper.verify_checkpoint(cp,helper.digest(run/'job.json'))
 status=json.loads((run/'status.json').read_text())
 if status.get('step')!=manifest['step'] or status.get('job_sha256')!=manifest['job_sha256'] or status.get('status') not in ('complete','paused'):raise ValueError('training status identity mismatch')
 adapter=out/'adapter';adapter.mkdir(exist_ok=True)
 for name in ('adapter_config.json','adapter_model.safetensors'):
  source=run/'adapter'/name
  if source.is_symlink() or sha(source)!=manifest['files'][name]:raise ValueError('adapter/checkpoint mismatch')
  tmp=adapter/(name+'.tmp');shutil.copyfile(source,tmp);tmp.replace(adapter/name)
 for name in ('trainer_state_final.json','training_metrics.json'):
  value=json.loads((run/name).read_text())
  if value.get('global_step')!=manifest['step'] or value.get('job_sha256')!=manifest['job_sha256']:raise ValueError('final output mismatch')
  atomic(out/name,value)
 return status

def _run(root,cfg,deadline):
 root=Path(root);data,model,recipe=validate(root,cfg)
 if stop_reason(root,deadline):raise RuntimeError('training authority unavailable or stop requested')
 actual=subprocess.check_output(['docker','image','inspect',cfg['image_id'],'--format','{{.Id}}'],text=True).strip()
 if actual!=cfg['image_id']:raise ValueError('image identity mismatch')
 out=root/'runs'/cfg['job_id'];out.mkdir(parents=True,exist_ok=True)
 if out.is_symlink() or out.resolve().parent!=(root/'runs').resolve():raise ValueError('job directory symlink rejected')
 pin=out/'job-config.json';identity=dict(cfg,model_revision=REVISION)
 if pin.exists():
  saved=json.loads(pin.read_text());identity['baseline_evaluation_id']=saved.get('baseline_evaluation_id')
  if saved!=identity:raise ValueError('immutable night job differs; use new job')
 identity['baseline_evaluation_id']=baseline(root,cfg,out,deadline)
 if pin.exists() and json.loads(pin.read_text())!=identity:raise ValueError('baseline changed')
 if not pin.exists():atomic(pin,identity)
 if stop_reason(root,deadline) or deadline-time.time()<1800:raise RuntimeError('insufficient authorized training window')
 atomic(out/'last-start.json',{'time':time.time(),'deadline':deadline,'backend':'llamafactory'})
 with (out/'training.log').open('a') as log:
  process=subprocess.Popen(command(root,cfg,out,data,model,recipe),stdout=log,stderr=subprocess.STDOUT)
  requested=False
  try:
   while process.poll() is None:
    reason=stop_reason(root,deadline)
    if reason and not requested:
     (root/'state/night-stop').touch();atomic(out/'stop-request.json',{'time':time.time(),'reason':reason});requested=True
    time.sleep(5)
  finally:
   if process.poll() is None:
    (root/'state/night-stop').touch()
    process.wait()  # wait for checkpoint cleanup before publishing any exit state
  rc=process.returncode
 step=None;status=None
 if rc==0:status=publish_outputs(out,recipe,cfg);step=status['step']
 atomic(out/'last-exit.json',{'time':time.time(),'returncode':rc,'step':step,'backend':'llamafactory','status':status['status'] if status else 'failed'})
 if rc==0 and status['status']=='complete' and step>=cfg['max_steps']:
  # Explicit pending marker also makes non-Brain reviewed jobs enter existing evaluation.
  if not (out/'evaluation-pending.json').exists():atomic(out/'evaluation-pending.json',{'job_id':cfg['job_id'],'freeze_id':cfg['evaluation_freeze_id'],'baseline_evaluation_id':identity['baseline_evaluation_id'],'requested':time.time()})
  import post_eval
  post_eval.run(root,identity,out,deadline)
  atomic(out/'job-complete.json',{'step':step,'time':time.time(),'backend':'llamafactory'})
 return rc

def run(root,cfg,deadline):
 def terminate(signum,frame):raise InterruptedError('training runner terminated')
 previous=signal.signal(signal.SIGTERM,terminate)
 try:return _run(root,cfg,deadline)
 except Exception as exc:
  job=cfg.get('job_id','')
  if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',job):
   out=Path(root)/'runs'/job
   if not out.is_symlink() and (out/'train-run/job.json').is_file():
    atomic(out/'last-exit.json',{'time':time.time(),'returncode':1,'backend':'llamafactory','status':'failed','error_type':type(exc).__name__})
  raise
 finally:signal.signal(signal.SIGTERM,previous)
