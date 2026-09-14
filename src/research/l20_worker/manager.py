"""Adopt one owned Docker worker across controller restarts; never stop other GPUs."""
import json,subprocess,time,os
try:
 from .common import ROOT,config,atomic,sha,LOCK,REVISION
 from .artifacts import verify_phase,verify_checkpoint,model_path
 from . import phase_control as phases
except ImportError:
 from common import ROOT,config,atomic,sha,LOCK,REVISION
 from artifacts import verify_phase,verify_checkpoint,model_path
 import phase_control as phases
NAME='rtl-l20-training'
SWANLAB_SECRETS='/mnt/data/rtl-l20-training/worker/secrets'
def native_training(job,phase):return phase=='training' and job.get('telemetry')=='swanlab-native-v1'
def expected_network(job,phase):return 'rtl-swanlab' if native_training(job,phase) else 'none'
def verify_telemetry_mounts(container,job,phase):
 mounts=container.get('Mounts') or []
 expected={SWANLAB_SECRETS+'/'+name:'/run/secrets/'+name for name in ('swanlab-api-key','swanlab-proxy')}
 secret_mounts=[m for m in mounts if str(m.get('Destination','')).startswith('/run/secrets')]
 if not native_training(job,phase):
  if secret_mounts:raise RuntimeError('foreign telemetry mounts')
  return
 if len(secret_mounts)!=2 or any(not any(m.get('Source')==source and m.get('Destination')==dest and m.get('Type')=='bind' and m.get('RW') is False for m in secret_mounts) for source,dest in expected.items()):raise RuntimeError('foreign telemetry mounts')
def command(args,timeout=20):return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def inspect():
 r=command(['docker','inspect',NAME])
 if r.returncode==0:return json.loads(r.stdout)[0]
 error=r.stderr.lower()
 if 'no such object: '+NAME in error or 'no such container: '+NAME in error:return None
 raise RuntimeError('container_inspection_unavailable')

def start(folder,phase):
 cfg=config();job=json.loads((folder/'job.json').read_text())
 if not phases.permitted(folder,phase):raise ValueError('phase not authorized')
 for name,expected in job['recipe_sha256'].items():
  if sha((folder/'recipe'/name).read_bytes())!=expected:raise ValueError('pinned recipe changed')
 if any(sha((folder/(n+'.jsonl')).read_bytes())!=job[n+'_sha256'] for n in ('data','prompts','knowledge_prompts')):raise ValueError('input changed')
 gpu=command(['querygpu','--id','0','--query-gpu=memory.free','--format=csv,noheader,nounits'])
 try:
  if gpu.returncode or int(gpu.stdout.strip())<40960:return False
 except ValueError:return False
 if not (os.path.exists(cfg['model_ready'])):return False
 model=model_path(cfg,folder)
 if model is None:return False
 (folder/'run').mkdir(exist_ok=True)
 args=['docker','run','-d','--name',NAME,'--label','rtl.l20.job='+folder.name,'--label','rtl.l20.phase='+phase,'--network',expected_network(job,phase),'--gpus','device=0','--cpus','8','--memory','24g','--memory-swap','24g','--shm-size','2g','--cap-drop','ALL','--security-opt','no-new-privileges','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONUNBUFFERED=1','--mount','type=bind,src='+str(model)+',dst=/model,readonly','--mount','type=bind,src='+str(folder)+',dst=/job',job['image_id'],'python3']
 if native_training(job,phase):
  atomic(folder/'swanlab-config.json',{'project':'RTL-RL','workspace':'krli','job_id':folder.name,'device_label':'L20 GPU0'})
  insert=args.index(job['image_id'])
  for name in ('swanlab-api-key','swanlab-proxy'):
   args[insert:insert]=['--mount','type=bind,src='+SWANLAB_SECRETS+'/'+name+',dst=/run/secrets/'+name+',readonly'];insert+=2
 if phases.managed(folder):args[args.index('--network'):args.index('--network')]=['--label','rtl.l20.run_uid='+phases.control(folder)['run_uid']]
 if phase=='training':
  args+=['/job/recipe/train.py','--model','/model','--data','/job/data.jsonl','--output','/job/run','--dataset-id',job['dataset_id'],'--max-steps',str(job['max_steps']),'--max-length','1024','--stop-file','/job/pause.request']
  if native_training(job,phase):args+=['--swanlab-config','/job/swanlab-config.json']
  if (folder/'run/latest').exists():
   verify_checkpoint(folder)
   args+=['--resume-from','latest']
 else:
  output=folder/(phase+'.jsonl')
  knowledge_output=folder/('knowledge_'+phase+'.jsonl')
  if knowledge_output.exists():knowledge_output.rename(folder/('knowledge_'+phase+'.interrupted-'+str(time.time_ns())+'.jsonl'))
  if output.exists():output.rename(folder/(phase+'.interrupted-'+str(time.time_ns())+'.jsonl'))
  args+=['/job/recipe/generate_eval.py','--model','/model','--prompts','/job/prompts.jsonl','--prompts-sha256',job['prompts_sha256'],'--output','/job/'+phase+'.jsonl']
  args+=['--knowledge-prompts','/job/knowledge_prompts.jsonl','--knowledge-prompts-sha256',job['knowledge_prompts_sha256'],'--knowledge-output','/job/knowledge_'+phase+'.jsonl']
  if phase=='candidate':args+=['--adapter','/job/run/adapter']
 if not phases.permitted(folder,phase):return False
 r=command(args,60)
 if r.returncode:raise RuntimeError('container_start_failed')
 return True

def failure(folder,state,reason):
 state.update(attempts=state.get('attempts',0)+1,last_error=reason,retry_at=time.time()+300)
 if state['attempts']>=3:state['phase']='failed'
 atomic(folder/'state.json',state)

def step():
 with LOCK:
  return _step()

def _step():
 cfg=config();container=inspect()
 pending=[]
 for folder in (ROOT/'jobs').glob('l20-*'):
  state=json.loads((folder/'state.json').read_text())
  if state['phase'] not in ('complete','failed'):pending.append((folder,state))
 if len(pending)>1:raise RuntimeError('multiple active experiments')
 if not pending:return
 folder,state=pending[0]
 state=phases.recover_grant(folder,state)
 lease_paused=phases.managed(folder) and state['phase'] in ('training','queued_training') and not phases.permitted(folder,'training')
 if (ROOT/'pause').exists() or not (ROOT/'enabled').exists() or lease_paused or phases.cancelled(folder):(folder/'pause.request').touch()
 else:(folder/'pause.request').unlink(missing_ok=True)
 if container:
  labels=container['Config'].get('Labels') or {}
  if labels.get('rtl.l20.job')!=folder.name:raise RuntimeError('foreign container identity')
  job=json.loads((folder/'job.json').read_text())
  phase=labels.get('rtl.l20.phase')
  if phase not in ('baseline','training','candidate'):raise RuntimeError('foreign container phase')
  verify_telemetry_mounts(container,job,phase)
  devices=container.get('HostConfig',{}).get('DeviceRequests') or []
  if container.get('Image')!=job['image_id'] or container.get('HostConfig',{}).get('NetworkMode')!=expected_network(job,phase) or len(devices)!=1 or devices[0].get('DeviceIDs')!=['0']:
   raise RuntimeError('foreign container resources')
  if phases.managed(folder) and (not phases.granted(folder,phase) or labels.get('rtl.l20.run_uid')!=phases.control(folder)['run_uid']):raise RuntimeError('foreign orchestrator binding')
  if container['State']['Running']:
   if phase=='training' and not phases.permitted(folder,phase):(folder/'pause.request').touch()
   state.update(phase=phase,container_id=container['Id']);atomic(folder/'state.json',state);return
  log=command(['docker','logs',NAME],30)
  with (folder/(phase+'.worker.log')).open('a') as f:f.write(log.stdout[-1024*1024:]+log.stderr[-1024*1024:])
  code=container['State']['ExitCode']
  removed=command(['docker','rm',NAME])
  if removed.returncode:raise RuntimeError('owned_container_remove_failed')
  if phases.cancelled(folder):
   state['last_exit_code']=code;phases.finish_cancel(folder,state);return
  if code:
   state.update(attempts=state.get('attempts',0)+1,last_exit_code=code,retry_at=time.time()+300)
   state['phase']='failed' if state['attempts']>=3 else 'queued_'+phase
  elif phase=='baseline':
   try:verify_phase(folder,'baseline')
   except Exception as exc:
    state['phase']='queued_baseline';failure(folder,state,'invalid_baseline_artifact');return
   state['phase']=phases.after(folder,'baseline')
  elif phase=='training':
   try:done=verify_phase(folder,'training')
   except Exception:
    state['phase']='queued_training';failure(folder,state,'invalid_training_artifact');return
   state.update(phase=phases.after(folder,'training',done),step=done['step'])
  else:
   try:verify_phase(folder,'candidate')
   except Exception:
    state['phase']='queued_candidate';failure(folder,state,'invalid_candidate_artifact');return
   state['phase']=phases.after(folder,'candidate')
  atomic(folder/'state.json',state);return
 if phases.cancelled(folder):phases.finish_cancel(folder,state);return
 if state['phase'] in ('baseline','training','candidate'):
  previous=state['phase']
  try:
   artifact=verify_phase(folder,previous)
   state['phase']=phases.after(folder,previous,artifact)
   atomic(folder/'state.json',state)
  except Exception:
   state['phase']='queued_'+previous;failure(folder,state,'owned_worker_disappeared')
   return
 if state['phase'].startswith('awaiting_') or (ROOT/'pause').exists() or not (ROOT/'enabled').exists():return
 if time.time()<state.get('retry_at',0):return
 phase={'queued':'baseline','queued_baseline':'baseline','queued_training':'training','queued_candidate':'candidate'}[state['phase']]
 if not phases.permitted(folder,phase):return
 try:
  if start(folder,phase):state.update(phase=phase,started_at=time.time());atomic(folder/'state.json',state)
 except Exception as exc:
  # A timed-out docker client may already have created the owned container.
  # Adopt it on the next iteration instead of marking a live job terminal.
  observed=inspect()
  if observed is not None and (observed.get('Config',{}).get('Labels') or {}).get('rtl.l20.job')==folder.name:
   state.update(phase=phase,last_error='start_response_uncertain');atomic(folder/'state.json',state)
  else:failure(folder,state,type(exc).__name__)

def loop():
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'jobs').mkdir(exist_ok=True)
 while True:
  try:step();atomic(ROOT/'heartbeat.json',{'time':time.time(),'status':'ready'})
  except Exception as exc:atomic(ROOT/'heartbeat.json',{'time':time.time(),'status':'error','error_type':type(exc).__name__})
  time.sleep(10)
