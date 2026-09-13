"""Durable, authenticated Tekton phase grants; legacy jobs retain automatic flow."""
import json,re,time
try:
 from .common import atomic,sha
 from .artifacts import verify_phase
except ImportError:
 from common import atomic,sha
 from artifacts import verify_phase
PHASES=('baseline','training','candidate')
UID=re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

def managed(folder):return json.loads((folder/'job.json').read_text()).get('orchestrator')=='tekton'
def control(folder):
 path=folder/'phase-control.json';job_sha=sha((folder/'job.json').read_bytes())
 if not path.exists():return {'job_sha256':job_sha,'run_uid':None,'authorized':[],'completed':[],'leases':{}}
 if path.is_symlink():raise ValueError('invalid phase binding')
 value=json.loads(path.read_text())
 if value.get('job_sha256')!=job_sha:raise ValueError('phase job binding changed')
 if not isinstance(value.get('run_uid'),str) or not UID.fullmatch(value['run_uid']):raise ValueError('invalid run binding')
 for key in ('authorized','completed'):
  sequence=value.get(key)
  if not isinstance(sequence,list) or sequence!=list(PHASES[:len(sequence)]):raise ValueError('invalid phase sequence')
 if len(value['completed'])>len(value['authorized']):raise ValueError('completion without authorization')
 return value

def verified(folder,phase):
 result=verify_phase(folder,phase)
 if phase=='training' and result.get('status')!='complete':raise ValueError('training only paused')
 return result

def granted(folder,phase):
 return not managed(folder) or phase in control(folder)['authorized']
def permitted(folder,phase):
 if not managed(folder):return True
 value=control(folder)
 return not value.get('cancel_requested') and phase in value['authorized'] and value.get('leases',{}).get(phase,0)>time.time()

def receipt(folder,phase):
 names=['job.json','model-binding.json']
 if phase=='training':
  names+=['run/'+n for n in ['job.json','status.json','trainer_state_final.json','training_metrics.json','latest','adapter/adapter_config.json','adapter/adapter_model.safetensors']]
  pointer=(folder/'run/latest').read_text().strip()
  if not re.fullmatch(r'checkpoint-[0-9]{6}',pointer):raise ValueError('invalid checkpoint receipt')
  names+=['run/'+pointer+'/'+p.name for p in (folder/'run'/pointer).iterdir() if p.is_file()]
 else:
  names += [phase+'.jsonl','knowledge_'+phase+'.jsonl','prompts.jsonl','knowledge_prompts.jsonl']
  if phase=='candidate':names+=['run/adapter/adapter_config.json','run/adapter/adapter_model.safetensors']
 records=[]
 for name in names:
  path=folder/name
  if path.is_symlink() or not path.is_file():raise ValueError('invalid receipt input')
  st=path.stat();records.append({'path':name,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns})
 return {'files':records,'verified_at':time.time()}

def receipt_valid(folder,value,phase):
 saved=value.get('receipts',{}).get(phase)
 if not isinstance(saved,dict) or not saved.get('files'):return False
 try:
  for row in saved['files']:
   path=folder/row['path']
   if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):return False
   st=path.stat()
   if [st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=[row['bytes'],row['mtime_ns'],row['ctime_ns']]:return False
  return True
 except (OSError,KeyError,TypeError):return False

def completed(folder,phase):
 if not managed(folder):return
 value=control(folder)
 if phase not in value['authorized']:raise ValueError('phase has no run authorization')
 verified(folder,phase)
 if phase not in value['completed']:
  if PHASES[len(value['completed'])]!=phase:raise ValueError('out-of-order completion')
  value.setdefault('receipts',{})[phase]=receipt(folder,phase)
  value['completed'].append(phase);value['updated_at']=time.time();atomic(folder/'phase-control.json',value)

def after(folder,phase,artifact=None):
 if phase=='training' and artifact['status']!='complete':return 'queued_training'
 if managed(folder):
  completed(folder,phase)
  if phase!='candidate':return 'awaiting_'+PHASES[PHASES.index(phase)+1]+'_authorization'
 return {'baseline':'queued_training','training':'queued_candidate','candidate':'awaiting_evaluation'}[phase]

def status(folder):
 state=json.loads((folder/'state.json').read_text());job=json.loads((folder/'job.json').read_text());is_managed=managed(folder)
 value=control(folder) if is_managed else {'run_uid':None,'authorized':[],'completed':[]}
 phases={}
 for phase in PHASES:
  ok=False
  if phase in value['completed']:ok=receipt_valid(folder,value,phase)
  phases[phase]={'authorized':phase in value['authorized'],'verified_complete':ok,'lease_expires_at':value.get('leases',{}).get(phase), 'lease_active':not value.get('cancel_requested',False) and value.get('leases',{}).get(phase,0)>time.time()}
 next_phase=None
 for index,phase in enumerate(PHASES):
  if phase not in value['authorized'] and all(phases[p]['verified_complete'] for p in PHASES[:index]):next_phase=phase;break
 error=state.get('last_error');safe_errors={'invalid_baseline_artifact','invalid_training_artifact','invalid_candidate_artifact','owned_worker_disappeared','start_response_uncertain','ValueError','RuntimeError','TimeoutExpired','FileNotFoundError','PermissionError','tekton_cancelled'}
 return {'job_id':folder.name,'job_sha256':sha((folder/'job.json').read_bytes()),'dataset_id':job.get('dataset_id'),'max_steps':job.get('max_steps'),'freeze_id':job.get('freeze_id'),'comparison':state.get('comparison') if state['phase']=='complete' else None,'orchestrator':'tekton' if is_managed else 'legacy','run_uid':value['run_uid'],'cancel_requested':bool(value.get('cancel_requested')),'phase':state['phase'],'attempts':state.get('attempts',0),'last_error':error if error in safe_errors else 'worker_error' if error else None,'verified_complete':{p:v['verified_complete'] for p,v in phases.items()},'phases':phases,'next_authorizable_phase':next_phase if is_managed and not value.get('cancel_requested') and state['phase'] not in ('failed','complete') else None}

def authorize(folder,body):
 if not isinstance(body,dict) or set(body)!={'phase','run_uid'}:raise ValueError('fixed phase fields required')
 phase=body['phase'];uid=body['run_uid']
 if not isinstance(phase,str) or phase not in PHASES or not isinstance(uid,str) or not UID.fullmatch(uid):raise ValueError('invalid phase or run UID')
 if not managed(folder):raise ValueError('legacy job cannot be taken over')
 value=control(folder);state=json.loads((folder/'state.json').read_text())
 if value['run_uid'] not in (None,uid):raise ValueError('job bound to another run')
 if value.get('cancel_requested'):raise ValueError('job cancellation requested')
 # Repeated requests only return progress; they never reset retries or relaunch work.
 if phase in value['authorized']:
  value.setdefault('leases',{})[phase]=time.time()+180;atomic(folder/'phase-control.json',value)
  return status(folder)
 if state['phase'] in ('failed','complete'):raise ValueError('terminal job')
 index=PHASES.index(phase)
 if value['authorized']!=list(PHASES[:index]) or value['completed']!=list(PHASES[:index]):raise ValueError('phase predecessor incomplete')
 for prior in PHASES[:index]:verified(folder,prior)
 if state['phase'] not in ('queued','queued_baseline','awaiting_'+phase+'_authorization'):raise ValueError('job not ready for phase')
 value['run_uid']=uid;value['authorized'].append(phase);value.setdefault('leases',{})[phase]=time.time()+180;atomic(folder/'phase-control.json',value)
 # Grant is durable first. Manager can recover the queue transition after a crash.
 state.update(phase='queued_'+phase);atomic(folder/'state.json',state)
 return status(folder)

def recover_grant(folder,state):
 if not managed(folder):return state
 value=control(folder)
 if state['phase'].startswith('awaiting_') and state['phase'].endswith('_authorization'):
  phase=state['phase'][9:-14]
  if phase in value['authorized'] and phase not in value['completed']:
   state['phase']='queued_'+phase;atomic(folder/'state.json',state)
 return state


def cancelled(folder):return managed(folder) and bool(control(folder).get('cancel_requested'))

def abort(folder,body):
 if not isinstance(body,dict) or set(body)!={'run_uid'} or not isinstance(body['run_uid'],str) or not UID.fullmatch(body['run_uid']):raise ValueError('fixed abort fields required')
 if not managed(folder):raise ValueError('legacy job cannot be aborted by Tekton')
 value=control(folder);state=json.loads((folder/'state.json').read_text())
 if value['run_uid'] not in (None,body['run_uid']):raise ValueError('job bound to another run')
 if state['phase']=='complete':return status(folder)
 value.update(run_uid=body['run_uid'],cancel_requested=True);value.setdefault('cancel_requested_at',time.time())
 atomic(folder/'phase-control.json',value);(folder/'pause.request').touch()
 return status(folder)

def finish_cancel(folder,state):
 state.update(phase='failed',reason='tekton_cancelled',last_error='tekton_cancelled',cancel_finished_at=time.time())
 atomic(folder/'state.json',state)
