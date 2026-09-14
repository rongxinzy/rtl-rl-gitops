"""Fixed-service GPU rotation, reconciled under the existing worker lock."""
import json,time,subprocess,urllib.request,hashlib
from pathlib import Path
try:
 from .common import ROOT,config,atomic
except ImportError:
 from common import ROOT,config,atomic
SERVICE='rtl-backup-glm.service'
BACKUP=Path('/root/rtl-rl/backup-glm')
PHASES={'training','pausing','starting_inference','inference_ready','stopping_inference','blocked','error'}
def command(args):return subprocess.run(args,capture_output=True,text=True,timeout=15)
def read():
 path=ROOT/'rotation.json'
 return json.loads(path.read_text()) if path.exists() else {'requested_role':'training','phase':'training','attempts':0}
def status():
 state=read();paused=(ROOT/'rotation-pause').exists()
 return {'requested_role':state['requested_role'],'phase':state['phase'],'ready':state['phase']=='inference_ready' and healthy(),'training_allowed':not paused and not (ROOT/'pause').exists() and (ROOT/'enabled').exists(),'rotation_paused':paused,'enabled':bool(config().get('rotation_enabled',False))}
def request(body):
 if not isinstance(body,dict) or set(body)!={'role'} or body['role'] not in ('inference','training'):raise ValueError('invalid rotation request')
 if not config().get('rotation_enabled',False):raise ValueError('rotation disabled')
 state=read()
 if state['requested_role']!=body['role']:
  state={'requested_role':body['role'],'phase':'pausing' if body['role']=='inference' else 'stopping_inference','attempts':0}
  atomic(ROOT/'rotation.json',state)
 # The request itself closes admission before the next reconciliation.
 if body['role']=='inference':(ROOT/'rotation-pause').touch()
 return status()
def service_state():
 r=command(['systemctl','show',SERVICE,'--property=ActiveState','--value'])
 if r.returncode:raise RuntimeError('service_state_unavailable')
 value=r.stdout.strip()
 if value not in ('active','activating','deactivating','inactive','failed'):raise RuntimeError('service_state_unknown')
 return value
def gpu_free():
 r=command(['querygpu','--query-compute-apps=pid','--format=csv,noheader,nounits'])
 if r.returncode or r.stdout.strip():return False
 r=command(['querygpu','--query-gpu=memory.used','--format=csv,noheader,nounits'])
 try:return r.returncode==0 and len(r.stdout.splitlines())==2 and all(int(x.strip())<512 for x in r.stdout.splitlines())
 except ValueError:return False
def prepared():
 try:
  model=json.loads((BACKUP/'model-manifest.json').read_text());build=json.loads((BACKUP/'BUILD_READY.json').read_text());ready=json.loads((BACKUP/'readiness/READY.json').read_text())
  revision='621d456e93e926e4b52f85cff5f634358c1828f9';engine='d94f44e79aa219d8057e8de21f95360a187ebf41'
  if model['revision']!=revision or ready['model_revision']!=revision or ready['engine_revision']!=engine or build['source_revision']!=engine or ready['quantization']!='UD-IQ1_S':return False
  if (BACKUP/'llama.cpp/PINNED_REVISION').read_text().strip()!=engine:return False
  checks={'chat_text','responses_json','responses_stream','responses_tool_stream','responses_tool_roundtrip'}
  if not checks.issubset(ready['protocol_checks']):return False
  verified=json.loads((BACKUP/'models'/revision/'verification.json').read_text())
  if verified.get('sha256_verified') is not True or verified['revision']!=revision or verified['files']!=model['files']:return False
  names={f'GLM-5.3-Flash-UD-IQ1_S-{i:05d}-of-00003.gguf' for i in range(1,4)}
  if {Path(x['path']).name for x in model['files']}!=names:return False
  if any((BACKUP/'models'/revision/Path(x['path']).name).stat().st_size!=x['size'] for x in model['files']):return False
  if not build['files']:return False
  for name,digest in build['files'].items():
   path=(BACKUP/name).resolve()
   if not path.is_relative_to((BACKUP/'llama.cpp/build/bin').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:return False
  return True
 except (OSError,ValueError,KeyError,TypeError):return False
def healthy():
 try:
  with urllib.request.urlopen('http://127.0.0.1:18001/health',timeout=3) as response:return response.status==200
 except Exception:return False
def reconcile(inspect):
 if not config().get('rotation_enabled',False):return
 state=read();now=time.time()
 def save(phase):state['phase']=phase;atomic(ROOT/'rotation.json',state)
 try:
  active=service_state()
  if state['requested_role']=='training':
   if active not in ('inactive','failed'):
    (ROOT/'rotation-pause').touch()
    if state.get('stop_sent_at',0)+60<=now:
     result=command(['systemctl','stop','--no-block',SERVICE]);state['stop_sent_at']=now
     if result.returncode:save('error');return
    save('stopping_inference');return
   # Already training: a running owned job is expected, do not block it.
   if not (ROOT/'rotation-pause').exists():save('training');return
   if not gpu_free():save('blocked');return
   (ROOT/'rotation-pause').unlink(missing_ok=True);save('training');return
  (ROOT/'rotation-pause').touch()
  container=inspect()
  if container is not None:
   # Manager validates ownership and handles checkpoint/exit/removal.
   save('pausing');return
  if active in ('active','activating'):
   if healthy():save('inference_ready');return
   if now-state.get('start_sent_at',now)>900:save('error');return
   save('starting_inference');return
  if active=='deactivating':save('blocked');return
  if state.get('attempts',0)>=3:save('error');return
  if state.get('retry_at',0)>now:save('starting_inference');return
  if not gpu_free() or not prepared():save('blocked');return
  state['attempts']=state.get('attempts',0)+1;state['retry_at']=now+300;state['start_sent_at']=now
  # Persist before invoking systemd: uncertain responses never cause rapid retries.
  save('starting_inference')
  result=command(['systemctl','start','--no-block',SERVICE])
  if result.returncode:save('error')
 except Exception:save('error')
