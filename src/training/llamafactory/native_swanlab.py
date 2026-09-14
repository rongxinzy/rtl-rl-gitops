"""Native SDK run ownership and LF callback integration; no credentials in recipe config."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re

SDK_VERSION='0.10.0'
KEY_PATH=Path('/run/secrets/swanlab-api-key')

def settings(path):
 value=json.loads(Path(path).read_text())
 if set(value)!={'project','workspace','job_id','device_label'}:raise ValueError('invalid native telemetry settings')
 for key,val in value.items():
  pattern=r'[A-Za-z0-9][A-Za-z0-9_. -]{0,127}' if key=='device_label' else r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}'
  if not isinstance(val,str) or not re.fullmatch(pattern,val) or val!=val.strip():raise ValueError('invalid native telemetry identity')
 return value

def run_identity(job_sha,meta):
 if not re.fullmatch('[0-9a-f]{64}',job_sha):raise ValueError('invalid job binding')
 return 'lf-native-'+hashlib.sha256((meta['workspace']+'/'+meta['project']+'/'+meta['job_id']+'/'+job_sha).encode()).hexdigest()[:24]

class NativeRun:
 def __init__(self,sdk,out,identity):self.sdk=sdk;self.out=out;self.identity=identity
 def event(self,phase,step):
  # Text originates exclusively from the finite program state machine.
  if phase not in ('started','resumed','checkpoint','paused','complete','failed'):raise ValueError('unknown telemetry event')
  try:self.sdk.log({'lifecycle/event':self.sdk.Text(phase),'lifecycle/step':int(step)},step=int(step))
  except Exception:raise RuntimeError('native telemetry event failed') from None
 def finish(self,phase,step):
  self.event(phase,step)
  try:self.sdk.finish(state='success' if phase=='complete' else 'aborted' if phase=='paused' else 'crashed')
  except Exception:raise RuntimeError('native telemetry finish failed') from None
  self._status(phase,step)
 def _status(self,phase,step):
  path=self.out/'swanlab-native.json';tmp=path.with_suffix('.tmp')
  tmp.write_text(json.dumps({**self.identity,'phase':phase,'step':step},sort_keys=True));tmp.replace(path)

def start(out,job_sha,meta,training,resume_step,sdk=None,key_path=KEY_PATH,proxy_path=Path('/run/secrets/swanlab-proxy')):
 out=Path(out);identity={'owner':'llamafactory-native','run_id':run_identity(job_sha,meta),'project':meta['project'],'workspace':meta['workspace'],'job_id':meta['job_id'],'job_sha256':job_sha,'sdk_version':SDK_VERSION}
 path=out/'swanlab-native.json'
 if path.exists():
  prior=json.loads(path.read_text())
  if any(prior.get(k)!=v for k,v in identity.items()):raise ValueError('native run binding changed')
 if sdk is None:
  import swanlab as sdk
 if sdk.__version__!=SDK_VERSION:raise ValueError('native SDK version mismatch')
 public={k:training[k] for k in ('backend','model_revision','dataset_id','max_steps','max_length','rank','learning_rate','seed','llamafactory_commit') if k in training}
 acceptance=training.get('acceptance_only') is True
 public.update(acceptance_only=acceptance,task_type='backend-acceptance' if acceptance else 'sft',task_description='Official Qwen RTL NF4 QLoRA supervised training with independent frozen evaluation',job_id=meta['job_id'],job_sha256=job_sha,device_label=meta['device_label'],telemetry_owner='llamafactory-native')
 if acceptance:public['task_description']='CPU-only native SwanLab callback and cloud resume acceptance; synthetic metrics, no model training or capability result'
 try:
  # Neither login output nor exceptions may expose the mounted credential.
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   proxy=Path(proxy_path).read_text().strip()
   if not proxy.startswith(('http://','https://')):raise ValueError('invalid telemetry proxy')
   os.environ['HTTPS_PROXY']=proxy;os.environ['HTTP_PROXY']=proxy
   key=Path(key_path).read_text().strip()
   if not key:raise ValueError('empty telemetry credential')
   sdk.login(api_key=key,save=False);del key
   sdk.init(id=identity['run_id'],resume='allow',project=meta['project'],workspace=meta['workspace'],name=meta['job_id'],job_type=public['task_type'],group='rtl-llamafactory',description=public['task_description'],tags=['rtl','backend-acceptance' if acceptance else 'sft','llamafactory','official-qwen','nf4','qlora',meta['device_label']],config=public,log_dir=str(out/'swanlab'),settings=sdk.Settings(terminal={'proxy_type':'none'},probe={'hardware':True,'runtime':False,'requirements':False,'git':False,'swanlab':False,'monitor':True}),mode='online')
 except Exception:
  raise RuntimeError('native telemetry initialization failed') from None
 run=NativeRun(sdk,out,identity);run._status('resumed' if resume_step else 'started',resume_step);run.event('resumed' if resume_step else 'started',resume_step)
 return run

def install_callback_guard(tuner):
 """Keep LF's actual callback, but prohibit implicit full config serialization."""
 original=tuner.get_swanlab_callback
 def factory(args):
  if getattr(args,'swanlab_api_key',None) is not None:raise ValueError('credentials must not enter finetuning arguments')
  callback=original(args)
  if not hasattr(callback,'_log_config'):raise RuntimeError('native callback version mismatch')
  callback._log_config=False
  return callback
 tuner.get_swanlab_callback=factory
 return original
