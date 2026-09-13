"""Verify immutable model lineage, complete checkpoints and bounded eval outputs."""
import importlib.util,json,re,sys
from pathlib import Path
try:from .common import sha,REVISION,atomic,canonical
except ImportError:from common import sha,REVISION,atomic,canonical

def module(folder,name):
 path=folder/'recipe'/ (name+'.py')
 job=json.loads((folder/'job.json').read_text())
 if path.is_symlink() or sha(path.read_bytes())!=job['recipe_sha256'].get(name+'.py'):raise ValueError('recipe integrity changed')
 # Provenance imports the matching pinned state helper; never a global unrelated module.
 spec=importlib.util.spec_from_file_location('l20_pinned_'+name,path)
 result=importlib.util.module_from_spec(spec)
 previous=sys.modules.get('state')
 try:
  if name!='state':sys.modules['state']=module(folder,'state')
  spec.loader.exec_module(result)
 finally:
  if name!='state':
   if previous is None:sys.modules.pop('state',None)
   else:sys.modules['state']=previous
 return result

def model_path(cfg,folder):
 ready=Path(cfg['model_ready'])
 if not ready.exists():return None
 metadata=json.loads(ready.read_text())
 path=Path(metadata['path'])
 if cfg.get('model_revision')!=REVISION or path.is_symlink():raise ValueError('invalid model provenance configuration')
 if not path.is_dir():return None
 module(folder,'provenance').verify_model(path)
 binding={'model_manifest_sha256':sha((path/'verification.json').read_bytes())}
 target=folder/'model-binding.json'
 if target.exists() and json.loads(target.read_text())!=binding:raise ValueError('model changed across phases')
 if not target.exists():atomic(target,binding)
 return path

def adapter_digest(folder):
 return sha(canonical({n:sha((folder/'run/adapter'/n).read_bytes()) for n in ('adapter_config.json','adapter_model.safetensors')}))

def generation_binding(folder,phase,row):
 job=json.loads((folder/'job.json').read_text())
 expected={'job_sha256':sha((folder/'job.json').read_bytes()),'model_manifest_sha256':json.loads((folder/'model-binding.json').read_text())['model_manifest_sha256'],'generation_recipe_sha256':job['recipe_sha256']['generate_eval.py'],'adapter_sha256':adapter_digest(folder) if phase=='candidate' else None}
 if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('generation immutable binding mismatch')

def verify_checkpoint(folder):
 run=folder/'run';name=(run/'latest').read_text().strip()
 if not re.fullmatch(r'checkpoint-[0-9]{6}',name):raise ValueError('invalid checkpoint pointer')
 helper=module(folder,'state')
 return helper.verify_checkpoint(run/name,helper.digest(run/'job.json'))

def verify_phase(folder,phase):
 job=json.loads((folder/'job.json').read_text())
 if phase=='training':
  run=folder/'run'
  identity=json.loads((run/'job.json').read_text())
  if any(identity.get(k)!=job[k] for k in ('dataset_id','data_sha256','max_steps','model_revision')) or identity.get('max_length')!=1024 or identity.get('rank')!=8:raise ValueError('training identity mismatch')
  if any(job['recipe_sha256'].get(k)!=v for k,v in identity.get('recipe_sha256',{}).items()):raise ValueError('training recipe mismatch')
  if identity.get('model_manifest_sha256')!=json.loads((folder/'model-binding.json').read_text())['model_manifest_sha256']:raise ValueError('training model binding mismatch')
  status=json.loads((run/'status.json').read_text());checkpoint=verify_checkpoint(folder)
  if status.get('status') not in ('complete','paused') or type(status.get('step')) is not int or status['step']!=checkpoint['step'] or status.get('job_sha256')!=checkpoint['job_sha256']:raise ValueError('inconsistent training completion')
  if status['status']=='complete' and status['step']<job['max_steps']:raise ValueError('premature completion')
  for name in ('trainer_state_final.json','training_metrics.json'):
   data=json.loads((run/name).read_text())
   if data.get('global_step')!=status['step'] or data.get('job_sha256')!=status['job_sha256']:raise ValueError('inconsistent final metrics')
  for name in ('adapter_config.json','adapter_model.safetensors'):
   path=run/'adapter'/name
   if path.is_symlink() or not path.is_file() or path.stat().st_size==0:raise ValueError('missing adapter')
   if sha(path.read_bytes())!=checkpoint['files'][name]:raise ValueError('adapter/checkpoint mismatch')
  return status
 path=folder/(phase+'.jsonl');raw=path.read_bytes()
 if path.is_symlink() or len(raw)>256*1024:raise ValueError('invalid eval artifact')
 rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
 if sha((folder/'prompts.jsonl').read_bytes())!=job['prompts_sha256']:raise ValueError('prompts changed')
 prompts=[json.loads(line) for line in (folder/'prompts.jsonl').read_text().splitlines() if line.strip()]
 if len(rows)!=3 or {r['task_id'] for r in rows}!={r['task_id'] for r in prompts}:raise ValueError('incomplete evaluation')
 for row in rows:
  generation_binding(folder,phase,row)
  if row.get('model_revision')!=REVISION or row.get('prompts_sha256')!=job['prompts_sha256'] or row.get('quantization')!='bnb-nf4':raise ValueError('evaluation identity mismatch')
  if not isinstance(row.get('content'),str) or row.get('finish_reason') not in ('stop','length'):raise ValueError('invalid generation')
  if row.get('adapter')!=('/job/run/adapter' if phase=='candidate' else None):raise ValueError('adapter provenance mismatch')
 quiz=json.loads((folder/'job.json').read_text())
 if 'knowledge_prompts_sha256' in quiz:
  verify_knowledge(folder,phase,quiz)
 return rows

def verify_all(folder):
 for phase in ('baseline','training','candidate'):verify_phase(folder,phase)


def verify_knowledge(folder,phase,job):
 path=folder/('knowledge_'+phase+'.jsonl');raw=path.read_bytes()
 if path.is_symlink() or len(raw)>256*1024:raise ValueError('invalid knowledge generations')
 rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
 if sha((folder/'knowledge_prompts.jsonl').read_bytes())!=job['knowledge_prompts_sha256']:raise ValueError('knowledge prompts changed')
 prompts=[json.loads(x) for x in (folder/'knowledge_prompts.jsonl').read_text().splitlines()]
 if len(rows)!=10 or len({r['task_id'] for r in rows})!=10 or {r['task_id'] for r in rows}!={p['task_id'] for p in prompts}:raise ValueError('knowledge generation incomplete')
 for r in rows:
  generation_binding(folder,phase,r)
  if r.get('model_revision')!=REVISION or r.get('prompts_sha256')!=job['knowledge_prompts_sha256'] or r.get('adapter')!=(None if phase=='baseline' else '/job/run/adapter') or r.get('quantization')!='bnb-nf4':raise ValueError('knowledge generation provenance')
  if not isinstance(r.get('content'),str) or r.get('finish_reason') not in ('stop','length'):raise ValueError('invalid knowledge generation')
 return rows
