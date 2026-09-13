"""Fixed local job identities for the L20 training branch."""
import hashlib,json,os,re,shutil,time,threading
from pathlib import Path
ROOT=Path(os.environ.get('L20_WORK_ROOT','/mnt/data/rtl-l20-training/worker'))
LOCK=threading.RLock()
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
PROFILE=re.compile(r'^[a-z][a-z0-9-]{0,63}$')
HEX=re.compile(r'^[0-9a-f]{64}$');ID=re.compile(r'^l20-[0-9a-f]{24}$')
def sha(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def config():return json.loads((ROOT/'config.json').read_text())
def validate(body):
 required={'dataset_id','freeze_id','data_sha256','prompts_sha256','data','prompts','max_steps','knowledge_prompts','knowledge_prompts_sha256','knowledge_freeze_id'}
 if not isinstance(body,dict) or not required<=set(body) or set(body)-required-{'profile_id'}:raise ValueError('fixed job fields required')
 if 'profile_id' in body and (not isinstance(body['profile_id'],str) or not PROFILE.fullmatch(body['profile_id'])):raise ValueError('invalid training profile')
 for name in ('dataset_id','freeze_id','data_sha256','prompts_sha256','knowledge_prompts_sha256','knowledge_freeze_id'):
  if not isinstance(body[name],str) or not HEX.fullmatch(body[name]):raise ValueError('invalid artifact identity')
 if type(body['max_steps']) is not int or not 1<=body['max_steps']<=20:raise ValueError('step budget')
 for name in ('data','prompts','knowledge_prompts'):
  if not isinstance(body[name],str) or len(body[name].encode())>2*1024*1024 or sha(body[name])!=body[name+'_sha256']:raise ValueError('invalid artifact contents')
 rows=[json.loads(x) for x in body['data'].splitlines() if x.strip()]
 if not 8<=len(rows)<=256 or len({x['task_id'] for x in rows})!=len(rows):raise ValueError('8..256 distinct tasks required')
 for row in rows:
  if row.get('split')!='train' or row.get('validation_level') not in ('Q2','K1-grounded'):raise ValueError('only trusted train split')
  if row['validation_level']=='K1-grounded' and (row.get('kind') not in ('explain','predict','repair') or any(not isinstance(row.get(k),str) or not HEX.fullmatch(row[k]) for k in ('knowledge_evidence_sha256','knowledge_source_sha256'))):raise ValueError('knowledge evidence missing')
 prompts=[json.loads(x) for x in body['prompts'].splitlines() if x.strip()]
 if len(prompts)!=3 or len({x['task_id'] for x in prompts})!=3 or any(set(x)!={'task_id','spec'} for x in prompts):raise ValueError('three prompt-only evaluation rows required')
 if {x['task_id'] for x in rows}&{x['task_id'] for x in prompts}:raise ValueError('evaluation overlap')
 quiz=[json.loads(x) for x in body['knowledge_prompts'].splitlines() if x.strip()]
 if len(quiz)!=10 or len({x['task_id'] for x in quiz})!=10 or any(set(x)!={'task_id','question','code','output_schema'} for x in quiz):raise ValueError('ten closed-book prompt-only records required')
 if {x['task_id'] for x in rows}&{x['task_id'] for x in quiz}:raise ValueError('knowledge evaluation overlap')
 return body

def training_profile(cfg,profile_id=None):
 # Only administrators select paths/images in local configuration. Clients send an ID.
 if profile_id is None:return cfg
 profiles=cfg.get('training_profiles',{})
 selected=profiles.get(profile_id) if isinstance(profiles,dict) else None
 if not isinstance(selected,dict) or set(selected)!={'image_id','recipe_path','recipe_sha256'}:raise ValueError('unknown or invalid training profile')
 if not isinstance(selected['image_id'],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',selected['image_id']):raise ValueError('invalid profile image')
 source=selected['recipe_path']
 if not isinstance(source,str) or not Path(source).is_absolute() or not Path(source).is_dir() or Path(source).is_symlink():raise ValueError('invalid profile recipe')
 if not isinstance(selected['recipe_sha256'],dict) or not selected['recipe_sha256'] or any(not isinstance(k,str) or not re.fullmatch(r'[A-Za-z0-9_]+\.py',k) or k.startswith('test_') or not isinstance(v,str) or not HEX.fullmatch(v) for k,v in selected['recipe_sha256'].items()):raise ValueError('invalid profile recipe hashes')
 return {**cfg,**selected}

def admit(body):
 validate(body);cfg=training_profile(config(),body.get('profile_id'))
 if cfg.get('model_revision')!=REVISION or not re.fullmatch(r'sha256:[0-9a-f]{64}',cfg.get('image_id','')):raise ValueError('untrusted model/image configuration')
 meta={k:v for k,v in body.items() if k not in ('data','prompts','knowledge_prompts')}
 meta.update(image_id=cfg['image_id'],model_revision=cfg['model_revision'])
 mode=cfg.get('orchestration_mode','legacy')
 if mode not in ('legacy','tekton'):raise ValueError('invalid orchestration mode')
 if mode=='tekton':meta['orchestrator']='tekton'
 source=Path(cfg['recipe_path'])
 if 'profile_id' in body and any(p.is_symlink() or not p.is_file() for p in source.glob('*.py')):raise ValueError('invalid profile recipe files')
 recipe={p.name:sha(p.read_bytes()) for p in source.glob('*.py') if not p.name.startswith('test_')}
 if 'profile_id' in body and (recipe!=cfg['recipe_sha256'] or any(p.is_symlink() for p in source.glob('*.py'))):raise ValueError('invalid profile recipe files')
 meta['recipe_sha256']=recipe;ident='l20-'+sha(canonical(meta))[:24];folder=ROOT/'jobs'/ident
 if folder.exists():
  if json.loads((folder/'job.json').read_text())!=meta:raise ValueError('job identity conflict')
  return {'status':'already_admitted','job_id':ident}
 if shutil.disk_usage(ROOT).free<20*1024**3:raise ValueError('need 20GiB local free disk')
 active=[p for p in (ROOT/'jobs').glob('l20-*/state.json') if json.loads(p.read_text()).get('phase') not in ('complete','failed')]
 if active:raise ValueError('another L20 experiment is pending')
 stage=ROOT/'jobs'/('.'+ident);stage.mkdir(parents=True,exist_ok=True)
 (stage/'recipe').mkdir(exist_ok=True)
 for name,expected in recipe.items():
  raw=(source/name).read_bytes()
  if sha(raw)!=expected:raise ValueError('recipe changed')
  (stage/'recipe'/name).write_bytes(raw)
 for name in ('data','prompts','knowledge_prompts'):(stage/(name+'.jsonl')).write_text(body[name])
 atomic(stage/'job.json',meta);atomic(stage/'state.json',{'phase':'queued','created_at':time.time(),'attempts':0})
 stage.rename(folder)
 return {'status':'admitted','job_id':ident}

def status():
 rows=[]
 for folder in sorted((ROOT/'jobs').glob('l20-*'),key=lambda p:p.stat().st_mtime):
  try:
   job=json.loads((folder/'job.json').read_text());state=json.loads((folder/'state.json').read_text())
   binding=json.loads((folder/'phase-control.json').read_text()) if (folder/'phase-control.json').exists() else {}
   rows.append({'job_id':folder.name,'orchestrator':job.get('orchestrator','legacy'),'run_uid':binding.get('run_uid'),**{k:job[k] for k in ('dataset_id','freeze_id','max_steps')},**state})
  except (OSError,ValueError,KeyError):continue
 return {'enabled':(ROOT/'enabled').exists(),'paused':(ROOT/'pause').exists(),'jobs':rows[-20:],'current_job':rows[-1] if rows else None}


def compare_complete(folder,body):
 if not isinstance(body,dict) or set(body)!={'comparison_id','outcome'} or not isinstance(body['comparison_id'],str) or not HEX.fullmatch(body['comparison_id']) or body['outcome'] not in ('improved','unchanged','regressed'):raise ValueError('invalid comparison')
 state=json.loads((folder/'state.json').read_text())
 if state['phase']=='complete':
  if state.get('comparison')!=body:raise ValueError('comparison already committed')
  return {'status':'complete','job_id':folder.name}
 from importlib import import_module
 phases=import_module((__package__+'.phase_control') if __package__ else 'phase_control')
 if phases.cancelled(folder):raise ValueError('job cancellation requested')
 if state['phase']!='awaiting_evaluation':raise ValueError('not awaiting evaluation')
 artifacts=import_module((__package__+'.artifacts') if __package__ else 'artifacts')
 artifacts.verify_all(folder)
 state.update(phase='complete',comparison=body,finished_at=time.time());atomic(folder/'state.json',state)
 return {'status':'complete','job_id':folder.name}
