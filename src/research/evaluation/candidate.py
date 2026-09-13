"""Fixed completed-adapter evaluation and deterministic same-freeze comparison."""
import hashlib,json,pathlib,re,time,datetime
from core import MODEL_REVISION,canonical,sha,publish,classification

def file_hash(path):
 value=hashlib.sha256()
 with pathlib.Path(path).open('rb') as source:
  for chunk in iter(lambda:source.read(16*1024*1024),b''):value.update(chunk)
 return value.hexdigest()

def adapter_identity(directory):
 directory=pathlib.Path(directory)
 if directory.is_symlink():raise ValueError('adapter directory symlink rejected')
 hashes={}
 for name in ['adapter_config.json','adapter_model.safetensors']:
  path=directory/name
  if path.is_symlink() or not path.is_file():raise ValueError('missing or symlinked adapter file')
  hashes[name]=file_hash(path)
 return sha(canonical(hashes)),hashes

def validate_job(root,job_id):
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',job_id):raise ValueError('invalid job id')
 runs=pathlib.Path(root)/'runs';path=runs/job_id
 if path.is_symlink() or path.resolve().parent!=runs.resolve():raise ValueError('job path escape rejected')
 if not all((path/name).is_file() for name in ['job-config.json','last-exit.json','trainer_state_final.json']):raise ValueError('training completion artifacts absent')
 cfg=json.loads((path/'job-config.json').read_text());exit_state=json.loads((path/'last-exit.json').read_text());state=json.loads((path/'trainer_state_final.json').read_text())
 if cfg.get('job_id')!=job_id or cfg.get('model_revision')!=MODEL_REVISION:raise ValueError('job/model identity mismatch')
 recipe=cfg.get('recipe_sha256',{})
 if not recipe or not all(re.fullmatch(r'[0-9a-f]{64}',v) for v in recipe.values()):raise ValueError('missing pinned training recipe hashes')
 if exit_state.get('returncode')!=0 or exit_state.get('step',-1)<cfg['max_steps'] or state.get('global_step',-1)<cfg['max_steps']:raise ValueError('training has not completed successfully')
 identity,files=adapter_identity(path/'adapter')
 return path,cfg,{'job_id':job_id,'adapter_sha256':identity,'adapter_files':files,'job_config_sha256':file_hash(path/'job-config.json'),'recipe_sha256':sha(canonical(recipe)),'recipe_file_sha256':recipe}

def compare(baseline,candidate):
 for result in [baseline,candidate]:
  actual={k:0 for k in ['pass','fail','infrastructure','unknown']}
  for row in result['results']:
   label=classification(row['report'])
   if label!=row['classification']:raise ValueError('classification/report mismatch')
   actual[label]+=1
  if actual!=result['counts']:raise ValueError('counts/results mismatch')
 expected={x['task_id'] for x in baseline['results']}
 if len(expected)!=len(baseline['results']) or len({x['task_id'] for x in candidate['results']})!=len(candidate['results']):raise ValueError('duplicate task IDs')
 if baseline.get('metadata',{}).get('training_image_id')!=candidate.get('metadata',{}).get('training_image_id'):raise ValueError('evaluation runtime image mismatch')
 if baseline['freeze_id']!=candidate['freeze_id'] or baseline['model_revision']!=candidate['model_revision']:raise ValueError('comparison version mismatch')
 if expected!={x['task_id'] for x in candidate['results']}:raise ValueError('comparison task mismatch')
 old={x['task_id']:x for x in baseline['results']};regressions=[]
 for row in candidate['results']:
  prior=old[row['task_id']]
  for field in ['spec_sha256','reference_sha256','testbench_sha256']:
   if row[field]!=prior[field]:raise ValueError('comparison source mismatch')
  if prior['classification']=='pass' and row['classification']!='pass':regressions.append(row['task_id'])
 infra=sum(r['counts']['infrastructure'] for r in [baseline,candidate]);unknown=sum(r['counts']['unknown'] for r in [baseline,candidate])
 delta=candidate['counts']['pass']-baseline['counts']['pass'];complete=infra==0 and unknown==0
 outcome='inconclusive' if not complete else 'regressed' if regressions or delta<0 else 'improved' if delta>0 else 'unchanged'
 return {'comparison_complete':complete,'infrastructure_errors':infra,'unknown_results':unknown,'pass_delta':delta,'canary_regression':bool(regressions),'canary_regressions':regressions,'outcome':outcome}

def evaluation_pins(path,cfg,schedule,identity,freeze_id):
 job_id=identity['job_id'];current=schedule if schedule.get('job_id')==job_id else {}
 expected_id=cfg.get('baseline_evaluation_id') or current.get('baseline_evaluation_id')
 expected_freeze=cfg.get('evaluation_freeze_id') or current.get('evaluation_freeze_id')
 pending_path=path/'evaluation-pending.json'
 if pending_path.is_symlink():raise ValueError('pending evaluation symlink rejected')
 if pending_path.exists():
  pending=json.loads(pending_path.read_text())
  for key,value in [('job_id',job_id),('adapter_sha256',identity['adapter_sha256']),('freeze_id',freeze_id)]:
   if (key=='job_id' or key in pending) and pending.get(key)!=value:raise ValueError('pending evaluation identity mismatch: '+key)
  pending_id=pending.get('baseline_evaluation_id')
  if 'baseline_evaluation_id' in pending and (not isinstance(pending_id,str) or not re.fullmatch(r'[0-9a-f]{64}',pending_id)):raise ValueError('invalid pending baseline pin')
  if expected_id and pending_id and expected_id!=pending_id:raise ValueError('pending baseline conflicts with job pin')
  expected_id=expected_id or pending_id
  if not expected_id:raise ValueError('pending evaluation requires a pinned baseline')
 if expected_freeze and expected_freeze!=freeze_id:raise ValueError('scheduled evaluation freeze mismatch')
 return expected_id

def evaluate(a,bundle,run_local):
 root=pathlib.Path(a.root);schedule=json.loads((root/'scheduling/job.json').read_text());job_id=a.job_id or schedule['job_id']
 path,cfg,identity=validate_job(root,job_id)
 authority=root/'state/operator-authority.json'
 deadline=a.deadline
 if deadline is None and authority.exists():deadline=json.loads(authority.read_text()).get('deadline')
 if deadline is None:deadline=json.loads((path/'last-start.json').read_text()).get('deadline')
 if deadline is None or deadline-time.time()<1800:raise ValueError('less than 30 minutes before controller deadline')
 expected_id=evaluation_pins(path,cfg,schedule,identity,bundle['freeze_id'])
 runtime=cfg.get('image_id')
 if not isinstance(runtime,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',runtime):raise ValueError('missing immutable job runtime image')
 requested=getattr(a,'runtime_image_id',None)
 if requested is not None and requested!=runtime:raise ValueError('evaluation image conflicts with job pin')
 identity['training_image_id']=runtime
 if cfg.get('backend') is not None:identity['training_backend']=cfg['backend']
 baselines=[]
 for p in sorted((root/'research/evaluation/artifacts/runs').glob('qwen-base-*/result.json')):
  b=json.loads(p.read_text())
  if b.get('kind')=='qwen_base_baseline' and b.get('freeze_id')==bundle['freeze_id'] and b.get('model_revision')==MODEL_REVISION and (not expected_id or file_hash(p)==expected_id) and b.get('metadata',{}).get('training_image_id')==runtime:baselines.append((p,b))
 if not baselines:raise ValueError('no matching frozen Qwen base baseline')
 baseline_path,baseline=baselines[-1]
 identity.update(dataset_id=cfg.get('dataset_id') or (schedule.get('dataset_id') if schedule['job_id']==job_id else None))
 output=root/'research/evaluation/artifacts/comparisons'/f"{job_id}-{identity['adapter_sha256'][:16]}-{bundle['freeze_id'][:12]}-{file_hash(baseline_path)[:12]}.json"
 if output.exists():
  cached=json.loads(output.read_text())
  if any(cached.get(k)!=identity[k] for k in ['job_id','adapter_sha256','job_config_sha256','recipe_sha256']):raise ValueError('cached comparison identity mismatch')
  if file_hash(cached['baseline_result_path'])!=cached['baseline_evaluation_id'] or file_hash(cached['candidate_result_path'])!=cached['candidate_evaluation_id']:raise ValueError('cached evaluation artifact changed')
  old_base=json.loads(pathlib.Path(cached['baseline_result_path']).read_text())
  old_candidate=json.loads(pathlib.Path(cached['candidate_result_path']).read_text())
  if old_base.get('metadata',{}).get('training_image_id')!=runtime or old_candidate.get('metadata',{}).get('training_image_id')!=runtime:raise ValueError('cached comparison runtime mismatch')
  checked=compare(old_base,old_candidate)
  if any(cached.get(k)!=v for k,v in checked.items()):raise ValueError('cached comparison outcome mismatch')
  return {'status':'complete','comparison_path':str(output),'cached':True,**cached}
 candidate_summary=run_local(a,bundle,adapter=path/'adapter',metadata=identity)
 candidate_path=pathlib.Path(candidate_summary['result_path']);candidate=json.loads(candidate_path.read_text())
 if adapter_identity(path/'adapter')[0]!=identity['adapter_sha256']:raise ValueError('adapter changed during evaluation')
 result={**identity,**compare(baseline,candidate),'schema_version':1,'freeze_id':bundle['freeze_id'],'model_revision':MODEL_REVISION,'baseline_evaluation_id':file_hash(baseline_path),'candidate_evaluation_id':file_hash(candidate_path),'baseline_result_path':str(baseline_path),'candidate_result_path':str(candidate_path),'scope':'internal 3val smoke only; not a production release gate','created_at':time.time()}
 if not result['comparison_complete']:output=output.with_name(output.stem+'-inconclusive-'+str(time.time_ns())+'.json')
 publish(output,result)
 return {'status':'complete','comparison_path':str(output),**result}
