"""Validated artifacts cross the L20 boundary; model proposals never carry commands."""
import json,hashlib,urllib.request,importlib.util,sys,subprocess,re,time
from pathlib import Path
from .core import atomic,digest,HEX

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args):return None
OPENER=urllib.request.build_opener(NoRedirect())
JOB_ID=re.compile(r'^l20-[0-9a-f]{24}$')

def call(executor,path,body=None):
 token=(executor.root/'secrets/l20-worker-token').read_text().strip()
 req=urllib.request.Request('http://172.18.6.123:18766'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
 with OPENER.open(req,timeout=3 if path=='/status' else 15) as response:
  raw=response.read(8*1024*1024+1)
  if len(raw)>8*1024*1024:raise ValueError('oversize worker response')
  return json.loads(raw)

def status(executor):
 try:return call(executor,'/status')
 except Exception as exc:return {'available':False,'error_type':type(exc).__name__}

def dataset(executor,ident):
 from research.data_factory.cli import read_dataset
 folder=executor.root/'research/data_factory/state/datasets'/ident
 if folder.exists():return folder,read_dataset(folder)
 from research.knowledge.factory import read_dataset as read_knowledge
 folder=executor.root/'research/knowledge/artifacts'/ident
 return folder,read_knowledge(folder)

def knowledge_status():
 try:
  from research.knowledge.factory import status
  return status()
 except Exception as exc:return {'ready':False,'error_type':type(exc).__name__,'datasets':[]}

def admit(executor,params):
 from .evidence import refresh
 from research.data_factory.cli import read_dataset
 refresh(executor)
 folder,manifest=dataset(executor,params['dataset_id'])
 if manifest['train']<8:raise ValueError('need 8 distinct verified train tasks')
 baseline=json.loads((executor.state/'verified'/('evaluation-'+params['evaluation_id']+'.json')).read_text())
 if not baseline['baseline_complete']:raise ValueError('baseline incomplete')
 freeze_path=executor.root/'research/evaluation/artifacts/frozen'/(baseline['freeze_id']+'.json')
 frozen=json.loads(freeze_path.read_text())
 data=(folder/'sft_train.jsonl').read_text()
 from research.knowledge.evaluation import reject_training_overlap
 reject_training_overlap([json.loads(x) for x in data.splitlines()])
 check=subprocess.run(['/usr/bin/python3',str(executor.root/'research/evaluation/cli.py'),'check-training','--freeze-file',str(freeze_path),'--training',str(folder/'sft_train.jsonl')],capture_output=True,text=True,timeout=60)
 if check.returncode or json.loads(check.stdout)['status']!='clean':raise ValueError('evaluation contamination')
 previous=call(executor,'/status').get('current_job')
 if previous and previous.get('phase')=='complete' and previous.get('comparison',{}).get('outcome') in ('unchanged','regressed'):
  _,old=dataset(executor,previous['dataset_id'])
  semantics=lambda m:{t['semantic_sha256'] for t in m['tasks'] if t['split']=='train'}
  if semantics(old)==semantics(manifest):raise ValueError('L20 next experiment needs new semantics')
 prompts=''.join(json.dumps({'task_id':t['task_id'],'spec':t['spec']},ensure_ascii=False)+'\n' for t in frozen['tasks'])
 body={'dataset_id':params['dataset_id'],'freeze_id':baseline['freeze_id'],'data_sha256':hashlib.sha256(data.encode()).hexdigest(),'prompts_sha256':hashlib.sha256(prompts.encode()).hexdigest(),'data':data,'prompts':prompts,'max_steps':params.get('max_steps',20)}
 from research.knowledge.evaluation import freeze,prompts as knowledge_prompts
 knowledge=freeze();quiz=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in knowledge_prompts(knowledge))
 body.update(knowledge_freeze_id=knowledge['freeze_id'],knowledge_prompts=quiz,knowledge_prompts_sha256=hashlib.sha256(quiz.encode()).hexdigest())
 result=call(executor,'/jobs',body)
 if not JOB_ID.fullmatch(result.get('job_id','')):raise ValueError('invalid worker job identity')
 out=executor.root/'research/l20_artifacts'/result['job_id'];out.mkdir(parents=True,exist_ok=True)
 atomic(out/'admission.json',{k:v for k,v in body.items() if k not in ('data','prompts','knowledge_prompts')})
 return result

def validate_generations(payload,job,frozen):
 from research.knowledge.evaluation import freeze,prompts
 knowledge=freeze()
 if job['knowledge_freeze_id']!=knowledge['freeze_id']:raise ValueError('knowledge freeze mismatch')
 sets={'': [{'task_id':t['task_id'],'spec':t['spec']} for t in frozen['tasks']], 'knowledge_':prompts(knowledge)}
 for prefix,questions in sets.items():
  expected_hash=hashlib.sha256(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in questions).encode()).hexdigest()
  if job[prefix+'prompts_sha256']!=expected_hash:raise ValueError('untrusted prompt content')
  expected={q['task_id'] for q in questions}
  for phase in ('baseline','candidate'):
   name=prefix+phase;raw=payload[name+'_raw'];rows=payload[name]
   if not isinstance(raw,str) or len(raw.encode())>256*1024 or hashlib.sha256(raw.encode()).hexdigest()!=payload[name+'_sha256']:raise ValueError('generation file hash mismatch')
   if [json.loads(x) for x in raw.splitlines() if x.strip()]!=rows:raise ValueError('generation file content mismatch')
   if len(rows)!=len(expected) or {r['task_id'] for r in rows}!=expected:raise ValueError('incomplete generation')
   identity={'model_revision':job['model_revision'],'quantization':'bnb-nf4','prompts_sha256':expected_hash,'adapter':None if phase=='baseline' else '/job/run/adapter','adapter_sha256':None if phase=='baseline' else payload['adapter_sha256'],'model_manifest_sha256':payload['model_manifest_sha256'],'job_sha256':payload['job_sha256'],'generation_recipe_sha256':job['recipe_sha256']['generate_eval.py']}
   if job['model_revision']!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':raise ValueError('unapproved model revision')
   if any(not HEX.fullmatch(payload[k]) for k in ('adapter_sha256','model_manifest_sha256','job_sha256')):raise ValueError('missing immutable binding')
   for row in rows:
    if any(row.get(k)!=v for k,v in identity.items()):raise ValueError('generation provenance mismatch')
    if not isinstance(row.get('content'),str) or row.get('finish_reason') not in ('stop','length'):raise ValueError('invalid generation')
 return knowledge

def verify_cached(out,report,current):
 if report.get('comparison_complete') is not True or report.get('outcome')!=current['comparison']['outcome']:raise ValueError('incomplete comparison report')
 required={'baseline-generation.json','candidate-generation.json','baseline-judged.json','candidate-judged.json','knowledge-baseline.json','knowledge-candidate.json','payload.json','admission.json'}
 hashes=report.get('artifact_hashes',{})
 if set(hashes)!=required:raise ValueError('missing comparison evidence')
 for name,expected in hashes.items():
  if hashlib.sha256((out/name).read_bytes()).hexdigest()!=expected:raise ValueError('cached evidence changed')
 payload=json.loads((out/'payload.json').read_text());job=payload['job']
 if job['dataset_id']!=report['dataset_id'] or payload['adapter_sha256']!=report['adapter_sha256']:raise ValueError('cached job mismatch')
 from .iteration import comparison_outcome
 before={};after={}
 for label,dest in (('baseline',before),('candidate',after)):
  judged=json.loads((out/(label+'-judged.json')).read_text())
  if judged['counts']['infrastructure'] or judged['counts']['unknown']:raise ValueError('cached evaluation inconclusive')
  dest.update({r['task_id']:r['classification'] for r in judged['results']})
  from research.knowledge.evaluation import freeze,score
  rows=payload['knowledge_'+label]
  scored=score(freeze(),[{'task_id':r['task_id'],'content':r['content'] if r['finish_reason']=='stop' else ''} for r in rows],job['model_revision'])
  if scored!=json.loads((out/('knowledge-'+label+'.json')).read_text()):raise ValueError('cached knowledge score changed')
  dest.update({'knowledge:'+r['task_id']:'pass' if r['passed'] else 'fail' for r in scored['results']})
 if comparison_outcome(before,after)!=report['outcome']:raise ValueError('cached outcome inconsistent')

def compare(executor,job_id=None,run_uid=None):
 if job_id is not None:
  if not JOB_ID.fullmatch(job_id):raise ValueError('invalid bound job identity')
  current=call(executor,'/jobs/'+job_id+'/status')
  if current.get('job_id')!=job_id or current.get('orchestrator')!='tekton' or current.get('run_uid')!=run_uid:raise ValueError('Tekton ownership mismatch')
 else:
  current=call(executor,'/status').get('current_job')
  if current and current.get('orchestrator')=='tekton':raise ValueError('Tekton comparison needs explicit run binding')
 if not current:raise ValueError('No L20 job')
 if not JOB_ID.fullmatch(current.get('job_id','')):raise ValueError('invalid worker job identity')
 if current.get('phase')=='complete':
  report=json.loads((executor.root/'research/l20_artifacts'/current['job_id']/'comparison.json').read_text())
  if report.get('job_id')!=current['job_id'] or digest(report)!=current['comparison']['comparison_id']:raise ValueError('stored comparison integrity mismatch')
  out=executor.root/'research/l20_artifacts'/current['job_id']
  verify_cached(out,report,current)
  payload=json.loads((out/'payload.json').read_text())
  frozen=json.loads((executor.root/'research/evaluation/artifacts/frozen'/(payload['job']['freeze_id']+'.json')).read_text())
  evaluation_path=executor.root/'research/evaluation/cli.py'
  spec=importlib.util.spec_from_file_location('l20_cached_eval_cli',evaluation_path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli);cli.verify(frozen)
  validate_generations(payload,payload['job'],frozen)
  admitted=json.loads((out/'admission.json').read_text())
  if any(payload['job'].get(k)!=v for k,v in admitted.items()):raise ValueError('cached admission changed')
  return {'status':'complete','job_id':current['job_id'],'comparison_complete':True,**current['comparison']}
 if current.get('phase')!='awaiting_evaluation':return {'status':'deferred','reason':'L20 still working'}
 ident=current['job_id'];payload=call(executor,'/jobs/'+ident+'/artifacts');job=payload['job']
 out=executor.root/'research/l20_artifacts'/ident;out.mkdir(parents=True,exist_ok=True)
 admitted=json.loads((out/'admission.json').read_text())
 if any(job[k]!=v for k,v in admitted.items()):raise ValueError('worker job binding mismatch')
 frozen=json.loads((executor.root/'research/evaluation/artifacts/frozen'/(job['freeze_id']+'.json')).read_text())
 path=executor.root/'research/evaluation/cli.py'
 spec=importlib.util.spec_from_file_location('l20_eval_cli',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
 cli.verify(frozen)
 validate_generations(payload,job,frozen)
 atomic(out/'payload.json',payload)
 if payload['training_metrics'].get('global_step')!=job['max_steps']:raise ValueError('training incomplete')
 if not HEX.fullmatch(payload['adapter_sha256']):raise ValueError('adapter digest missing')
 expected={t['task_id'] for t in frozen['tasks']};runs={}
 for label in ('baseline','candidate'):
  rows=payload[label]
  if len(rows)!=len(expected) or {r['task_id'] for r in rows}!=expected:raise ValueError('incomplete heldout generation')
  if any(r.get('model_revision')!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0' or r.get('quantization')!='bnb-nf4' or r.get('prompts_sha256')!=job['prompts_sha256'] for r in rows):raise ValueError('generation identity mismatch')
  if any(r.get('adapter')!=(None if label=='baseline' else '/job/run/adapter') for r in rows):raise ValueError('adapter binding mismatch')
  atomic(out/(label+'-generation.json'),rows)
  result=cli.judge_all(frozen,rows,'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',out/(label+'-attempt-'+str(time.time_ns())+'.json'),'l20_qlora_'+label,{'job_id':ident,'quantization':'bnb-nf4','image_id':job['image_id'],'recipe_sha256':job['recipe_sha256']})
  attempt=Path(result['result_path']);runs[label]=json.loads(attempt.read_text())
  atomic(out/(label+'-judged.json'),runs[label]);attempt.unlink()
  if result['counts']['infrastructure'] or result['counts']['unknown']:raise ValueError('inconclusive evaluation')
 before={r['task_id']:r['classification'] for r in runs['baseline']['results']};after={r['task_id']:r['classification'] for r in runs['candidate']['results']}
 from .iteration import comparison_outcome
 from research.knowledge.evaluation import freeze as knowledge_freeze,score as knowledge_score
 knowledge=knowledge_freeze()
 if knowledge['freeze_id']!=job['knowledge_freeze_id']:raise ValueError('knowledge freeze mismatch')
 knowledge_runs={}
 for label in ('baseline','candidate'):
  rows=payload['knowledge_'+label]
  knowledge_runs[label]=knowledge_score(knowledge,[{'task_id':r['task_id'],'content':r['content'] if r.get('finish_reason')=='stop' else ''} for r in rows],job['model_revision'])
  atomic(out/('knowledge-'+label+'.json'),knowledge_runs[label])
 before.update({'knowledge:'+r['task_id']:'pass' if r['passed'] else 'fail' for r in knowledge_runs['baseline']['results']})
 after.update({'knowledge:'+r['task_id']:'pass' if r['passed'] else 'fail' for r in knowledge_runs['candidate']['results']})
 outcome=comparison_outcome(before,after)
 report={'job_id':ident,'dataset_id':job['dataset_id'],'freeze_id':job['freeze_id'],'outcome':outcome,'baseline_counts':runs['baseline']['counts'],'candidate_counts':runs['candidate']['counts'],'adapter_sha256':payload['adapter_sha256'],'quantization':'bnb-nf4','scope':'3 RTL + 10 closed-book knowledge applications; independent L20 NF4 SFT branch','knowledge_freeze_id':knowledge['freeze_id'],'knowledge_counts':{k:{n:v[n] for n in ('passed','total','result_id')} for k,v in knowledge_runs.items()},'generation_hashes':{k:digest(payload[k]) for k in ('baseline','candidate')},'comparison_complete':True}
 report['artifact_hashes']={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('baseline-generation.json','candidate-generation.json','baseline-judged.json','candidate-judged.json','knowledge-baseline.json','knowledge-candidate.json','payload.json','admission.json')}
 cid=digest(report);atomic(out/'comparison.json',report)
 call(executor,'/jobs/'+ident+'/evaluation',{'comparison_id':cid,'outcome':outcome})
 return {'status':'complete','job_id':ident,'comparison_complete':True,'comparison_id':cid,'outcome':outcome}
