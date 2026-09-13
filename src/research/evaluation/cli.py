#!/usr/bin/env python3
"""JSON-only CLI for fixed evaluation actions. Errors are structured; no arbitrary shell."""
import argparse,hashlib,json,pathlib,sys,subprocess,datetime,re,time,os,fcntl,urllib.request
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from core import freeze,verify,publish,sha,classification,MODEL_REVISION,build_index,overlaps,digest_text,public_prompts

def load(a):
 path=pathlib.Path(a.freeze_file or pathlib.Path(a.root)/'research/evaluation/artifacts/latest.json')
 bundle=json.loads(path.read_text());verify(bundle);return bundle

def judge_all(bundle,generations,revision,output,mode,metadata=None):
 sys.path.insert(0,bundle['root']);from judge.runner import run_judge
 tasks={x['task_id']:x for x in bundle['tasks']};results=[]
 for row in generations:
  task=tasks[row['task_id']];code=row['content'];fence=re.search(r'```(?:verilog|systemverilog|sv)?\s*\n(.*?)```',code,re.S)
  if fence:code=fence[1].strip()
  if row.get('finish_reason')=='length':report={'status':'generation_truncated'}
  else:report=run_judge(code,task['testbench'],top=task['top'],trusted_testbench=True,image=task['image_id'],timeout=15)
  results.append({'task_id':row['task_id'],'spec_sha256':task['spec_sha256'],'reference_sha256':task['reference_sha256'],'testbench_sha256':task['testbench_sha256'],'candidate_sha256':sha(code),'classification':classification(report),'report':report})
 counts={k:sum(x['classification']==k for x in results) for k in ['pass','fail','infrastructure','unknown']}
 result={'evaluation_cli_sha256':sha(pathlib.Path(__file__).read_bytes()),'metadata':metadata or {},'schema_version':1,'kind':mode,'freeze_id':bundle['freeze_id'],'model_revision':revision,'judge_runner_sha256':bundle['judge_runner_sha256'],'scope':'internal 3val smoke; not public benchmark ability evidence','public_benchmarks_executed':False,'results':results,'counts':counts}
 for key in ['job_id','adapter_sha256']:
  if key in (metadata or {}):result[key]=metadata[key]
 path=publish(output,result);return {'status':'complete','result_path':path,'evaluation_id':sha(pathlib.Path(path).read_bytes()),'freeze_id':bundle['freeze_id'],'counts':counts,'scope':result['scope']}

def local_baseline(a,bundle,adapter=None,metadata=None):
 deadline=time.monotonic()+(1500 if adapter else 1800)
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
 if not (now.hour>=22 or (now.hour*60+now.minute)<385):raise ValueError('GPU evaluation requires a start in the 22:00–06:25 Beijing window')
 # Fixed GPU and resource limits; callers cannot inject Docker options or model paths.
 gpu=subprocess.check_output(['/usr/bin/querygpu','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
 rows={int(x.split(',')[0]):[int(v.strip()) for v in x.split(',')[1:]] for x in gpu.splitlines()}
 if 1 not in rows or rows[1][0]<71680 or rows[1][1]>256:raise ValueError('GPU1 is not idle with >=70GiB free')
 root=pathlib.Path(a.root);model=root/'models/Qwen3.8-27B'/MODEL_REVISION
 verification=json.loads((model/'verification.json').read_text())
 if verification.get('repo')!='Qwen/Qwen3.8-27B' or verification.get('revision')!=MODEL_REVISION:raise ValueError('model revision mismatch')
 for entry in verification['files']:
  path=model/entry['file'];digest=hashlib.sha256()
  with path.open('rb') as source:
   for chunk in iter(lambda:source.read(16*1024*1024),b''):digest.update(chunk)
  if path.stat().st_size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:raise ValueError('model file verification failed: '+entry['file'])
 run_dir=root/'research/evaluation/artifacts/runs'/(('qwen-candidate-' if adapter else 'qwen-base-')+str(time.time_ns()));run_dir.mkdir(parents=True)
 inp=run_dir/'input';out=run_dir/'output';inp.mkdir();out.mkdir()
 (inp/'prompts.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in public_prompts(bundle)))
 image_id=subprocess.check_output(['docker','image','inspect','rtl-training:20260912-swanlab','--format','{{.Id}}'],text=True).strip()
 name='rtl-eval-'+str(time.time_ns());script=pathlib.Path(__file__).with_name('gpu_candidate.py' if adapter else 'gpu_generate.py')
 cmd=['docker','run','--rm','--name',name,'--network','none','--gpus','device=1','--cpus','8','--memory','20g','--shm-size','2g','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--mount',f'type=bind,src={model},dst=/model,readonly','--mount',f'type=bind,src={inp},dst=/input,readonly','--mount',f'type=bind,src={out},dst=/output','--mount',f'type=bind,src={script},dst=/eval.py,readonly',image_id,'/eval.py']
 if adapter:cmd[-2:-2]=['--mount',f'type=bind,src={adapter},dst=/adapter,readonly']
 try:
  with (run_dir/'worker.log').open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()-360))
  if process.returncode:raise RuntimeError('GPU worker infrastructure failure; see '+str(run_dir/'worker.log'))
 except subprocess.TimeoutExpired:raise RuntimeError('GPU evaluation exceeded 30 minute deadline')
 finally:subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 generations=[json.loads(s) for s in (out/'generations.jsonl').read_text().splitlines()]
 if {x['task_id'] for x in generations}!={x['task_id'] for x in bundle['tasks']}:raise RuntimeError('incomplete generation batch')
 return judge_all(bundle,generations,MODEL_REVISION,run_dir/'result.json','qwen_candidate_evaluation' if adapter else 'qwen_base_baseline',{**(metadata or {}),'model_manifest_sha256':sha((model/'verification.json').read_bytes()),'model_weights_rehashed':True,'generation_worker_sha256':sha(script.read_bytes()),'gpu_index':1,'training_image_id':image_id})

def execute(a):
 root=pathlib.Path(a.root).resolve();art=root/'research/evaluation/artifacts'
 if a.action=='freeze':
  paths=a.training or [str(root/'processed/bootstrap-v2'/n) for n in ['sft_train.jsonl','rl_train.jsonl']]+[str(root/'processed/v2/export'/n) for n in ['sft_train.jsonl','rl_train.jsonl']]
  bundle=freeze(root,paths);path=publish(art/'frozen'/(bundle['freeze_id']+'.json'),bundle)
  # latest is an atomic pointer copy; content-addressed frozen artifacts stay immutable.
  tmp=art/'latest.tmp';tmp.write_text(json.dumps(bundle));tmp.replace(art/'latest.json')
  return {'status':'frozen','freeze_id':bundle['freeze_id'],'freeze_file':path,'admitted_tasks':len(bundle['tasks']),'public_sources':[{'source':s['source'],'revision':s['revision'],'files':len(s['files']),'overlaps':len(s['overlap_paths']),'admission':s['admission']} for s in bundle['public_sources']]}
 bundle=load(a)
 if a.action=='check-training':
  if not a.training:raise ValueError('check-training requires --training paths')
  index,sources=build_index(a.training);hits=[]
  for task in bundle['tasks']:
   found=overlaps(task,index)
   if found:hits.append({'task_id':task['task_id'],'reasons':found})
  for source in bundle['public_sources']:
   for entry in source['files']:
    if entry['kind']!='inventory' and digest_text((pathlib.Path(source['path'])/entry['relative_path']).read_text()) in index['texts']:hits.append({'public_path':entry['relative_path'],'reasons':['normalized_text']})
  return {'status':'clean' if not hits else 'blocked','freeze_id':bundle['freeze_id'],'training_sources':sources,'overlap_count':len(hits),'overlaps':hits[:50],'semantic_leakage_checked':False}
 if a.action=='status':return {'status':'verified','freeze_id':bundle['freeze_id'],'admitted_tasks':len(bundle['tasks']),'public_benchmark_status':'pending_reviewed_harness_adapter'}
 if a.action=='reference-check':
  generations=[{'task_id':t['task_id'],'content':t['reference']} for t in bundle['tasks']]
  return judge_all(bundle,generations,'frozen-reference',art/'runs'/('reference-'+str(time.time_ns())+'.json'),'reference_validation')
 if a.action=='baseline-local':return local_baseline(a,bundle)
 if a.action=='candidate-local':
  from candidate import evaluate
  return evaluate(a,bundle,local_baseline)
 if not a.endpoint or not a.model or not a.model_revision:raise ValueError('baseline-endpoint requires endpoint, model, model-revision')
 if not re.fullmatch(r'[0-9a-f]{40,64}',a.model_revision):raise ValueError('model revision must be immutable content/commit hash')
 from urllib.parse import urlsplit
 u=urlsplit(a.endpoint)
 if u.scheme not in ['http','https'] or u.username or u.password or u.query:raise ValueError('invalid endpoint')
 rows=[]
 for task in bundle['tasks']:
  body={'model':a.model,'messages':[{'role':'user','content':task['spec']}],'temperature':0,'max_tokens':512,'stream':False}
  req=urllib.request.Request(a.endpoint.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as response:choice=json.load(response)['choices'][0]
  rows.append({'task_id':task['task_id'],'content':choice['message']['content'],'finish_reason':choice.get('finish_reason')})
 return judge_all(bundle,rows,a.model_revision,art/'runs'/('endpoint-'+str(time.time_ns())+'.json'),'endpoint_baseline_asserted_revision')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','status','reference-check','baseline-local','baseline-endpoint','check-training','candidate-local']);p.add_argument('--root',default='/root/rtl-rl');p.add_argument('--freeze-file');p.add_argument('--training',action='append');p.add_argument('--job-id');p.add_argument('--deadline',type=float);p.add_argument('--endpoint');p.add_argument('--model');p.add_argument('--model-revision');a=p.parse_args()
 try:
  if a.action in ['baseline-local','candidate-local'] and os.environ.get('RTL_EXECUTOR_LOCK_HELD')!='1':
   with open(pathlib.Path(a.root)/'research/executor/state/executor.lock','a') as lock:
    wait_until=time.monotonic()+min(60,max(0,a.deadline-time.time()-1800) if a.deadline else 60)
    while True:
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
     except BlockingIOError:
      if time.monotonic()>=wait_until:raise
      time.sleep(1)
    result=execute(a)
  else:result=execute(a)
  print(json.dumps(result))
 except Exception as exc:
  failure={'status':'blocked' if isinstance(exc,(ValueError,BlockingIOError)) else 'infrastructure_error','error_type':type(exc).__name__,'message':str(exc),'action':a.action,'model_revision':MODEL_REVISION,'time':time.time()}
  try:
   if a.action=='candidate-local':failure['job_id']=a.job_id or json.loads((pathlib.Path(a.root)/'scheduling/job.json').read_text())['job_id']
   latest=pathlib.Path(a.root)/'research/evaluation/artifacts/latest.json'
   if latest.exists():failure['freeze_id']=json.loads(latest.read_text()).get('freeze_id')
   path=pathlib.Path(a.root)/'research/evaluation/artifacts/failures'/(a.action+'-'+str(time.time_ns())+'.json')
   failure['failure_artifact']=str(path);publish(path,failure)
  except Exception:failure['failure_artifact_error']=True
  print(json.dumps(failure));raise SystemExit(2)
if __name__=='__main__':main()
