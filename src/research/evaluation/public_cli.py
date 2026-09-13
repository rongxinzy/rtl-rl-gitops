#!/usr/bin/env python3
"""Incremental VerilogEval public sample; does not alter internal freeze or executor."""
import argparse,fcntl,json,os,pathlib,re,subprocess,sys,tempfile,time,uuid
from core import canonical,sha,publish,build_index,digest_text,MODEL_REVISION
import cli
IMAGE='sha256:fc7918cb652529351e0d0faf0d72e81932da0594ff3d319ec736237a4f266a0e'
REV='c498220d0a52248f8e3fdffe279075215bde2da6'
SCOPE='VerilogEval v2 spec-to-RTL deterministic first 5 lexicographic tasks; custom strict sandbox; not full official benchmark'

def base(root):return pathlib.Path(root)/'research/evaluation/artifacts/public'
def freeze(root):
 old=json.loads((pathlib.Path(root)/'research/evaluation/artifacts/latest.json').read_text())
 source=next(x for x in old['public_sources'] if x['revision']==REV);source_root=pathlib.Path(source['path']);repo=source_root/'NVlabs-verilog-eval-c498220';dataset=repo/'dataset_spec-to-rtl'
 files={x['relative_path']:x['sha256'] for x in source['files']}
 def read(path):
  content=path.read_bytes()
  if path.is_symlink() or sha(content)!=files[str(path.relative_to(source_root))]:raise ValueError('source differs from previous immutable inventory')
  return content.decode()
 training=[x['path'] for x in old['training_sources']];index,training_sources=build_index(training);tasks=[]
 prompts=sorted(dataset.glob('*_prompt.txt'))
 for prompt in prompts[:5]:
  task_id=prompt.name.removesuffix('_prompt.txt');ref=read(dataset/(task_id+'_ref.sv'));tb=read(dataset/(task_id+'_test.sv'));spec=read(prompt)
  if any(digest_text(x) in index['texts'] for x in [spec,ref]) or task_id in index['id']:raise ValueError('public sample training overlap')
  tasks.append({'task_id':task_id,'spec':spec,'reference':ref,'testbench':tb,'spec_sha256':sha(spec),'reference_sha256':sha(ref),'testbench_sha256':sha(tb),'top':'TopModule','image_id':IMAGE})
 bundle={'schema_version':1,'root':str(pathlib.Path(root).resolve()),'scope':SCOPE,'source_revision':REV,'source_path':str(repo),'available_tasks':len(prompts),'tasks':tasks,'license':'MIT','license_sha256':sha(read(repo/'LICENSE')),'upstream_makefile_sha256':sha(read(repo/'Makefile.in')),'training_sources':training_sources,'semantic_leakage_checked':False,'judge_runner_sha256':old['judge_runner_sha256'],'adapter_sha256':sha(pathlib.Path(__file__).read_bytes()),'worker_sha256':sha(pathlib.Path(__file__).with_name('public_worker.py').read_bytes())}
 bundle['freeze_id']=sha(canonical(bundle));path=publish(base(root)/'frozen'/(bundle['freeze_id']+'.json'),bundle)
 pointer=base(root)/'latest.tmp';pointer.write_text(json.dumps(bundle));pointer.replace(base(root)/'latest.json');return {'status':'frozen','freeze_file':path,'freeze_id':bundle['freeze_id'],'tasks':len(tasks),'available_tasks':len(prompts),'scope':SCOPE}

def load(root):
 b=json.loads((base(root)/'latest.json').read_text());body={k:v for k,v in b.items() if k!='freeze_id'}
 if sha(canonical(body))!=b['freeze_id'] or sha(pathlib.Path(__file__).read_bytes())!=b['adapter_sha256'] or sha(pathlib.Path(__file__).with_name('public_worker.py').read_bytes())!=b['worker_sha256']:raise ValueError('public freeze/adapter version mismatch')
 return b

def judge(b,task,dut):
 sys.path.insert(0,b['root']);from judge.runner import policy_error,docker_command
 error=policy_error(dut,'TopModule')
 if error:return {'classification':'fail','policy_error':error}
 work=base(b['root'])/'work';work.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=work) as directory:
  folder=pathlib.Path(directory);folder.chmod(0o755)
  for name,content in [('dut.sv',dut),('ref.sv',task['reference']),('test.sv',task['testbench'])]:
   p=folder/name;p.write_text(content);p.chmod(0o444)
  name='rtl-public-'+uuid.uuid4().hex;worker=pathlib.Path(__file__).with_name('public_worker.py')
  cmd=docker_command(IMAGE,name,str(folder));cmd[-1:-1]=['--entrypoint','python3','--mount',f'type=bind,src={worker},dst=/public.py,readonly'];cmd.append('/public.py')
  try:
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=55)
   if r.returncode:return {'classification':'infrastructure','error':r.stderr[-4000:]}
   return json.loads(r.stdout)
  except (OSError,ValueError,subprocess.TimeoutExpired) as exc:return {'classification':'infrastructure','error':type(exc).__name__}
  finally:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)

def run_judges(b,generations,revision,output,mode,metadata=None):
 tasks={t['task_id']:t for t in b['tasks']};rows=[]
 if len(generations)!=len(tasks) or {x['task_id'] for x in generations}!=set(tasks):raise ValueError('incomplete public generation')
 for item in generations:
  t=tasks[item['task_id']];dut=item['content'];fence=re.search(r'```(?:verilog|systemverilog|sv)?\s*\n(.*?)```',dut,re.S)
  if fence:dut=fence[1].strip()
  report={'classification':'unknown','reason':'generation_truncated'} if item.get('finish_reason')=='length' else judge(b,t,dut)
  rows.append({'task_id':t['task_id'],'classification':report['classification'],'report':report,'candidate_sha256':sha(dut),**{k:t[k] for k in ['spec_sha256','reference_sha256','testbench_sha256']}})
 result={'schema_version':1,'kind':'public_sample_'+mode,'freeze_id':b['freeze_id'],'model_revision':revision,'source_revision':REV,'scope':SCOPE,'metadata':metadata or {},'results':rows,'counts':{k:sum(r['classification']==k for r in rows) for k in ['pass','fail','infrastructure','unknown']}}
 path=publish(base(b['root'])/'runs'/('result-'+str(time.time_ns())+'.json'),result)
 return {'status':'complete','result_path':path,'evaluation_id':sha(pathlib.Path(path).read_bytes()),'counts':result['counts'],'scope':SCOPE}

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','reference-check','baseline-local']);p.add_argument('--root',default='/root/rtl-rl');a=p.parse_args()
 try:
  if a.action=='freeze':result=freeze(a.root)
  else:
   b=load(a.root)
   if a.action=='reference-check':
    rows=[{'task_id':t['task_id'],'content':re.sub(r'\bRefModule\b','TopModule',t['reference'])} for t in b['tasks']]
    result=run_judges(b,rows,'frozen-reference',None,'reference')
    negative=judge(b,b['tasks'][0],"module TopModule(output zero); assign zero = 1'b1; endmodule")
    result['negative_control']=negative
    publish(base(a.root)/'controls'/('negative-'+str(time.time_ns())+'.json'),negative)
    if result['counts']['pass']!=5 or negative['classification']!='fail':raise ValueError('reference/negative-control validation failed')
   else:
    with open(pathlib.Path(a.root)/'research/executor/state/executor.lock','a') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     cli.judge_all=run_judges;result=cli.local_baseline(a,b)
  print(json.dumps(result))
 except Exception as exc:print(json.dumps({'status':'blocked' if isinstance(exc,(ValueError,BlockingIOError)) else 'infrastructure_error','error':str(exc)}));raise SystemExit(2)
if __name__=='__main__':main()
