#!/usr/bin/env python3
"""Fixed JSON plan/run/status API for a bounded CPU-only authored data factory."""
import argparse,sys,fcntl,hashlib,json,os,pathlib,re,subprocess,tempfile,time
from .dsl import make_dsl
from .families import FAMILIES,HOLDOUT,WIDTHS,VERSION,make_task,sha,canonical
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STATE=pathlib.Path(os.environ.get('RTL_DATA_FACTORY_STATE',str(HERE/'state')))
def revision():return sha(b''.join((HERE/p).read_bytes() for p in ('families.py','dsl.py','cli.py')))
def atomic(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
def jsonl(p,rows):p.write_text(''.join(canonical(x)+'\n' for x in rows))
def read_dataset(path,ancestors=None,budget=None):
 m=json.loads((path/'manifest.json').read_text())
 if 'composition_version' in m:
  from .compose import validate_composed
  return validate_composed(path,m,ancestors or set(),budget if budget is not None else [128])
 if m['dataset_id']!=path.name or not m.get('tasks'):raise ValueError('invalid promoted manifest')
 required={f'{kind}_{split}.jsonl' for kind in ('sft','rl','registry') for split in ('train','val')}|{'evidence.jsonl'}
 if set(m['files'])!=required:raise ValueError('incomplete promoted files')
 for name,expected in m['files'].items():
  p=path/name
  if p.is_symlink() or sha(p.read_bytes())!=expected:raise ValueError('promoted file checksum mismatch')
 evidence=[json.loads(line) for line in (path/'evidence.jsonl').read_text().splitlines()]
 if len(evidence)!=len(m['tasks']) or len({e['task_id'] for e in evidence})!=len(evidence):raise ValueError('invalid evidence cardinality')
 indexed={e['task_id']:e for e in evidence}
 for t in m['tasks']:
  task=make_dsl(t['dsl']) if 'dsl' in t else make_task(t['family_id'],t['width'])
  if t['split']!=task['split'] or t['task_id']!=task['task_id']:raise ValueError('holdout contamination')
  if t['semantic_sha256']!=task['semantic_sha256']:raise ValueError('semantic binding mismatch')
  e=indexed[t['task_id']]
  if e.get('admitted') is not True or not validate(task,e['reference'],e['mutants'],m['judge_image']):raise ValueError('invalid judge evidence')
 return m

def dataset_manifests():
 return sorted(p for p in (STATE/'datasets').glob('*/manifest.json') if re.fullmatch('[0-9a-f]{64}',p.parent.name) and not p.parent.is_symlink())

def admitted_index():
 ids=set();semantics=set()
 for p in dataset_manifests():
  m=read_dataset(p.parent)
  for t in m['tasks']:ids.add(t['task_id']);semantics.add(t['semantic_sha256'])
 return ids,semantics

def select(limit):
 if not 1<=limit<=8:raise ValueError('batch limit must be 1..8')
 seen,semantics=admitted_index();tasks=[]
 for w in WIDTHS:
  for family in FAMILIES:
   t=make_task(family,w)
   if t['task_id'] in seen or t['semantic_sha256'] in semantics:continue
   tasks.append({'family':family,'width':w});semantics.add(t['semantic_sha256'])
   if len(tasks)==limit:return tasks
 return tasks

def plan(limit):
 body={'schema_version':1,'generator_revision':revision(),'purpose':'authored_finite_Q2_expansion','tasks':select(limit)}
 pid=sha(canonical(body));p=STATE/'plans'/f'{pid}.json';p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():atomic(p,body)
 return {'status':'planned' if body['tasks'] else 'exhausted','plan_id':pid,'task_count':len(body['tasks']),
 'max_judge_calls':3*len(body['tasks']),'gpu_required':False,'holdout_families':sorted(HOLDOUT)}

def propose(payload):
 task=make_dsl(payload)
 seen,semantics=admitted_index()
 if task['task_id'] in seen or task['semantic_sha256'] in semantics:raise ValueError('duplicate task or truth table')
 # Pending proposals also reserve their truth tables to avoid AST-equivalent spam.
 for path in (STATE/'plans').glob('*.json'):
  previous=json.loads(path.read_text())
  if previous.get('generator_revision')!=revision():continue
  for descriptor in previous.get('tasks',[]):
   old=make_dsl(descriptor['dsl']) if 'dsl' in descriptor else make_task(descriptor['family'],descriptor['width'])
   if old['semantic_sha256']==task['semantic_sha256']:
    return {'status':'already_planned','plan_id':path.stem,'task_count':1,'gpu_required':False}
 body={'schema_version':1,'generator_revision':revision(),'purpose':'typed_DSL_Q2_expansion','tasks':[{'dsl':task['dsl']}]}
 pid=sha(canonical(body));path=STATE/'plans'/f'{pid}.json';path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():atomic(path,body)
 return {'status':'planned','plan_id':pid,'task_count':1,'max_judge_calls':3,'gpu_required':False}

def stage_pass(result,name):return result.get(name,{}).get('status')=='pass'
def validate(task,reference,mutants,image):
 def bound(result,rtl):
  return result.get('dut_sha256')==sha(rtl) and result.get('testbench_sha256')==sha(task['testbench']) and result.get('image')==image and result.get('trusted_testbench') is True
 if not bound(reference,task['reference']) or reference.get('status')!='pass' or reference.get('quality')!='Q2':return False
 if not all(stage_pass(reference,s) for s in ('compile','simulation','synthesis')):return False
 if len(mutants)!=2:return False
 return all(bound(r,rtl) and stage_pass(r,'compile') and r.get('simulation',{}).get('status')=='fail' and r.get('status')=='not_verified'
            for r,rtl in zip(mutants,task['mutants']))

def execute(pid):
 if not re.fullmatch('[0-9a-f]{64}',pid):raise ValueError('plan_id must be SHA256 hex')
 dst=STATE/'datasets'/pid
 if dst.exists():
  read_dataset(dst)
  return {'status':'already_promoted','dataset_id':pid,'dataset_path':str(dst)}
 body=json.loads((STATE/'plans'/f'{pid}.json').read_text())
 if sha(canonical(body))!=pid or body['generator_revision']!=revision():raise ValueError('plan hash or generator revision mismatch')
 if not 1<=len(body['tasks'])<=8:raise ValueError('empty or oversized plan')
 seen,semantics=admitted_index();tasks=[]
 for d in body['tasks']:
  if set(d) not in ({'family','width'},{'dsl'}):raise ValueError('unexpected task fields')
  t=make_dsl(d['dsl']) if 'dsl' in d else make_task(d['family'],d['width'])
  if t['task_id'] in seen or t['semantic_sha256'] in semantics:raise ValueError('duplicate task or truth table; replan')
  seen.add(t['task_id']);semantics.add(t['semantic_sha256']);tasks.append(t)
 from judge.runner import run_judge
 image=subprocess.check_output(['docker','image','inspect','rtl-judge:local','--format','{{.Id}}'],text=True,timeout=15).strip()
 if not re.fullmatch('sha256:[0-9a-f]{64}',image):raise ValueError('invalid judge image id')
 evidence=[]
 for t in tasks:
  reference=run_judge(t['reference'],t['testbench'],image=image,trusted_testbench=True,timeout=10)
  mutants=[run_judge(x,t['testbench'],image=image,trusted_testbench=True,timeout=10) for x in t['mutants']]
  evidence.append({'task_id':t['task_id'],'admitted':validate(t,reference,mutants,image),'reference':reference,'mutants':mutants})
 attempt=STATE/'attempts'/f'{pid}.json';attempt.parent.mkdir(parents=True,exist_ok=True)
 atomic(attempt,{'plan_id':pid,'passed':all(e['admitted'] for e in evidence),'evidence':evidence,'time':time.time()})
 if not all(e['admitted'] for e in evidence):return {'status':'rejected','plan_id':pid,'evidence_path':str(attempt),'promoted':False}
 (STATE/'datasets').mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.stage-',dir=STATE/'datasets') as staging:
  out=pathlib.Path(staging);meta=[]
  for split in ('train','val'):
   subset=[t for t in tasks if t['split']==split]
   common=lambda t:{'task_id':t['task_id'],'family_id':t['family_id'],'width':t['width'],'split':split,'coverage':t['coverage'],
    'validation_level':'Q2','testbench_trusted':True,'source':t['family_id'] if 'dsl' in t else VERSION,'generator_revision':body['generator_revision'],'image_id':image,
    'spec_sha256':sha(t['spec']),'reference_sha256':sha(t['reference']),'testbench_sha256':sha(t['testbench']),'semantic_sha256':t['semantic_sha256'],**({'dsl':t['dsl']} if 'dsl' in t else {})}
   jsonl(out/f'sft_{split}.jsonl',[{**common(t),'messages':[{'role':'user','content':t['spec']},{'role':'assistant','content':t['reference']}]} for t in subset])
   registry=[{**common(t),'spec':t['spec'],'top':t['top'],'testbench':t['testbench']} for t in subset]
   jsonl(out/f'registry_{split}.jsonl',registry);jsonl(out/f'rl_{split}.jsonl',registry);meta.extend(common(t) for t in subset)
  jsonl(out/'evidence.jsonl',evidence)
  files={p.name:sha(p.read_bytes()) for p in out.iterdir()}
  manifest={'schema_version':1,'dataset_id':pid,'generator_revision':body['generator_revision'],'judge_revision':sha((ROOT/'judge/runner.py').read_bytes()),
   'judge_image':image,'tasks':meta,'files':files,'train':sum(t['split']=='train' for t in tasks),'val':sum(t['split']=='val' for t in tasks),
   'quality_scope':'Q2 finite exhaustive binary combinational truth tables plus 2 killed mutants; no formal proof, no unknown-value semantics or model-quality claim',
   'validated':True,'active_registry_modified':False,'promoted_unix':time.time()}
  atomic(out/'manifest.json',manifest)
  for p in out.iterdir():p.chmod(0o444)
  out.chmod(0o555);out.rename(dst)
 return {'status':'promoted','dataset_id':pid,'dataset_path':str(dst),'manifest_path':str(dst/'manifest.json'),'train':manifest['train'],'val':manifest['val'],'promoted':True}

def status():
 ps=dataset_manifests();ms=[read_dataset(p.parent) for p in ps]
 available=[{'dataset_id':m['dataset_id'],'kind':'composed' if 'composition_version' in m else ('dsl' if any('dsl' in t for t in m['tasks']) else 'authored'),'validated':True,'data':str((p.parent/'rl_train.jsonl').relative_to(ROOT)),
  'data_sha256':m['files']['rl_train.jsonl'],'registry':str((p.parent/'registry_train.jsonl').relative_to(ROOT)),
  'registry_sha256':m['files']['registry_train.jsonl'],'train':m['train'],'val':m['val']} for p,m in zip(ps,ms)]
 plans=[]
 for p in sorted((STATE/'plans').glob('*.json')):
  b=json.loads(p.read_text())
  if b.get('generator_revision')==revision() and b.get('tasks') and not (STATE/'datasets'/p.stem).exists():plans.append(p.stem)
 return {'status':'ok','available_plans':plans[:32],'available_datasets':available,'datasets':len(ms),'train':sum(m['train'] for m in ms),'val':sum(m['val'] for m in ms),
 'remaining_tasks':len(select(8)),'remaining_tasks_is_capped':True,'scope':VERSION,'gpu_required':False,'state_root':str(STATE)}

def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
 for name in ('propose','compose'):
  p=sp.add_parser(name);p.add_argument('--stdin',action='store_true',required=True)
 p=sp.add_parser('plan');p.add_argument('--limit',type=int,default=4)
 p=sp.add_parser('run');p.add_argument('--plan-id',required=True);sp.add_parser('status');args=ap.parse_args()
 try:
  STATE.mkdir(parents=True,exist_ok=True)
  with (STATE/'factory.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if args.command in ('propose','compose'):
    raw=sys.stdin.read(16385)
    if len(raw)>16384:raise ValueError('proposal too large')
    from .compose import compose
    result=(propose if args.command=='propose' else compose)(json.loads(raw))
   else:result=plan(args.limit) if args.command=='plan' else execute(args.plan_id) if args.command=='run' else status()
 except BlockingIOError:result={'status':'busy','retryable':True}
 except Exception as e:result={'status':'error','error_type':type(e).__name__,'error':str(e)}
 print(json.dumps(result,ensure_ascii=False))
 return 0 if result['status'] not in ('error','rejected') else 1
if __name__=='__main__':raise SystemExit(main())
