"""Frozen evaluation identities and conservative contamination checks."""
import hashlib,json,re,pathlib,os
SCHEMA=1
SOURCES={'NVlabs__verilog-eval':'c498220d0a52248f8e3fdffe279075215bde2da6','hkust-zhiyao__RTLLM':'51ed553d0ffd32797a1a0a13e051656bf302c81f'}
MODEL_REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
def sha(data):return hashlib.sha256(data if isinstance(data,bytes) else data.encode()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest_text(text):
 text=re.sub(r'/\*.*?\*/|//[^\n]*',' ',text,flags=re.S)
 return sha(''.join(text.split()).casefold())
def read_rows(path):
 with pathlib.Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def record_fields(x):
 fields={'id':x.get('task_id',x.get('id')),'family':x.get('family_id')}
 specs=[x.get(k) for k in ['spec','prompt','question'] if isinstance(x.get(k),str)]
 codes=[x.get(k) for k in ['rtl','reference','response','completion'] if isinstance(x.get(k),str)]
 for message in x.get('messages',[]):
  if message.get('role')=='user':specs.append(message.get('content',''))
  if message.get('role')=='assistant':codes.append(message.get('content',''))
 fields['texts']={digest_text(t) for t in specs+codes if t};return fields

def build_index(paths):
 index={'id':set(),'family':set(),'texts':set()};sources=[]
 for path in paths:
  path=pathlib.Path(path).resolve(); count=0
  for item in read_rows(path):
   f=record_fields(item);count+=1
   for k in ['id','family']:
    if f[k]:index[k].add(f[k])
   index['texts'].update(f['texts'])
  sources.append({'path':str(path),'sha256':sha(path.read_bytes()),'rows':count})
 return index,sources

def overlaps(item,index):
 f=record_fields(item);hits=[]
 for k in ['id','family']:
  if f[k] and f[k] in index[k]:hits.append(k)
 if f['texts']&index['texts']:hits.append('normalized_text')
 return hits

def freeze(root,training):
 root=pathlib.Path(root).resolve();index,training_sources=build_index(training)
 authored={x['task_id']:x for x in read_rows(root/'processed/bootstrap-v2/authored_tasks.jsonl')}
 tasks=[]
 for row in read_rows(root/'processed/bootstrap-v2/registry_val.jsonl'):
  if row.get('split')!='val' or row.get('testbench_trusted') is not True:raise ValueError('untrusted/non-val internal task')
  ref=authored[row['task_id']]['reference'];task={**row,'reference':ref}
  if sha(ref)!=row['reference_sha256'] or sha(row['testbench'])!=row['testbench_sha256']:raise ValueError('reference/testbench source hash mismatch')
  hits=overlaps(task,index)
  if hits:raise ValueError('internal evaluation leakage: '+row['task_id']+' '+','.join(hits))
  task['spec_sha256']=sha(task['spec']);task['admission']='internal_smoke_only';tasks.append(task)
 if len(tasks)!=3:raise ValueError('expected exactly three frozen internal validation tasks')
 public=[]
 for name,revision in SOURCES.items():
  directory=root/'data/evaluation'/name/revision;manifest=json.loads((directory/'download_manifest.json').read_text())
  if manifest.get('revision')!=revision or manifest.get('download_verified') is not True:raise ValueError('public source revision/download mismatch')
  entries=[];conflicts=[]
  for path in sorted(directory.rglob('*')):
   if path.is_symlink():raise ValueError('source symlink rejected')
   if not path.is_file():continue
   data=path.read_bytes();kind='inventory'
   if path.name.endswith('_prompt.txt') or path.name=='design_description.txt':kind='prompt'
   if path.name.endswith('_ref.sv') or path.name.startswith('verified_'):kind='reference'
   hit=kind!='inventory' and digest_text(data.decode('utf-8')) in index['texts']
   entry={'relative_path':str(path.relative_to(directory)),'sha256':sha(data),'kind':kind,'training_overlap':hit};entries.append(entry)
   if hit:conflicts.append(entry['relative_path'])
  public.append({'source':manifest['repo'],'revision':revision,'path':str(directory),'tree_sha256':sha(canonical(entries)),'files':entries,'overlap_paths':conflicts,'admission':'pending_reviewed_harness_adapter'})
 body={'evaluation_core_sha256':sha(pathlib.Path(__file__).read_bytes()),'internal_sources':[{'path':str(root/'processed/bootstrap-v2'/name),'sha256':sha((root/'processed/bootstrap-v2'/name).read_bytes())} for name in ['registry_val.jsonl','authored_tasks.jsonl']],'schema_version':SCHEMA,'training_sources':training_sources,'public_sources':public,'tasks':tasks,'judge_runner_sha256':sha((root/'judge/runner.py').read_bytes()),'judge_worker_sha256':sha((root/'judge/worker.py').read_bytes()),'root':str(root),'limits':['Exact/normalized text and internal-family checks only; semantic leakage not excluded','Public source inventories frozen, no public benchmark execution claimed','Three bootstrap validation tasks are infrastructure smoke only']}
 body['freeze_id']=sha(canonical(body));return body

def verify(bundle):
 if bundle.get('schema_version')!=SCHEMA:raise ValueError('unsupported freeze schema')
 if bundle.get('evaluation_core_sha256')!=sha(pathlib.Path(__file__).read_bytes()):raise ValueError('evaluation core version changed')
 payload=dict(bundle);identity=payload.pop('freeze_id')
 if sha(canonical(payload))!=identity:raise ValueError('freeze content hash mismatch')
 for source in bundle['training_sources']+bundle['internal_sources']:
  if sha(pathlib.Path(source['path']).read_bytes())!=source['sha256']:raise ValueError('training source changed since freeze')
 for source in bundle['public_sources']:
  for entry in source['files']:
   if sha((pathlib.Path(source['path'])/entry['relative_path']).read_bytes())!=entry['sha256']:raise ValueError('public source drift')
 for module in ['runner','worker']:
  if sha((pathlib.Path(bundle['root'])/'judge'/f'{module}.py').read_bytes())!=bundle[f'judge_{module}_sha256']:raise ValueError('judge version changed')
 return True

def classification(report):
 status=report.get('status')
 if status=='pass' and report.get('quality')=='Q2':return 'pass'
 if status in ['infrastructure_error','infrastructure_timeout']:return 'infrastructure'
 if any(report.get(k,{}).get('status')=='timeout' for k in ['compile','simulation','synthesis']):return 'unknown'
 if any(report.get(k,{}).get('status')=='fail' for k in ['compile','simulation','synthesis']):return 'fail'
 if status in ['policy_rejected','fail','failed'] or status in ['compile_error','simulation_failed','synthesis_failed']:return 'fail'
 return 'unknown'

def publish(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=canonical(value)+'\n'
 try:
  with path.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
 except FileExistsError:
  if path.read_text()!=data:raise ValueError('refusing to overwrite immutable artifact')
 return str(path)

def public_prompts(bundle):
 return [{'task_id':t['task_id'],'spec':t['spec']} for t in bundle['tasks']]
