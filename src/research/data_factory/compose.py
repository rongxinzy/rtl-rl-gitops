"""Immutable train-only composition with recursive source and evidence binding."""
import json,pathlib,re,tempfile,time
from .families import canonical,sha
VERSION='train-compose-v1'
HEX=re.compile('[0-9a-f]{64}')
KINDS=('sft','rl','registry')
def source_ids(payload):
 if not isinstance(payload,dict) or set(payload)!={'dataset_ids'}:raise ValueError('invalid compose fields')
 ids=payload['dataset_ids']
 if not isinstance(ids,list) or not 2<=len(ids)<=16 or not all(isinstance(x,str) and HEX.fullmatch(x) for x in ids):raise ValueError('compose requires 2..16 dataset IDs')
 ids=sorted(set(ids))
 if len(ids)<2:raise ValueError('compose requires 2 distinct sources')
 return ids

def materialize(root,ids,ancestors,budget):
 from .cli import read_dataset
 if len(ancestors)>8:raise ValueError('composition depth limit')
 sources=[];tasks=[];rows={k:[] for k in KINDS};evidence=[];seen=set();task_ids=set()
 for did in ids:
  if did in ancestors:raise ValueError('composition cycle')
  path=root/did
  if path.is_symlink():raise ValueError('symlink source')
  budget[0]-=1
  if budget[0]<0:raise ValueError('composition source traversal limit')
  manifest=read_dataset(path,ancestors,budget)
  sources.append({'dataset_id':did,'manifest_sha256':sha((path/'manifest.json').read_bytes())})
  selected=[t for t in manifest['tasks'] if t['split']=='train']
  indexed={}
  for kind in KINDS:
   content=[json.loads(line) for line in (path/f'{kind}_train.jsonl').read_text().splitlines()]
   indexed[kind]={r['task_id']:r for r in content}
   if len(indexed[kind])!=len(content) or set(indexed[kind])!={t['task_id'] for t in selected}:raise ValueError('train row cardinality mismatch')
   if any(r.get('split')!='train' for r in content):raise ValueError('val contamination')
  ev={e['task_id']:e for e in map(json.loads,(path/'evidence.jsonl').read_text().splitlines())}
  for task in selected:
   key=task['semantic_sha256']
   if key in seen or task['task_id'] in task_ids:continue
   seen.add(key);task_ids.add(task['task_id']);tasks.append(task);evidence.append(ev[task['task_id']])
   for kind in KINDS:rows[kind].append(indexed[kind][task['task_id']])
   if len(tasks)>256:raise ValueError('compose train row limit 256')
 if not tasks:raise ValueError('no train rows')
 files={'evidence.jsonl':''.join(canonical(e)+'\n' for e in evidence)}
 for kind in KINDS:
  files[f'{kind}_train.jsonl']=''.join(canonical(r)+'\n' for r in rows[kind])
  files[f'{kind}_val.jsonl']=''
 return sources,tasks,files

def descriptor(sources):return {'composition_version':VERSION,'sources':sources}

def validate_composed(path,m,ancestors,budget):
 if m.get('composition_version')!=VERSION:raise ValueError('unknown composition version')
 ids=source_ids({'dataset_ids':[s['dataset_id'] for s in m['sources']]})
 if path.name in ancestors:raise ValueError('composition cycle')
 sources,tasks,files=materialize(path.parent,ids,ancestors|{path.name},budget)
 if sources!=m['sources'] or sha(canonical(descriptor(sources)))!=path.name:raise ValueError('composition source binding mismatch')
 if m.get('tasks')!=tasks or m.get('train')!=len(tasks) or m.get('val')!=0:raise ValueError('composition metadata mismatch')
 if m.get('files')!={name:sha(body) for name,body in files.items()}:raise ValueError('composition output binding mismatch')
 for name,body in files.items():
  file=path/name
  if file.is_symlink() or file.read_text()!=body:raise ValueError('composition contents mismatch')
 return m

def compose(payload):
 from . import cli
 ids=source_ids(payload);root=cli.STATE/'datasets'
 sources,tasks,files=materialize(root,ids,set(),[128])
 body=descriptor(sources);did=sha(canonical(body));dst=root/did
 if did in ids:raise ValueError('self composition')
 if dst.exists():
  cli.read_dataset(dst)
  return {'status':'already_composed','dataset_id':did,'train':len(tasks),'val':0,'validated':True}
 with tempfile.TemporaryDirectory(prefix='.compose-',dir=root) as temporary:
  out=pathlib.Path(temporary)
  for name,content in files.items():(out/name).write_text(content)
  manifest={**body,'schema_version':1,'dataset_id':did,'tasks':tasks,'files':{name:sha(content) for name,content in files.items()},
            'train':len(tasks),'val':0,'validated':True,'active_registry_modified':False,'promoted_unix':time.time(),
            'quality_scope':'Train-only composition of source-verified Q2 rows; source judge evidence preserved, no new judge run'}
  cli.atomic(out/'manifest.json',manifest)
  for file in out.iterdir():file.chmod(0o444)
  out.chmod(0o555);out.rename(dst)
 cli.read_dataset(dst)
 return {'status':'composed','dataset_id':did,'train':len(tasks),'val':0,'validated':True,'source_count':len(sources),'gpu_required':False}
