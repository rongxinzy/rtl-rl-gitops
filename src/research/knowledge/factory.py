"""Fixed source-grounded lessons, with real code evidence and immutable train artifacts."""
import argparse,json,os,pathlib,re,subprocess,tempfile
from .curriculum import VERSION,WIDTHS,FAMILIES,lesson,messages,sha,canonical
HERE=pathlib.Path(__file__).parent
ROOT=pathlib.Path(os.getenv('RTL_KNOWLEDGE_ROOT','/root/rtl-rl/research/knowledge'))
PINS={'lowRISC/style-guides':'9c15ff5dce23eef969e00ab3715153967419eadd','YosysHQ/yosys':'d0e71cfb7bcafe2b437f3edc1789b69e99ecd55a'}
def write(path,value):path.write_text(canonical(value)+'\n')
def sources():
 rows=json.loads((HERE/'sources/manifest.json').read_text())
 if len(rows)!=4:raise ValueError('fixed four source/license artifacts required')
 for r in rows:
  if PINS.get(r['repo'])!=r['revision'] or r['license'] not in ('CC-BY-4.0','ISC'):raise ValueError('source/revision/license mismatch')
  p=HERE/'sources'/r['local_file']
  if p.resolve().parent!=(HERE/'sources').resolve() or sha(p.read_bytes())!=r['sha256']:raise ValueError('source hash mismatch')
 return rows

def build():
 from judge.runner import run_judge
 src=sources();image=subprocess.check_output(['docker','image','inspect','rtl-judge:local','--format','{{.Id}}'],text=True).strip()
 recipe={n:sha((HERE/n).read_bytes()) for n in ('factory.py','curriculum.py')}
 identity={'version':VERSION,'sources':src,'recipe':recipe,'judge_image':image,'families':FAMILIES,'widths':WIDTHS};ident=sha(canonical(identity))
 parent=ROOT/'artifacts';parent.mkdir(parents=True,exist_ok=True);target=parent/ident
 if target.exists():read_dataset(target);return {'status':'already_built','dataset_id':ident,'validated':True}
 with tempfile.TemporaryDirectory(prefix='.building-',dir=parent) as tmp:
  folder=pathlib.Path(tmp);rows=[];lessons=[];evidence={}
  for family in FAMILIES:
   for width in WIDTHS:
    item=lesson(family,width);key=item['lesson_id']
    good=run_judge(item['reference'],item['testbench'],trusted_testbench=True,image=image)
    wrong=run_judge(item['mutant'],item['testbench'],trusted_testbench=True,image=image)
    if good.get('status')!='pass' or wrong.get('compile',{}).get('status')!='pass' or wrong.get('simulation',{}).get('status')!='fail':
     failure=ROOT/'last-build-failure.json';write(failure,{'lesson':key,'reference':good,'mutant':wrong});raise ValueError('knowledge code proof failed')
    proof={'lesson_id':key,'reference':good,'mutant':wrong,'coverage':item['coverage'],'reference_sha256':sha(item['reference']),'testbench_sha256':sha(item['testbench'])}
    evidence[key]=proof;lessons.append(item)
    for kind,prompt,answer in messages(item):
     rows.append({'task_id':key+':'+kind,'family_id':'knowledge_'+family,'split':'train','validation_level':'K1-grounded','kind':kind,
      'semantic_sha256':sha(canonical({'lesson_semantics':item['semantic_sha256'],'kind':kind})),
      'knowledge_evidence_sha256':sha(canonical(proof)),'knowledge_source_sha256':sha(canonical(src)),
      'verification_scope':'Source-reviewed authored explanation; associated reference code simulated and synthesized; prose is not formally proved.',
      'messages':[{'role':'user','content':prompt},{'role':'assistant','content':answer}]})
  (folder/'sft_train.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows));write(folder/'lessons.json',lessons);write(folder/'evidence.json',evidence);write(folder/'sources.json',src)
  files={p.name:sha(p.read_bytes()) for p in folder.iterdir() if p.is_file()}
  manifest={'dataset_id':ident,'kind':'knowledge_sft','identity':identity,'train':len(rows),'val':0,'files':files,'tasks':[{'task_id':r['task_id'],'split':'train','semantic_sha256':r['semantic_sha256']} for r in rows],
   'scope':'54 authored instructional examples from 18 code lessons; not a full industrial corpus; evaluation stored separately','validated':True}
  write(folder/'manifest.json',manifest)
  folder.rename(target)
 return {'status':'built','dataset_id':ident,'validated':True,'train':len(rows)}

def read_dataset(folder):
 folder=pathlib.Path(folder);m=json.loads((folder/'manifest.json').read_text());src=sources()
 if m['dataset_id']!=folder.name or sha(canonical(m['identity']))!=m['dataset_id'] or m['identity']['sources']!=src:raise ValueError('dataset provenance mismatch')
 for name,digest in m['files'].items():
  p=folder/name
  if p.is_symlink() or p.resolve().parent!=folder.resolve() or sha(p.read_bytes())!=digest:raise ValueError('dataset artifact integrity mismatch')
 proof=json.loads((folder/'evidence.json').read_text());lessons=json.loads((folder/'lessons.json').read_text());rows=[json.loads(x) for x in (folder/'sft_train.jsonl').read_text().splitlines()]
 if len(rows)!=m['train'] or len({r['task_id'] for r in rows})!=len(rows):raise ValueError('duplicate/count mismatch')
 by_id={x['lesson_id']:x for x in lessons}
 for row in rows:
  key=row['task_id'].rsplit(':',1)[0];item=by_id[key];p=proof[key]
  if row['split']!='train' or row['validation_level']!='K1-grounded' or row['knowledge_evidence_sha256']!=sha(canonical(p)) or row['knowledge_source_sha256']!=sha(canonical(src)):raise ValueError('unverified knowledge row')
  if p['reference']['status']!='pass' or p['mutant']['simulation']['status']!='fail' or p['reference_sha256']!=sha(item['reference']) or p['testbench_sha256']!=sha(item['testbench']):raise ValueError('invalid code evidence')
  expected=next((q,a) for kind,q,a in messages(item) if kind==row['kind'])
  if [v['content'] for v in row['messages']]!=list(expected):raise ValueError('unreviewed knowledge content')
 return m

def status():
 rows=[]
 for p in sorted((ROOT/'artifacts').glob('*/manifest.json')):
  m=read_dataset(p.parent);rows.append({k:m[k] for k in ('dataset_id','kind','train','val','validated')})
 return {'datasets':rows,'ready':bool(rows)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','status']);a=p.parse_args();print(json.dumps(build() if a.action=='build' else status()))
