import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research.executor.core import Executor,digest
from research.executor import l20_branch
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.ex=Executor(self.root,self.root/'state')
 def tearDown(self):self.tmp.cleanup()
 def test_status_without_credential_is_unavailable(self):
  self.assertFalse(l20_branch.status(self.ex)['available'])
 def test_no_redirect_for_private_worker_key(self):
  self.assertIsNone(l20_branch.NoRedirect().redirect_request(None,None,302,'',{},'http://other'))
 def test_admission_has_bounded_budget_and_no_model_parameter(self):
  body={'action_id':'test','action':'l20.admit','params':{'dataset_id':'a'*64,'evaluation_id':'b'*64,'max_steps':20}}
  self.ex.validate(body)
  for extra in [{'max_steps':21},{'model':'huihui/test'},{'url':'http://other'}]:
   with self.subTest(extra=extra),self.assertRaises(ValueError):self.ex.validate({**body,'params':{**body['params'],**extra}})
 def test_training_profile_id_allowed_but_not_execution_parameters(self):
  body={'action_id':'test','action':'l20.admit','params':{'dataset_id':'a'*64,'evaluation_id':'b'*64,'profile_id':'lf-v1'}}
  self.ex.validate(body)
  for extra in ({'profile_id':'../recipe'},{'profile_id':None},{'image_id':'sha256:'+'c'*64},{'recipe_path':'/tmp/recipe'}):
   with self.subTest(extra=extra),self.assertRaises(ValueError):self.ex.validate({**body,'params':{**body['params'],**extra}})
 def test_profile_reaches_worker_through_existing_data_gates(self):
  from types import SimpleNamespace
  dataset=self.root/'dataset';dataset.mkdir();(dataset/'sft_train.jsonl').write_text('{}\n')
  verified=self.ex.state/'verified';verified.mkdir(parents=True,exist_ok=True)
  (verified/('evaluation-'+'b'*64+'.json')).write_text(json.dumps({'baseline_complete':True,'freeze_id':'f'*64}))
  frozen=self.root/'research/evaluation/artifacts/frozen';frozen.mkdir(parents=True)
  (frozen/('f'*64+'.json')).write_text(json.dumps({'tasks':[]}))
  for profile in (None,'lf-v1'):
   calls=[]
   def call(ex,path,body=None):
    calls.append((path,body))
    return {'current_job':None} if path=='/status' else {'job_id':'l20-'+'a'*24}
   params={'dataset_id':'a'*64,'evaluation_id':'b'*64}
   if profile:params['profile_id']=profile
   with patch('research.executor.evidence.refresh'),patch.object(l20_branch,'dataset',return_value=(dataset,{'train':8})),patch('research.knowledge.evaluation.reject_training_overlap'),patch.object(l20_branch.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='{"status":"clean"}')) as contamination,patch.object(l20_branch,'call',side_effect=call):
    l20_branch.admit(self.ex,params)
   contamination.assert_called_once();payload=calls[-1][1]
   self.assertEqual(payload.get('profile_id'),profile)
   if profile is None:self.assertNotIn('profile_id',payload)
 def test_active_worker_cannot_be_compared(self):
  with patch.object(l20_branch,'call',return_value={'current_job':{'job_id':'l20-'+'a'*24,'phase':'training'}}):
   self.assertEqual(l20_branch.compare(self.ex)['status'],'deferred')
 def test_completed_worker_claim_needs_local_comparison(self):
  c={'current_job':{'job_id':'l20-'+'a'*24,'phase':'complete','comparison':{'comparison_id':'b'*64,'outcome':'improved'}}}
  with patch.object(l20_branch,'call',return_value=c),self.assertRaises(FileNotFoundError):l20_branch.compare(self.ex)
 def test_comparison_corruption_cannot_become_verified(self):
  ident='l20-'+'a'*24;folder=self.root/'research/l20_artifacts'/ident;folder.mkdir(parents=True)
  report={'job_id':ident,'outcome':'unchanged'};cid=digest(report);report['outcome']='improved';(folder/'comparison.json').write_text(json.dumps(report))
  c={'current_job':{'job_id':ident,'phase':'complete','comparison':{'comparison_id':cid,'outcome':'unchanged'}}}
  with patch.object(l20_branch,'call',return_value=c),self.assertRaises(ValueError):l20_branch.compare(self.ex)
 def test_matching_but_incomplete_report_is_not_evidence(self):
  ident='l20-'+'a'*24;folder=self.root/'research/l20_artifacts'/ident;folder.mkdir(parents=True)
  report={'job_id':ident,'outcome':'unchanged'};cid=digest(report);(folder/'comparison.json').write_text(json.dumps(report))
  c={'current_job':{'job_id':ident,'phase':'complete','comparison':{'comparison_id':cid,'outcome':'unchanged'}}}
  with patch.object(l20_branch,'call',return_value=c),self.assertRaises(ValueError):l20_branch.compare(self.ex)
 def test_tekton_comparison_requires_paired_identity(self):
  base={'action_id':'test','action':'l20.compare','params':{}}
  for params in ({'job_id':'l20-'+'a'*24},{'run_uid':'11111111-1111-1111-1111-111111111111'},{'job_id':'l20-'+'a'*24,'run_uid':'bad'}):
   with self.assertRaises(ValueError):self.ex.validate({**base,'params':params})
 def test_explicit_comparison_observes_bound_job(self):
  job='l20-'+'a'*24;uid='11111111-1111-1111-1111-111111111111'
  with patch.object(l20_branch,'call',return_value={'job_id':job,'run_uid':uid,'orchestrator':'tekton','phase':'training'}) as call:
   self.assertEqual(l20_branch.compare(self.ex,job,uid)['status'],'deferred')
   self.assertEqual(call.call_args.args[1],'/jobs/'+job+'/status')
 def test_legacy_compare_cannot_advance_tekton(self):
  with patch.object(l20_branch,'call',return_value={'current_job':{'orchestrator':'tekton'}}),self.assertRaises(ValueError):l20_branch.compare(self.ex)
 def fixture(self):
  from research.knowledge.evaluation import freeze,prompts
  import hashlib
  knowledge=freeze();frozen={'tasks':[{'task_id':str(i),'spec':'s'} for i in range(3)]}
  job={'knowledge_freeze_id':knowledge['freeze_id'],'model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','recipe_sha256':{'generate_eval.py':'a'*64}}
  payload={'adapter_sha256':'b'*64,'model_manifest_sha256':'c'*64,'job_sha256':'d'*64}
  for prefix,questions in (('',frozen['tasks']),('knowledge_',prompts(knowledge))):
   h=hashlib.sha256(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in questions).encode()).hexdigest();job[prefix+'prompts_sha256']=h
   for label in ('baseline','candidate'):
    name=prefix+label
    rows=[{'task_id':q['task_id'],'content':'{}','finish_reason':'stop','model_revision':job['model_revision'],'quantization':'bnb-nf4','prompts_sha256':h,'adapter':None if label=='baseline' else '/job/run/adapter','adapter_sha256':None if label=='baseline' else payload['adapter_sha256'],'model_manifest_sha256':payload['model_manifest_sha256'],'job_sha256':payload['job_sha256'],'generation_recipe_sha256':'a'*64} for q in questions]
    payload[name]=rows;self.serialize(payload,name)
  return payload,job,frozen
 def serialize(self,payload,name):
  import hashlib
  raw=''.join(json.dumps(r)+'\n' for r in payload[name]);payload[name+'_raw']=raw;payload[name+'_sha256']=hashlib.sha256(raw.encode()).hexdigest()
 def test_all_generations_bound_before_scoring(self):
  payload,job,frozen=self.fixture();l20_branch.validate_generations(payload,job,frozen)
  for key,value in [('model_revision','e'*40),('adapter_sha256','e'*64),('prompts_sha256','e'*64),('job_sha256','e'*64),('generation_recipe_sha256','e'*64)]:
   with self.subTest(key=key):
    changed=json.loads(json.dumps(payload));changed['knowledge_candidate'][0][key]=value;self.serialize(changed,'knowledge_candidate')
    with self.assertRaises(ValueError):l20_branch.validate_generations(changed,job,frozen)
 def test_missing_duplicate_and_wrong_file_hash(self):
  for mutation in ('missing','duplicate','hash'):
   payload,job,frozen=self.fixture()
   if mutation=='missing':payload['knowledge_baseline'].pop();self.serialize(payload,'knowledge_baseline')
   if mutation=='duplicate':payload['knowledge_baseline'][1]=payload['knowledge_baseline'][0];self.serialize(payload,'knowledge_baseline')
   if mutation=='hash':payload['knowledge_baseline_sha256']='f'*64
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):l20_branch.validate_generations(payload,job,frozen)
 def test_truncated_correct_answer_is_not_a_pass(self):
  from research.knowledge.evaluation import freeze,score
  knowledge=freeze();rows=[{'task_id':t['task_id'],'content':json.dumps(t['answer']),'finish_reason':'stop'} for t in knowledge['tasks']]
  rows[0]['finish_reason']='length'
  result=score(knowledge,[{'task_id':r['task_id'],'content':r['content'] if r['finish_reason']=='stop' else ''} for r in rows],'a'*40)
  self.assertEqual(result['passed'],9)
 def test_verified_cache_reuse_and_evidence_tampering(self):
  import hashlib
  from research.knowledge.evaluation import freeze,score
  payload,job,frozen=self.fixture();job['dataset_id']='e'*64;payload['job']=job
  folder=self.root/'cache';folder.mkdir()
  def save(name,value):(folder/name).write_text(json.dumps(value))
  save('payload.json',payload);save('admission.json',{'dataset_id':job['dataset_id']})
  for label in ('baseline','candidate'):
   save(label+'-generation.json',payload[label]);save(label+'-judged.json',{'counts':{'infrastructure':0,'unknown':0},'results':[{'task_id':str(i),'classification':'pass'} for i in range(3)]})
   save('knowledge-'+label+'.json',score(freeze(),[{'task_id':r['task_id'],'content':r['content']} for r in payload['knowledge_'+label]],job['model_revision']))
  report={'comparison_complete':True,'outcome':'unchanged','dataset_id':job['dataset_id'],'adapter_sha256':payload['adapter_sha256'],'artifact_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
  current={'comparison':{'outcome':'unchanged'}}
  l20_branch.verify_cached(folder,report,current)
  (folder/'knowledge-candidate.json').write_text('{}')
  with self.assertRaises(ValueError):l20_branch.verify_cached(folder,report,current)
if __name__=='__main__':unittest.main()


