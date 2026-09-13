import json,shutil,subprocess,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from . import common as c,manager as m,artifacts as a,server as s

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.job='l20-'+'a'*24
  self.folder=self.root/'jobs'/self.job;self.folder.mkdir(parents=True)
  (self.root/'enabled').touch()
  self.state={'phase':'queued','attempts':0}
  c.atomic(self.folder/'state.json',self.state)
  self.meta={'image_id':'sha256:'+'b'*64,'model_revision':c.REVISION,'max_steps':20,'recipe_sha256':{},'prompts_sha256':'c'*64}
  c.atomic(self.folder/'job.json',self.meta)
  self.patches=[patch.object(m,'ROOT',self.root),patch.object(c,'ROOT',self.root),patch.object(m,'config',return_value={})]
  for p in self.patches:p.start()
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
  self.tmp.cleanup()
 def read(self):return json.loads((self.folder/'state.json').read_text())
 def body(self):
  from research.knowledge.evaluation import freeze,prompts
  data=''.join(json.dumps({'task_id':'t'+str(i),'split':'train','validation_level':'Q2'})+'\n' for i in range(8))
  code=''.join(json.dumps({'task_id':'v'+str(i),'spec':'module'})+'\n' for i in range(3))
  quiz=''.join(json.dumps(x)+'\n' for x in prompts(freeze()))
  return {'dataset_id':'a'*64,'freeze_id':'b'*64,'data':data,'data_sha256':c.sha(data),'prompts':code,'prompts_sha256':c.sha(code),'max_steps':20,'knowledge_prompts':quiz,'knowledge_prompts_sha256':c.sha(quiz),'knowledge_freeze_id':freeze()['freeze_id']}
 def test_prompt_only_knowledge_admission(self):c.validate(self.body())
 def test_knowledge_answers_cannot_enter_prompt_field(self):
  b=self.body();rows=[json.loads(x) for x in b['knowledge_prompts'].splitlines()];rows[0]['answer']={'private':True}
  b['knowledge_prompts']=''.join(json.dumps(x)+'\n' for x in rows);b['knowledge_prompts_sha256']=c.sha(b['knowledge_prompts'])
  with self.assertRaises(ValueError):c.validate(b)
 def test_knowledge_tag_without_evidence_rejected(self):
  b=self.body();rows=[json.loads(x) for x in b['data'].splitlines()];rows[0]['validation_level']='K1-grounded'
  b['data']=''.join(json.dumps(x)+'\n' for x in rows);b['data_sha256']=c.sha(b['data'])
  with self.assertRaises(ValueError):c.validate(b)
 def test_shared_lock(self):self.assertIs(m.LOCK,s.LOCK);self.assertIs(m.LOCK,c.LOCK)
 def test_start_failure_stops_after_three_attempts(self):
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start',side_effect=RuntimeError('private-error')) as start:
   for stamp in (0,301,602):
    with patch.object(m.time,'time',return_value=stamp):m.step()
  self.assertEqual(self.read()['phase'],'failed');self.assertEqual(start.call_count,3)
  self.assertNotIn('private-error',json.dumps(self.read()))
 def test_waiting_resources_not_a_failure(self):
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start',return_value=False):m.step()
  self.assertEqual(self.read()['attempts'],0)
 def container(self,running=True):
  return {'Id':'owned','Image':self.meta['image_id'],'Config':{'Labels':{'rtl.l20.job':self.job,'rtl.l20.phase':'training'}},'HostConfig':{'NetworkMode':'none','DeviceRequests':[{'DeviceIDs':['0']}]},'State':{'Running':running,'ExitCode':0}}
 def test_restart_adopts_running_worker_without_relaunch(self):
  with patch.object(m,'inspect',return_value=self.container()),patch.object(m,'start') as start:m.step()
  start.assert_not_called();self.assertEqual(self.read()['phase'],'training')
 def test_pause_requests_checkpoint_without_killing_container(self):
  (self.root/'pause').touch()
  with patch.object(m,'inspect',return_value=self.container()),patch.object(m,'command') as cmd:m.step()
  self.assertTrue((self.folder/'pause.request').exists());cmd.assert_not_called()
 def test_exit_zero_without_training_artifacts_does_not_complete(self):
  result=subprocess.CompletedProcess([],0,'','')
  with patch.object(m,'inspect',return_value=self.container(False)),patch.object(m,'command',return_value=result):m.step()
  self.assertEqual(self.read()['phase'],'queued_training');self.assertEqual(self.read()['attempts'],1)
 def test_foreign_gpu_not_adopted(self):
  container=self.container();container['HostConfig']['DeviceRequests'][0]['DeviceIDs']=['1']
  with patch.object(m,'inspect',return_value=container),patch.object(m,'command') as cmd:
   with self.assertRaises(RuntimeError):m.step()
  cmd.assert_not_called()
 def test_comparison_hex_and_idempotency(self):
  c.atomic(self.folder/'state.json',{'phase':'awaiting_evaluation'})
  body={'comparison_id':'d'*64,'outcome':'unchanged'}
  with patch.object(a,'verify_all') as check:
   c.compare_complete(self.folder,body);c.compare_complete(self.folder,body)
  self.assertEqual(check.call_count,1)
  for bad in [{**body,'comparison_id':'z'*64},{**body,'comparison_id':'e'*64}]:
   with self.assertRaises(ValueError):c.compare_complete(self.folder,bad)
 def test_model_path_comes_only_from_ready_marker_and_provenance(self):
  model=self.root/'verified-model';model.mkdir();(model/'verification.json').write_text('{}');ready=self.root/'ready.json';c.atomic(ready,{'path':str(model)})
  with patch.object(a,'module') as module:
   self.assertEqual(a.model_path({'model_ready':str(ready),'model_revision':c.REVISION,'model_path':'/wrong'},self.folder),model)
  module.return_value.verify_model.assert_called_once_with(model)
  (model/'verification.json').write_text('{"changed":true}')
  with patch.object(a,'module'),self.assertRaises(ValueError):a.model_path({'model_ready':str(ready),'model_revision':c.REVISION},self.folder)
 def test_missing_ready_marker_waits(self):self.assertIsNone(a.model_path({'model_ready':str(self.root/'missing')},self.folder))
 def test_complete_checkpoint_binding_and_corruption(self):
  recipe=self.folder/'recipe';recipe.mkdir()
  source=Path(__file__).resolve().parents[2]/'training/l20/state.py'
  shutil.copyfile(source,recipe/'state.py')
  job={**self.meta,'dataset_id':'d'*64,'data_sha256':'e'*64,'recipe_sha256':{'state.py':c.sha(source.read_bytes())}}
  c.atomic(self.folder/'job.json',job)
  run=self.folder/'run';run.mkdir();checkpoint=run/'checkpoint-000020';checkpoint.mkdir();adapter=run/'adapter';adapter.mkdir()
  identity={k:job[k] for k in ('dataset_id','data_sha256','max_steps','model_revision')}
  identity.update(max_length=1024,rank=8,recipe_sha256=job['recipe_sha256'],model_manifest_sha256='f'*64)
  c.atomic(self.folder/'model-binding.json',{'model_manifest_sha256':'f'*64})
  c.atomic(run/'job.json',identity);job_sha=c.sha((run/'job.json').read_bytes())
  files={}
  for name,raw in [('adapter_config.json',b'{}'),('adapter_model.safetensors',b'weights'),('training-state.pt',b'optimizer')]:
   (checkpoint/name).write_bytes(raw);files[name]=c.sha(raw)
   if name!='training-state.pt':(adapter/name).write_bytes(raw)
  c.atomic(checkpoint/'complete.json',{'step':20,'job_sha256':job_sha,'files':files})
  (run/'latest').write_text(checkpoint.name)
  c.atomic(run/'status.json',{'status':'complete','step':20,'job_sha256':job_sha})
  for name in ('training_metrics.json','trainer_state_final.json'):c.atomic(run/name,{'global_step':20,'job_sha256':job_sha})
  self.assertEqual(a.verify_phase(self.folder,'training')['step'],20)
  (checkpoint/'training-state.pt').write_bytes(b'corrupted')
  with self.assertRaises(ValueError):a.verify_checkpoint(self.folder)
 def test_removed_container_with_verified_completion_advances(self):
  c.atomic(self.folder/'state.json',{'phase':'training','attempts':0})
  with patch.object(m,'inspect',return_value=None),patch.object(m,'verify_phase',return_value={'status':'complete','step':20}),patch.object(m,'start',return_value=False) as start:m.step()
  self.assertEqual(self.read()['phase'],'queued_candidate')
  self.assertEqual(start.call_args.args[1],'candidate')
 def test_partial_eval_artifact_rejected(self):
  (self.folder/'prompts.jsonl').write_text('\n'.join(json.dumps({'task_id':str(i),'spec':'s'}) for i in range(3)))
  (self.folder/'baseline.jsonl').write_text('{}\n')
  with self.assertRaises(ValueError):a.verify_phase(self.folder,'baseline')

 def test_recovery_requires_both_rtl_and_knowledge_bound_artifacts(self):
  job={**self.meta,'recipe_sha256':{'generate_eval.py':'a'*64},'knowledge_prompts_sha256':'x'}
  for name,count in (('prompts',3),('knowledge_prompts',10)):
   raw=''.join(json.dumps({'task_id':str(i)})+'\n' for i in range(count));(self.folder/(name+'.jsonl')).write_text(raw);job[name+'_sha256']=c.sha(raw)
  c.atomic(self.folder/'job.json',job);c.atomic(self.folder/'model-binding.json',{'model_manifest_sha256':'f'*64})
  for prefix,count in (('',3),('knowledge_',10)):
   rows=[{'task_id':str(i),'content':'{}','finish_reason':'stop','model_revision':c.REVISION,'prompts_sha256':job[prefix+'prompts_sha256'],'quantization':'bnb-nf4','adapter':None,'adapter_sha256':None,'job_sha256':c.sha((self.folder/'job.json').read_bytes()),'model_manifest_sha256':'f'*64,'generation_recipe_sha256':'a'*64} for i in range(count)]
   (self.folder/(prefix+'baseline.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
  self.assertEqual(len(a.verify_phase(self.folder,'baseline')),3)
  (self.folder/'knowledge_baseline.jsonl').unlink()
  with self.assertRaises(FileNotFoundError):a.verify_phase(self.folder,'baseline')
  c.atomic(self.folder/'state.json',{'phase':'baseline','attempts':0})
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start') as start:m.step()
  self.assertEqual(self.read()['phase'],'queued_baseline');self.assertEqual(self.read()['attempts'],1);start.assert_not_called()

if __name__=='__main__':unittest.main()
