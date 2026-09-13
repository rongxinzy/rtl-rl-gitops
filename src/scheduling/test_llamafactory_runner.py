import hashlib,json,pathlib,tempfile,unittest,sys
from unittest.mock import patch
import llamafactory_runner as lf
class RunnerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name)
  for name in ('state','recipe','model','runs'): (self.root/name).mkdir()
  for name in lf.FILES:(self.root/'recipe'/name).write_text('# pinned fixture\n')
  (self.root/'train.jsonl').write_text('{}\n')
  self.cfg=dict(backend='llamafactory',job_id='brain-lf-new',max_steps=20,max_length=1024,lora_rank=8,model_revision=lf.REVISION,llamafactory_commit='100e9a42c6c09f8f7849b70d60f3da445fb2024b',image_id='sha256:'+'a'*64,data='train.jsonl',model_path='model',recipe_path='recipe',data_sha256=lf.sha(self.root/'train.jsonl'),dataset_id='reviewed',evaluation_freeze_id='b'*64,source_baseline_evaluation_id='c'*64,recipe_sha256={n:lf.sha(self.root/'recipe'/n) for n in lf.FILES})
 def tearDown(self):self.temp.cleanup()
 def test_pins_and_path_rejection(self):
  lf.validate(self.root,self.cfg)
  for key,val in [('image_id','image:latest'),('data','../escape'),('max_steps',21),('backend','grpo'),('model_revision','wrong')]:
   cfg=dict(self.cfg);cfg[key]=val
   with self.assertRaises(ValueError):lf.validate(self.root,cfg)
 def test_tampered_recipe(self):
  (self.root/'recipe/train.py').write_text('changed')
  with self.assertRaises(ValueError):lf.validate(self.root,self.cfg)
 def test_container_scope(self):
  data,model,recipe=lf.validate(self.root,self.cfg);out=self.root/'runs'/self.cfg['job_id'];out.mkdir()
  cmd=lf.command(self.root,self.cfg,out,data,model,recipe)
  self.assertEqual(cmd[cmd.index('--gpus')+1],'device=0');self.assertEqual(cmd[cmd.index('--network')+1],'none')
  self.assertIn(self.cfg['image_id'],cmd);self.assertNotIn('--env-file',cmd);self.assertNotIn('--privileged',cmd)
  self.assertIn('/job/train-run',cmd);self.assertNotIn('--resume-from',cmd)
  (out/'train-run').mkdir();(out/'train-run/latest').write_text('checkpoint-000001')
  self.assertEqual(lf.command(self.root,self.cfg,out,data,model,recipe)[-2:],['--resume-from','latest'])
 def test_hold_and_authority_and_deadline(self):
  with patch.object(lf.time,'time',return_value=1000):
   self.assertIsNone(lf.stop_reason(self.root,4000))
   self.assertEqual(lf.stop_reason(self.root,1600),'deadline_save_margin')
   (self.root/'state/operator-enabled').touch()
   self.assertEqual(lf.stop_reason(self.root,4000),'authority_unavailable')
   lf.atomic(self.root/'state/operator-authority.json',dict(mode='training',valid_until=2000,deadline=4000))
   self.assertIsNone(lf.stop_reason(self.root,4000))
   (self.root/'state/scheduler-hold').touch()
   self.assertEqual(lf.stop_reason(self.root,4000),'scheduler_hold')
 def test_hold_prevents_any_docker_or_baseline(self):
  (self.root/'state/scheduler-hold').touch()
  with patch.object(lf.subprocess,'check_output') as docker:
   with self.assertRaises(RuntimeError):lf.run(self.root,self.cfg,10**12)
   docker.assert_not_called()
 def test_runner_failure_writes_sanitized_exit(self):
  out=self.root/'runs'/self.cfg['job_id'];(out/'train-run').mkdir(parents=True);(out/'train-run/job.json').write_text('{}')
  with patch.object(lf,'_run',side_effect=ValueError('private details never emitted')):
   with self.assertRaises(ValueError):lf.run(self.root,self.cfg,10**12)
  result=json.loads((out/'last-exit.json').read_text());self.assertEqual(result['error_type'],'ValueError');self.assertEqual(result['returncode'],1);self.assertNotIn('private',json.dumps(result))
 def test_runner_sigterm_handler_and_restoration(self):
  previous=lf.signal.getsignal(lf.signal.SIGTERM)
  def interrupted(*args):
   handler=lf.signal.getsignal(lf.signal.SIGTERM)
   self.assertNotEqual(handler,previous)
   handler(lf.signal.SIGTERM,None)
  with patch.object(lf,'_run',side_effect=interrupted):
   with self.assertRaises(InterruptedError):lf.run(self.root,self.cfg,10**12)
  self.assertEqual(lf.signal.getsignal(lf.signal.SIGTERM),previous)
  with patch.object(lf,'_run',return_value=0):self.assertEqual(lf.run(self.root,self.cfg,10**12),0)
  self.assertEqual(lf.signal.getsignal(lf.signal.SIGTERM),previous)
 def test_baseline_interrupt_runs_owned_child_finally(self):
  ready=self.root/'ready';cleaned=self.root/'cleaned';out=self.root/'runs/test';out.mkdir()
  script="import pathlib,time\ntry:\n pathlib.Path(%r).touch()\n while True: time.sleep(0.1)\nfinally:\n pathlib.Path(%r).write_text('owned cleanup')\n"%(str(ready),str(cleaned))
  with patch.object(lf,'stop_reason',side_effect=lambda *args:'authority_expired' if ready.exists() else None):
   with self.assertRaisesRegex(RuntimeError,'baseline interrupted'):lf.baseline_process(self.root,out,[sys.executable,'-c',script],10**12)
  self.assertEqual(cleaned.read_text(),'owned cleanup')
  request=json.loads((out/'baseline-stop-request.json').read_text());self.assertEqual(request['reason'],'authority_expired')
 def test_baseline_stop_prevents_spawn(self):
  with patch.object(lf,'stop_reason',return_value='deadline_save_margin'),patch.object(lf.subprocess,'Popen') as spawn:
   with self.assertRaises(RuntimeError):lf.baseline_process(self.root,self.root,['unused'],100)
   spawn.assert_not_called()
 def test_baseline_success_json(self):
  out=self.root/'runs/test';out.mkdir()
  with patch.object(lf,'stop_reason',return_value=None):
   result=lf.baseline_process(self.root,out,[sys.executable,'-c','print(\'{"status":"complete"}\')'],10**12)
  self.assertEqual(result['status'],'complete');self.assertFalse((out/'baseline-stop-request.json').exists())
 def test_cached_baseline_runtime_and_hash_validation(self):
  base=self.root/'research/evaluation/artifacts';(base/'frozen').mkdir(parents=True);(base/'runs/base').mkdir(parents=True)
  frozen={'tasks':[{'task_id':'a'}],'judge_runner_sha256':'d'*64}
  lf.atomic(base/'frozen'/(self.cfg['evaluation_freeze_id']+'.json'),frozen)
  report=dict(kind='qwen_base_baseline',model_revision=lf.REVISION,freeze_id=self.cfg['evaluation_freeze_id'],metadata={'training_image_id':self.cfg['image_id']},results=[{'task_id':'a'}],judge_runner_sha256='d'*64,counts={'infrastructure':0,'unknown':0})
  path=base/'runs/base/result.json';lf.atomic(path,report)
  out=self.root/'runs/test';out.mkdir();summary=dict(status='complete',result_path=str(path),evaluation_id=lf.sha(path),freeze_id=self.cfg['evaluation_freeze_id']);lf.atomic(out/'runtime-baseline.json',summary)
  self.assertEqual(lf.baseline(self.root,self.cfg,out,10**12),summary['evaluation_id'])
  report['metadata']['training_image_id']='sha256:'+'f'*64;lf.atomic(path,report);summary['evaluation_id']=lf.sha(path);lf.atomic(out/'runtime-baseline.json',summary)
  with self.assertRaises(ValueError):lf.baseline(self.root,self.cfg,out,10**12)
 def test_cached_baseline_missing_task_rejected(self):
  base=self.root/'research/evaluation/artifacts';(base/'frozen').mkdir(parents=True);(base/'runs/base').mkdir(parents=True)
  lf.atomic(base/'frozen'/(self.cfg['evaluation_freeze_id']+'.json'),{'tasks':[{'task_id':'a'}],'judge_runner_sha256':'d'*64})
  path=base/'runs/base/result.json';lf.atomic(path,{'results':[]})
  out=self.root/'runs/test';out.mkdir();lf.atomic(out/'runtime-baseline.json',dict(status='complete',result_path=str(path),evaluation_id=lf.sha(path),freeze_id=self.cfg['evaluation_freeze_id']))
  with self.assertRaises(ValueError):lf.baseline(self.root,self.cfg,out,10**12)
if __name__=='__main__':unittest.main()
