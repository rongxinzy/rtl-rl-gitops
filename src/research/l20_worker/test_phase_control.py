import json,subprocess,tempfile,unittest,shutil
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from . import common as c,manager as m,phase_control as p
UID='11111111-2222-3333-4444-555555555555'
OTHER='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.folder=self.root/'jobs'/('l20-'+'a'*24);self.folder.mkdir(parents=True)
  c.atomic(self.folder/'job.json',{'orchestrator':'tekton','image_id':'sha256:'+'b'*64});c.atomic(self.folder/'state.json',{'phase':'queued','attempts':0});(self.root/'enabled').touch()
  for name in ['model-binding.json','baseline.jsonl','knowledge_baseline.jsonl','candidate.jsonl','knowledge_candidate.jsonl','prompts.jsonl','knowledge_prompts.jsonl','run/job.json','run/status.json','run/trainer_state_final.json','run/training_metrics.json','run/adapter/adapter_config.json','run/adapter/adapter_model.safetensors','run/checkpoint-000020/complete.json','run/checkpoint-000020/training-state.pt']:
   path=self.folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
  (self.folder/'run/latest').write_text('checkpoint-000020')
  self.patches=[patch.object(m,'ROOT',self.root),patch.object(m,'config',return_value={})]
  for item in self.patches:item.start()
 def tearDown(self):
  for item in self.patches:item.stop()
  self.tmp.cleanup()
 def grant(self,phase,uid=UID):return p.authorize(self.folder,{'phase':phase,'run_uid':uid})
 def read(self):return json.loads((self.folder/'state.json').read_text())
 def finish(self,phase):
  with patch.object(p,'verify_phase',return_value={'status':'complete','step':20}):state=p.after(self.folder,phase,{'status':'complete','step':20})
  c.atomic(self.folder/'state.json',{'phase':state,'attempts':0})
 def test_mode_snapshots_only_new_jobs_and_preserves_legacy_identity(self):
  from .test_worker import Tests as ExistingTests
  body=ExistingTests.body(self);shutil.rmtree(self.folder)
  recipe=self.root/'recipe';recipe.mkdir();(recipe/'train.py').write_text('pass')
  cfg={'model_revision':c.REVISION,'image_id':'sha256:'+'b'*64,'recipe_path':str(recipe)}
  with patch.object(c,'ROOT',self.root),patch.object(c,'config',return_value=cfg),patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
   first=c.admit(body);old=self.root/'jobs'/first['job_id'];raw=(old/'job.json').read_bytes()
   self.assertNotIn('orchestrator',json.loads(raw));c.atomic(old/'state.json',{'phase':'complete'})
   cfg['orchestration_mode']='tekton';second=c.admit(body);new=self.root/'jobs'/second['job_id']
   self.assertNotEqual(first['job_id'],second['job_id']);self.assertEqual(json.loads((new/'job.json').read_text())['orchestrator'],'tekton')
   self.assertEqual((old/'job.json').read_bytes(),raw)
 def test_no_start_without_authorization(self):
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start') as start:m.step()
  start.assert_not_called()
 def test_fixed_fields_uid_and_no_skip_or_takeover(self):
  for body in [{'phase':'baseline','run_uid':UID,'ttl':9999},{'phase':'../x','run_uid':UID},{'phase':'baseline','run_uid':'arbitrary'}]:
   with self.assertRaises(ValueError):p.authorize(self.folder,body)
  with self.assertRaises(ValueError):self.grant('training')
  self.grant('baseline')
  with self.assertRaises(ValueError):self.grant('baseline',OTHER)
 def test_sequence_verified_markers_and_same_run_retries(self):
  self.grant('baseline');self.finish('baseline')
  with patch.object(p,'verify_phase',return_value={'status':'complete','step':20}):
   self.grant('baseline');self.grant('training')
  self.finish('training')
  with patch.object(p,'verify_phase',return_value={'status':'complete','step':20}):self.grant('candidate')
  self.finish('candidate')
  with patch.object(p,'verify_phase',return_value={'status':'complete','step':20}):result=self.grant('candidate')
  self.assertTrue(all(result['verified_complete'].values()));self.assertEqual(self.read()['phase'],'awaiting_evaluation')
 def test_marker_alone_never_means_success(self):
  self.grant('baseline');self.finish('baseline')
  (self.folder/'baseline.jsonl').write_text('changed after verification')
  with patch.object(p,'verify_phase',side_effect=ValueError('bad')):
   self.assertFalse(p.status(self.folder)['verified_complete']['baseline'])
   with self.assertRaises(ValueError):self.grant('training')
 def test_paused_training_is_not_completed(self):
  with patch.object(p,'verify_phase',return_value={'status':'paused','step':3}):
   with self.assertRaises(ValueError):p.verified(self.folder,'training')
 def test_lease_renewal_is_phase_scoped_and_fixed(self):
  with patch.object(p.time,'time',return_value=100):self.grant('baseline')
  self.assertEqual(p.control(self.folder)['leases']['baseline'],280)
  with patch.object(p.time,'time',return_value=281):self.assertFalse(p.permitted(self.folder,'baseline'))
  with patch.object(p.time,'time',return_value=300):self.grant('baseline')
  self.assertEqual(p.control(self.folder)['leases']['baseline'],480)
  self.finish('baseline')
  with patch.object(p,'verify_phase',return_value={}),patch.object(p.time,'time',return_value=310):self.grant('training')
  with patch.object(p,'verify_phase',return_value={}),patch.object(p.time,'time',return_value=400):self.grant('baseline')
  self.assertEqual(p.control(self.folder)['leases']['training'],490)
 def test_expired_training_pauses_without_kill_and_resumes_after_renewal(self):
  with patch.object(p.time,'time',return_value=100):self.grant('baseline')
  self.finish('baseline')
  with patch.object(p,'verify_phase',return_value={}),patch.object(p.time,'time',return_value=100):self.grant('training')
  c.atomic(self.folder/'state.json',{'phase':'training'})
  container={'Id':'owned','Image':'sha256:'+'b'*64,'Config':{'Labels':{'rtl.l20.job':self.folder.name,'rtl.l20.phase':'training','rtl.l20.run_uid':UID}},'HostConfig':{'NetworkMode':'none','DeviceRequests':[{'DeviceIDs':['0']}]},'State':{'Running':True}}
  with patch.object(m,'inspect',return_value=container),patch.object(m,'command') as command,patch.object(p.time,'time',return_value=281):m.step()
  command.assert_not_called();self.assertTrue((self.folder/'pause.request').exists())
  c.atomic(self.folder/'state.json',{'phase':'queued_training'})
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start') as start,patch.object(p.time,'time',return_value=281):m.step()
  start.assert_not_called()
  with patch.object(p,'verify_phase',return_value={}),patch.object(p.time,'time',return_value=300):self.grant('training')
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start',return_value=False) as start,patch.object(p.time,'time',return_value=301):m.step()
  start.assert_called_once();self.assertFalse((self.folder/'pause.request').exists())
 def test_verified_removed_container_stops_before_next_stage(self):
  self.grant('baseline');c.atomic(self.folder/'state.json',{'phase':'baseline'})
  with patch.object(m,'inspect',return_value=None),patch.object(m,'verify_phase',return_value=[]),patch.object(p,'verify_phase',return_value=[]),patch.object(m,'start') as start:m.step()
  start.assert_not_called();self.assertEqual(self.read()['phase'],'awaiting_training_authorization')
 def test_durable_grant_recovers_queue_after_crash(self):
  self.grant('baseline');self.finish('baseline')
  with patch.object(p,'verify_phase',return_value={}):self.grant('training')
  state={'phase':'awaiting_training_authorization'};c.atomic(self.folder/'state.json',state)
  self.assertEqual(p.recover_grant(self.folder,state)['phase'],'queued_training')
 def test_expired_baseline_may_finish_but_not_start_training(self):
  with patch.object(p.time,'time',return_value=100):self.grant('baseline')
  container={'Id':'owned','Image':'sha256:'+'b'*64,'Config':{'Labels':{'rtl.l20.job':self.folder.name,'rtl.l20.phase':'baseline','rtl.l20.run_uid':UID}},'HostConfig':{'NetworkMode':'none','DeviceRequests':[{'DeviceIDs':['0']}]},'State':{'Running':False,'ExitCode':0}}
  result=subprocess.CompletedProcess([],0,'','')
  with patch.object(p.time,'time',return_value=999),patch.object(m,'inspect',return_value=container),patch.object(m,'command',return_value=result),patch.object(m,'verify_phase',return_value=[]),patch.object(p,'verify_phase',return_value=[]),patch.object(m,'start') as start:m.step()
  start.assert_not_called();self.assertEqual(self.read()['phase'],'awaiting_training_authorization')
 def test_renewal_never_resets_failed_retry_budget(self):
  self.grant('baseline');c.atomic(self.folder/'state.json',{'phase':'failed','attempts':3})
  self.grant('baseline');self.assertEqual(self.read()['attempts'],3);self.assertEqual(self.read()['phase'],'failed')
 def test_container_with_wrong_run_uid_is_not_adopted(self):
  self.grant('baseline')
  container={'Id':'owned','Image':'sha256:'+'b'*64,'Config':{'Labels':{'rtl.l20.job':self.folder.name,'rtl.l20.phase':'baseline','rtl.l20.run_uid':OTHER}},'HostConfig':{'NetworkMode':'none','DeviceRequests':[{'DeviceIDs':['0']}]},'State':{'Running':True}}
  with patch.object(m,'inspect',return_value=container),self.assertRaises(RuntimeError):m.step()
 def test_status_uses_receipts_not_expensive_reverification(self):
  self.grant('baseline');self.finish('baseline')
  with patch.object(p,'verify_phase',side_effect=AssertionError('must not reread weights')):
   self.assertTrue(p.status(self.folder)['verified_complete']['baseline'])
 def test_legacy_job_cannot_be_converted_by_phase_request(self):
  c.atomic(self.folder/'job.json',{'image_id':'sha256:'+'b'*64})
  with self.assertRaises(ValueError):self.grant('baseline')
  self.assertTrue(p.permitted(self.folder,'training'))
 def test_docker_outage_never_means_container_absent(self):
  self.grant('baseline');p.abort(self.folder,{'run_uid':UID})
  with patch.object(m,'command',return_value=subprocess.CompletedProcess([],1,'','Cannot connect to Docker daemon')),self.assertRaises(RuntimeError):m.step()
  self.assertNotEqual(self.read()['phase'],'failed')
  with patch.object(m,'command',return_value=subprocess.CompletedProcess([],1,'','Error: No such object: rtl-l20-training')):self.assertIsNone(m.inspect())
 def test_actual_lowercase_docker_not_found(self):
  with patch.object(m,'command',return_value=subprocess.CompletedProcess([],1,'','error: no such object: rtl-l20-training')):self.assertIsNone(m.inspect())
 def test_abort_queued_binds_run_and_finishes_without_start(self):
  result=p.abort(self.folder,{'run_uid':UID})
  self.assertEqual(result['run_uid'],UID);self.assertTrue(result['cancel_requested'])
  self.assertNotEqual(result['phase'],'failed');self.assertTrue((self.folder/'pause.request').exists())
  with self.assertRaises(ValueError):self.grant('baseline')
  with patch.object(m,'inspect',return_value=None),patch.object(m,'start') as start:m.step()
  start.assert_not_called();self.assertEqual(self.read()['phase'],'failed');self.assertEqual(self.read()['reason'],'tekton_cancelled')
 def test_abort_running_preserves_container_until_exit(self):
  for phase in p.PHASES:
   with self.subTest(phase=phase):
    (self.folder/'phase-control.json').unlink(missing_ok=True)
    c.atomic(self.folder/'state.json',{'phase':'queued'})
    for prior in p.PHASES[:p.PHASES.index(phase)+1]:
     with patch.object(p,'verify_phase',return_value={'status':'complete'}):self.grant(prior)
     if prior!=phase:self.finish(prior)
    c.atomic(self.folder/'state.json',{'phase':phase})
    container={'Id':'owned','Image':'sha256:'+'b'*64,'Config':{'Labels':{'rtl.l20.job':self.folder.name,'rtl.l20.phase':phase,'rtl.l20.run_uid':UID}},'HostConfig':{'NetworkMode':'none','DeviceRequests':[{'DeviceIDs':['0']}]},'State':{'Running':True}}
    result=p.abort(self.folder,{'run_uid':UID});self.assertEqual(result['phase'],phase)
    self.assertFalse(result['phases'][phase]['lease_active'])
    with patch.object(m,'inspect',return_value=container),patch.object(m,'command') as command:m.step()
    command.assert_not_called();self.assertEqual(self.read()['phase'],phase)
    container['State']={'Running':False,'ExitCode':0}
    with patch.object(m,'inspect',return_value=container),patch.object(m,'command',return_value=subprocess.CompletedProcess([],0,'','')),patch.object(m,'start') as start:m.step()
    start.assert_not_called();self.assertEqual(self.read()['phase'],'failed')
    self.assertTrue((self.folder/'run/checkpoint-000020/training-state.pt').exists())
 def test_abort_rejects_takeover_and_extra_fields(self):
  self.grant('baseline')
  for body in ({'run_uid':OTHER},{'run_uid':UID,'force':True},{'run_uid':'bad'}):
   with self.assertRaises(ValueError):p.abort(self.folder,body)
  self.assertFalse(p.cancelled(self.folder))
 def test_abort_complete_does_not_mutate_evidence(self):
  self.grant('baseline');c.atomic(self.folder/'state.json',{'phase':'complete'})
  paths=[self.folder/n for n in ('job.json','state.json','phase-control.json')]
  before=[path.read_bytes() for path in paths]
  self.assertEqual(p.abort(self.folder,{'run_uid':UID})['phase'],'complete')
  self.assertEqual(before,[path.read_bytes() for path in paths]);self.assertFalse((self.folder/'pause.request').exists())
 def test_cancelled_comparison_cannot_release_as_success(self):
  self.grant('baseline');c.atomic(self.folder/'state.json',{'phase':'awaiting_evaluation'})
  p.abort(self.folder,{'run_uid':UID})
  with self.assertRaises(ValueError):c.compare_complete(self.folder,{'comparison_id':'a'*64,'outcome':'improved'})
 def test_global_status_exposes_bound_run(self):
  job=json.loads((self.folder/'job.json').read_text());job.update(dataset_id='d',freeze_id='f',max_steps=20);c.atomic(self.folder/'job.json',job)
  self.grant('baseline')
  with patch.object(c,'ROOT',self.root),patch.object(c,'config',return_value={}),patch.object(m.rotation,'ROOT',self.root),patch.object(m.rotation,'config',return_value={}):result=c.status()
  self.assertEqual(result['jobs'][0]['run_uid'],UID)
if __name__=='__main__':unittest.main()

