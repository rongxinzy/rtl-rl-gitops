import tempfile,unittest,json,io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from . import rotation as r,common as c,manager as m,server as s
GPU_FREE=r.gpu_free
PREPARED=r.prepared
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'enabled').touch()
  self.patches=[patch.object(r,'ROOT',self.root),patch.object(r,'config',return_value={'rotation_enabled':True}),patch.object(r,'service_state',return_value='inactive'),patch.object(r,'gpu_free',return_value=True),patch.object(r,'prepared',return_value=True),patch.object(r,'healthy',return_value=True),patch.object(r,'command',return_value=SimpleNamespace(returncode=0)),patch.object(r.time,'time',return_value=1000)]
  for p in self.patches:p.start()
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
  self.tmp.cleanup()
 def test_disabled_and_typed(self):
  with patch.object(r,'config',return_value={}):
   with self.assertRaises(ValueError):r.request({'role':'inference'})
  for value in [{},{'role':'shell'},{'role':'inference','cmd':'x'},[]]:
   with self.assertRaises((ValueError,TypeError)):r.request(value)
 def test_active_container_drains_without_stop(self):
  r.request({'role':'inference'});r.reconcile(lambda:{'State':{'Running':True}})
  self.assertEqual(r.status()['phase'],'pausing');r.command.assert_not_called();self.assertTrue((self.root/'rotation-pause').exists())
 def test_ready_requires_current_health(self):
  r.request({'role':'inference'});r.reconcile(lambda:None)
  r.command.assert_called_once_with(['systemctl','start','--no-block',r.SERVICE])
  with patch.object(r,'service_state',return_value='active'):r.reconcile(lambda:None)
  self.assertTrue(r.status()['ready'])
  with patch.object(r,'healthy',return_value=False):self.assertFalse(r.status()['ready'])
 def test_unknown_gpu_blocks_and_never_kills(self):
  r.request({'role':'inference'})
  with patch.object(r,'gpu_free',return_value=False):r.reconcile(lambda:None)
  self.assertEqual(r.status()['phase'],'blocked');r.command.assert_not_called()
 def test_cooldown_and_attempt_budget_survive_requests(self):
  r.request({'role':'inference'})
  for now in (1000,1001,1300,1600,1900):
   with patch.object(r.time,'time',return_value=now):r.request({'role':'inference'});r.reconcile(lambda:None)
  self.assertEqual(r.command.call_count,3);self.assertEqual(r.status()['phase'],'error')
 def test_failed_start_is_bounded(self):
  r.command.return_value.returncode=1;r.request({'role':'inference'});r.reconcile(lambda:None)
  self.assertEqual(r.status()['phase'],'error');r.reconcile(lambda:None);self.assertEqual(r.command.call_count,1)
 def test_resume_stops_only_fixed_service_and_preserves_user_pause(self):
  (self.root/'pause').touch();r.request({'role':'inference'});r.request({'role':'training'})
  with patch.object(r,'service_state',return_value='active'):r.reconcile(lambda:None)
  r.command.assert_called_once_with(['systemctl','stop','--no-block',r.SERVICE]);self.assertTrue((self.root/'rotation-pause').exists())
  with patch.object(r,'gpu_free',return_value=False):r.reconcile(lambda:None)
  self.assertTrue((self.root/'rotation-pause').exists());r.reconcile(lambda:None)
  self.assertFalse((self.root/'rotation-pause').exists());self.assertTrue((self.root/'pause').exists());self.assertFalse(r.status()['training_allowed'])
 def test_authenticated_api(self):
  handler=object.__new__(s.Handler);handler.path='/rotation/status';handler.reply=Mock();handler.auth=Mock(return_value=False)
  handler.get_locked();handler.reply.assert_called_with(401,{'error':'unauthorized'})
  handler.path='/rotation';handler.do_POST();handler.reply.assert_called_with(401,{'error':'unauthorized'})
  handler.auth.return_value=True;handler.get_locked();handler.reply.assert_called_with(404,{'error':'not_found'})
  handler.path='/rotation/status';handler.get_locked();self.assertEqual(handler.reply.call_args.args[0],200)
 def test_manager_never_starts_when_rotation_paused(self):
  folder=self.root/'jobs/l20-test';folder.mkdir(parents=True);c.atomic(folder/'job.json',{})
  (self.root/'rotation-pause').touch()
  with patch.object(m,'ROOT',self.root),patch.object(m,'config',return_value={}),patch.object(m,'command') as command:
   self.assertFalse(m.start(folder,'baseline'));command.assert_not_called()
 def test_gpu_inspection_fails_closed(self):
  for output in ('42\n','N/A\n'):
   r.command.return_value=SimpleNamespace(returncode=0,stdout=output)
   self.assertFalse(GPU_FREE())
  r.command.side_effect=[SimpleNamespace(returncode=0,stdout=''),SimpleNamespace(returncode=0,stdout='0\n0\n')]
  self.assertTrue(GPU_FREE())
 def test_manager_writes_checkpoint_pause_request(self):
  folder=self.root/'jobs/l20-test';folder.mkdir(parents=True);c.atomic(folder/'job.json',{});c.atomic(folder/'state.json',{'phase':'queued_training'})
  (self.root/'rotation-pause').touch()
  with patch.object(m,'ROOT',self.root),patch.object(m,'config',return_value={}),patch.object(m,'inspect',return_value=None),patch.object(m,'start') as start:
   m.step();start.assert_not_called()
  self.assertTrue((folder/'pause.request').exists())

 def test_preflight_manifest_and_build_hash(self):
  import hashlib
  root=self.root/'backup';rev='621d456e93e926e4b52f85cff5f634358c1828f9';engine='d94f44e79aa219d8057e8de21f95360a187ebf41'
  (root/'models'/rev).mkdir(parents=True);(root/'readiness').mkdir();(root/'llama.cpp/build/bin').mkdir(parents=True)
  files=[]
  for i in range(1,4):
   name=f'GLM-5.3-Flash-UD-IQ1_S-{i:05d}-of-00003.gguf';(root/'models'/rev/name).write_bytes(b'x');files.append({'path':'UD-IQ1_S/'+name,'size':1,'sha256':'a'*64})
  c.atomic(root/'model-manifest.json',{'revision':rev,'files':files})
  c.atomic(root/'models'/rev/'verification.json',{'revision':rev,'files':files,'sha256_verified':True})
  c.atomic(root/'readiness/READY.json',{'model_revision':rev,'engine_revision':engine,'quantization':'UD-IQ1_S','protocol_checks':['chat_text','responses_json','responses_stream','responses_tool_stream','responses_tool_roundtrip']})
  binary=root/'llama.cpp/build/bin/llama-server';binary.write_bytes(b'binary');(root/'llama.cpp/PINNED_REVISION').write_text(engine)
  c.atomic(root/'BUILD_READY.json',{'source_revision':engine,'files':{'llama.cpp/build/bin/llama-server':hashlib.sha256(b'binary').hexdigest()}})
  with patch.object(r,'BACKUP',root):
   self.assertTrue(PREPARED());binary.write_bytes(b'changed');self.assertFalse(PREPARED())
 def test_unprepared_never_starts(self):
  r.request({'role':'inference'})
  with patch.object(r,'prepared',return_value=False):r.reconcile(lambda:None)
  r.command.assert_not_called();self.assertEqual(r.status()['phase'],'blocked')
