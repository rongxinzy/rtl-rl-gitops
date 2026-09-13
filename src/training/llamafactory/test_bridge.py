import importlib.util,json,pathlib,sys,tempfile,unittest
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import state
spec=importlib.util.spec_from_file_location('lf_entry',HERE/'train.py');train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)

class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):return list(range(sum(len(x['content']) for x in messages)))

class BridgeTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name);self.out=self.root/'run';self.sha=state.bind(self.out,{'backend':'llamafactory','dataset_id':'test'})
 def tearDown(self):self.temp.cleanup()
 def native(self,step):
  source=self.root/f'native-{step}';source.mkdir()
  for name in ('adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth'):(source/name).write_bytes(b'fixture')
  (source/'trainer_state.json').write_text(json.dumps({'global_step':step}));return source
 def test_complete_native_resume_identity_and_integrity(self):
  cp=state.publish_native(self.out,self.native(1),1,self.sha)
  self.assertEqual(state.verify_checkpoint(cp,self.sha)['backend'],'llamafactory')
  with self.assertRaises(ValueError):state.verify_checkpoint(cp,'another-job')
  (cp/'optimizer.pt').write_bytes(b'changed')
  with self.assertRaises(ValueError):state.verify_checkpoint(cp,self.sha)
 def test_missing_native_state_never_published(self):
  source=self.native(1);(source/'rng_state.pth').unlink()
  with self.assertRaises(ValueError):state.publish_native(self.out,source,1,self.sha)
  self.assertFalse((self.out/'latest').exists())
 def test_step_mismatch_and_extra_files(self):
  with self.assertRaises(ValueError):state.publish_native(self.out,self.native(2),1,self.sha)
  cp=state.publish_native(self.out,self.native(3),3,self.sha);(cp/'unexpected').write_text('extra')
  with self.assertRaises(ValueError):state.verify_checkpoint(cp,self.sha)
 def test_retention_and_pointer(self):
  for i in (1,2,3):state.publish_native(self.out,self.native(i),i,self.sha)
  self.assertEqual((self.out/'latest').read_text(),'checkpoint-000003');self.assertFalse((self.out/'checkpoint-000001').exists())
 def test_immutable_job(self):
  with self.assertRaises(ValueError):state.bind(self.out,{'backend':'legacy'})
 def test_completed_resume_only_recreates_artifacts(self):
  cp=state.publish_native(self.out,self.native(2),2,self.sha)
  self.assertTrue(train.finalize_completed_resume(self.out,cp,self.sha,2))
  self.assertEqual(json.loads((self.out/'status.json').read_text())['status'],'complete')
  for name in ('adapter_config.json','adapter_model.safetensors'):
   self.assertEqual(state.digest(self.out/'adapter'/name),state.digest(cp/name))
  for name in ('trainer_state_final.json','training_metrics.json'):
   result=json.loads((self.out/name).read_text());self.assertEqual(result['global_step'],2);self.assertEqual(result['job_sha256'],self.sha)
  self.assertTrue(train.finalize_completed_resume(self.out,cp,self.sha,2))
  self.assertEqual(len(list(self.out.glob('checkpoint-*'))),1)
 def test_incomplete_resume_not_finalized(self):
  cp=state.publish_native(self.out,self.native(1),1,self.sha)
  self.assertFalse(train.finalize_completed_resume(self.out,cp,self.sha,2));self.assertFalse((self.out/'status.json').exists())
 def test_completed_resume_corruption_rejected(self):
  cp=state.publish_native(self.out,self.native(2),2,self.sha);(cp/'optimizer.pt').write_bytes(b'corrupt')
  with self.assertRaises(ValueError):train.finalize_completed_resume(self.out,cp,self.sha,2)
  self.assertFalse((self.out/'status.json').exists())
 def test_recover_rename_before_pointer_crash(self):
  from unittest.mock import patch
  state.publish_native(self.out,self.native(1),1,self.sha)
  with patch.object(state,'write_latest',side_effect=OSError('simulated crash')):
   with self.assertRaises(OSError):state.publish_native(self.out,self.native(2),2,self.sha)
  self.assertEqual((self.out/'latest').read_text(),'checkpoint-000001')
  cp=state.recover_latest(self.out,self.sha)
  self.assertEqual(cp.name,'checkpoint-000002');self.assertEqual((self.out/'latest').read_text(),cp.name)
  self.assertEqual(state.verify_checkpoint(cp,self.sha)['step'],2)
 def test_recover_first_checkpoint_without_pointer(self):
  from unittest.mock import patch
  with patch.object(state,'write_latest',side_effect=OSError('simulated crash')):
   with self.assertRaises(OSError):state.publish_native(self.out,self.native(1),1,self.sha)
  self.assertFalse((self.out/'latest').exists());self.assertEqual(state.recover_latest(self.out,self.sha).name,'checkpoint-000001')
 def test_recover_rejects_corrupt_or_foreign_orphan(self):
  state.publish_native(self.out,self.native(1),1,self.sha)
  cp=state.publish_native(self.out,self.native(2),2,self.sha)
  (self.out/'latest').write_text('checkpoint-000001')
  with self.assertRaises(ValueError):state.recover_latest(self.out,'other-job')
  (cp/'optimizer.pt').write_bytes(b'corrupt')
  with self.assertRaises(ValueError):state.recover_latest(self.out,self.sha)
  self.assertEqual((self.out/'latest').read_text(),'checkpoint-000001')
 def test_recover_rejects_step_directory_mismatch(self):
  cp=state.publish_native(self.out,self.native(1),1,self.sha);cp.rename(self.out/'checkpoint-000002')
  with self.assertRaises(ValueError):state.recover_latest(self.out,self.sha)
 def test_metrics_recovery_discards_uncommitted_step(self):
  p=self.out/'metrics.jsonl';p.write_text('{"step":1,"loss":1}\n{"step":2,"loss":2}\n{"step":')
  train.reconcile_metrics(self.out,1)
  self.assertEqual([json.loads(x)['step'] for x in p.read_text().splitlines()],[1])
 def test_metrics_duplicate_committed_step_rejected(self):
  (self.out/'metrics.jsonl').write_text('{"step":1}\n{"step":1}\n')
  with self.assertRaises(ValueError):train.reconcile_metrics(self.out,1)
 def test_lf_overlength_rejected(self):
  class Template:
   efficient_eos=False
   def encode_multiturn(self,*args):return [(list(range(1024)),[1])]
  p=self.root/'data.jsonl';p.write_text('\n'.join(json.dumps(r) for r in self.records()))
  with self.assertRaises(ValueError):train.prepare_rows(Tokenizer(),p,1024,Template())
 def records(self):return [{'task_id':str(i),'family_id':str(i%2),'split':'train','validation_level':'Q2','messages':[{'role':'user','content':'RTL?'},{'role':'assistant','content':'module foo; endmodule'}]} for i in range(5)]
 def prepare(self,rows):
  p=self.root/'data.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows));return train.prepare_rows(Tokenizer(),p,1024)
 def test_sharegpt_no_metadata_leak(self):
  rows,report=self.prepare(self.records());train.write_dataset(self.out/'dataset',rows)
  self.assertEqual(set(rows[0]),{'messages'});self.assertEqual(report['examples'],5)
  info=json.loads((self.out/'dataset/dataset_info.json').read_text());self.assertEqual(info['rtl_train']['formatting'],'sharegpt')
 def test_eval_and_unvalidated_and_multimodal_rejected(self):
  for key,value in [('split','eval'),('validation_level','Q3')]:
   rows=self.records();rows[0][key]=value
   with self.assertRaises(ValueError):self.prepare(rows)
  rows=self.records();rows[0]['messages'][0]['images']=['test']
  with self.assertRaises(ValueError):self.prepare(rows)
 def test_role_and_duplicate_rejected(self):
  rows=self.records();rows[0]['messages'][0]['role']='assistant'
  with self.assertRaises(ValueError):self.prepare(rows)
  rows=self.records();rows[1]['task_id']=rows[0]['task_id']
  with self.assertRaises(ValueError):self.prepare(rows)
if __name__=='__main__':unittest.main()
