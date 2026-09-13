import copy,hashlib,json,pathlib,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import pro_source,source,collect,relay

class ProTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=pathlib.Path(self.tmp.name);self.folder=self.root/'brain-lf-test';(self.folder/'train-run').mkdir(parents=True)
  self.identity={'backend':'llamafactory','model_revision':source.REVISION,'dataset_id':'a'*64,'data_sha256':'b'*64,'max_steps':2,'llamafactory_commit':'100e9a42c6c09f8f7849b70d60f3da445fb2024b','max_length':1024,'rank':8,'recipe_sha256':{n:'c'*64 for n in ('train.py','model.py','state.py','provenance.py','knowledge_data.py')}}
  self.cfg={**self.identity,'job_id':self.folder.name,'image_id':'sha256:'+'d'*64,'secret':'do-not-export','lora_rank':8,'recipe_sha256':{**self.identity['recipe_sha256'],'generate_eval.py':'e'*64}}
  self.put('job-config.json',self.cfg);self.put('train-run/job.json',self.identity)
  self.row={'step':1,'loss':0.5,'gradient_norm':1.0,'peak_allocated_bytes':123,'raw':'do-not-export'}
  (self.folder/'train-run/metrics.jsonl').write_text(json.dumps(self.row)+'\n')
 def put(self,name,value):(self.folder/name).write_text(json.dumps(value))
 def test_pro_identity_scalar_only_and_hardware_presentation(self):
  job=pro_source.collect(self.folder,{'training_gpu':{'name':'RTX 6000D'}})
  self.assertEqual(job['source'],'pro6000d');self.assertEqual(len(job['steps']),1)
  self.assertNotIn('do-not-export',json.dumps(job))
  cfg,desc,tags=relay.presentation(job,{'training_gpu':{'name':'L20'}})
  self.assertEqual(cfg['training_device_observation']['training_gpu']['name'],'RTX 6000D')
  self.assertIn('pro6000D',tags);self.assertNotIn('L20',tags)
  self.assertTrue(relay.run_id(job).startswith('pro-'))
 def test_mismatch_and_symlink_rejected(self):
  self.cfg['dataset_id']='e'*64;self.put('job-config.json',self.cfg)
  with self.assertRaises(ValueError):pro_source.collect(self.folder)
  self.cfg['dataset_id']=self.identity['dataset_id'];self.put('job-config.json',self.cfg)
  (self.folder/'train-run/metrics.jsonl').unlink();(self.folder/'train-run/metrics.jsonl').symlink_to(self.folder/'job-config.json')
  with self.assertRaises(OSError):pro_source.collect(self.folder)
 def test_no_completion_without_matching_final_files(self):
  self.assertFalse(pro_source.collect(self.folder)['training_complete'])
  (self.folder/'train-run/metrics.jsonl').write_text(''.join(json.dumps({**self.row,'step':n})+'\n' for n in (1,2)))
  binding=hashlib.sha256((self.folder/'train-run/job.json').read_bytes()).hexdigest()
  for name in ('training_metrics.json','trainer_state_final.json'):self.put('train-run/'+name,{'global_step':2,'job_sha256':binding})
  self.assertTrue(pro_source.collect(self.folder)['training_complete'])
  self.put('train-run/training_metrics.json',{'global_step':2,'job_sha256':'e'*64})
  with self.assertRaises(ValueError):pro_source.collect(self.folder)
 def test_raw_data_and_secret_paths_are_not_readable(self):
  for name in ('train-run/worker.log','data.jsonl','secrets/key','../job-config.json'):
   with self.assertRaises(ValueError):pro_source.read(self.folder,name)
 def test_l20_failure_preserves_exact_metrics_and_allows_pro(self):
  old={'job_id':'l20-'+'f'*24,'steps':[{'step':1}],'training_complete':False,'phase':'training','source_observed_at':10}
  pro=pro_source.collect(self.folder)
  result=collect.merge({'jobs':[old]},None,{'jobs':[pro]})
  preserved=result['jobs'][0]
  self.assertTrue(preserved['source_stale']);self.assertEqual(preserved['steps'],old['steps']);self.assertEqual(preserved['source_observed_at'],10)
  self.assertFalse(preserved['training_complete']);self.assertEqual(result['jobs'][1]['source'],'pro6000d')
  self.assertFalse(result['jobs'][1]['source_stale'])
 def test_collector_publishes_pro_when_ssh_fails(self):
  pro={'schema_version':1,'jobs':[pro_source.collect(self.folder)]}
  with patch.object(collect.pro_source,'snapshot',return_value=pro),patch.object(collect,'pull_l20',side_effect=OSError('must-not-export')):
   collect.main(self.root/'relay')
  result=json.loads((self.root/'relay/snapshot.json').read_text())
  self.assertEqual(len(result['jobs']),1);self.assertTrue(result['sources']['l20']['stale'])
  self.assertNotIn('must-not-export',json.dumps(result))
 def test_recipe_drift_is_rejected(self):
  self.identity['recipe_sha256']['train.py']='f'*64;self.put('train-run/job.json',self.identity)
  with self.assertRaises(ValueError):pro_source.collect(self.folder)
 def attempt(self,start=100,exit_time=None,code=0,status='paused'):
  self.put('last-start.json',{'time':start,'backend':'llamafactory'})
  if exit_time is not None:self.put('last-exit.json',{'time':exit_time,'returncode':code,'step':1,'backend':'llamafactory'})
  binding=hashlib.sha256((self.folder/'train-run/job.json').read_bytes()).hexdigest()
  self.put('train-run/status.json',{'step':1,'status':status,'job_sha256':binding})
 def test_failed_attempt_is_terminal_without_native_final_files(self):
  self.attempt(exit_time=110,code=137)
  job=pro_source.collect(self.folder)
  self.assertEqual(job['phase'],'failed');self.assertFalse(job['training_complete'])
  self.assertEqual(job['attempt_started_at'],100)
 def test_runner_failure_null_and_missing_step_are_terminal(self):
  self.attempt()
  for record in ({'time':110,'returncode':137,'step':None,'backend':'llamafactory'},
                 {'time':110,'returncode':1,'backend':'llamafactory'}):
   with self.subTest(record=record):
    self.put('last-exit.json',record)
    self.assertEqual(pro_source.collect(self.folder)['phase'],'failed')
  for invalid in ('1',True,-1,2):
   self.put('last-exit.json',{'time':110,'returncode':137,'step':invalid,'backend':'llamafactory'})
   with self.subTest(step=invalid),self.assertRaises(ValueError):pro_source.collect(self.folder)
  self.put('last-exit.json',{'time':110,'returncode':0,'step':None,'backend':'llamafactory'})
  with self.assertRaises(ValueError):pro_source.collect(self.folder)
 def test_normal_pause_and_new_start_ignore_old_exit(self):
  self.attempt(exit_time=110)
  job=pro_source.collect(self.folder);self.assertEqual(job['phase'],'paused')
  self.put('last-start.json',{'time':120,'backend':'llamafactory'})
  self.assertEqual(pro_source.collect(self.folder)['phase'],'training')
  (self.folder/'last-start.json').unlink()
  self.assertEqual(pro_source.collect(self.folder)['phase'],'queued_training')
 def test_invalid_lifecycle_timestamp_and_pause_binding_rejected(self):
  self.attempt(exit_time=float('nan'))
  with self.assertRaises(ValueError):pro_source.collect(self.folder)
  self.attempt(exit_time=110)
  self.put('train-run/status.json',{'step':1,'status':'paused','job_sha256':'wrong'})
  with self.assertRaises(ValueError):pro_source.collect(self.folder)
 def test_new_attempt_reopens_same_run_only_for_pro(self):
  self.attempt(exit_time=110,code=137);job=pro_source.collect(self.folder)
  receipt={'status':'failed','job_sha256':job['job_sha256'],'metadata_version':relay.METADATA_VERSION,'attempt_started_at':100}
  self.assertTrue(relay.receipt_terminal(job,receipt))
  self.put('last-start.json',{'time':120,'backend':'llamafactory'})
  resumed=pro_source.collect(self.folder)
  self.assertFalse(relay.receipt_terminal(resumed,receipt));self.assertEqual(relay.run_id(job),relay.run_id(resumed))
  self.assertTrue(relay.receipt_terminal({**resumed,'source':'l20'},receipt))
 def test_legacy_id_bytes_unchanged(self):
  job={'job_id':'l20-'+'a'*24,'job_sha256':'b'*64}
  expected='l20-'+hashlib.sha256((job['job_id']+':'+job['job_sha256']).encode()).hexdigest()[:24]
  self.assertEqual(relay.run_id(job),expected);self.assertEqual(relay.run_id({**job,'source':'l20'}),expected)

if __name__=='__main__':unittest.main()
