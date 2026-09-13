import importlib.util,json,pathlib,sys,tempfile,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import candidate
spec=importlib.util.spec_from_file_location('evaluation_cli_runtime',pathlib.Path(__file__).with_name('cli.py'))
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

class RuntimeTests(unittest.TestCase):
 def test_fixed_digest_selects_runtime_and_legacy_default_remains(self):
  image='sha256:'+'a'*64
  with patch.object(cli.subprocess,'check_output',return_value=image) as inspect:
   self.assertEqual(cli.evaluation_image(SimpleNamespace(runtime_image_id=image)),image)
   self.assertEqual(inspect.call_args.args[0][3],image)
   self.assertEqual(cli.evaluation_image(SimpleNamespace()),image)
   self.assertEqual(inspect.call_args.args[0][3],'rtl-training:20260912-swanlab')
 def test_client_cannot_override_job_image_or_use_mutable_tag(self):
  image='sha256:'+'a'*64
  for request in ('latest','sha256:'+'b'*64):
   with patch.object(cli.subprocess,'check_output') as inspect,self.assertRaises(ValueError):cli.evaluation_image(SimpleNamespace(runtime_image_id=request),{'training_image_id':image})
   inspect.assert_not_called()
 def test_inspected_image_must_equal_requested_digest(self):
  with patch.object(cli.subprocess,'check_output',return_value='sha256:'+'b'*64),self.assertRaises(ValueError):cli.evaluation_image(SimpleNamespace(runtime_image_id='sha256:'+'a'*64))
 def test_deadline_rejects_before_any_gpu_operation(self):
  with patch.object(cli.subprocess,'check_output') as query,self.assertRaises(ValueError):cli.local_baseline(SimpleNamespace(deadline=time.time()+1000),{})
  query.assert_not_called()
 def test_candidate_requires_same_runtime_baseline_before_generation(self):
  with tempfile.TemporaryDirectory() as temp:
   root=pathlib.Path(temp);(root/'scheduling').mkdir();(root/'scheduling/job.json').write_text(json.dumps({'job_id':'job'}))
   job=root/'runs/job';job.mkdir(parents=True)
   runs=root/'research/evaluation/artifacts/runs/qwen-base-old';runs.mkdir(parents=True)
   baseline={'kind':'qwen_base_baseline','freeze_id':'f'*64,'model_revision':candidate.MODEL_REVISION,'metadata':{'training_image_id':'sha256:'+'b'*64}}
   (runs/'result.json').write_text(json.dumps(baseline))
   config={'image_id':'sha256:'+'a'*64,'backend':'llamafactory'}
   identity={'job_id':'job','adapter_sha256':'a'*64}
   args=SimpleNamespace(root=str(root),job_id='job',deadline=time.time()+3600)
   with patch.object(candidate,'validate_job',return_value=(job,config,identity)),patch.object(candidate,'evaluation_pins',return_value=None),patch.object(cli,'local_baseline') as generate,self.assertRaisesRegex(ValueError,'no matching'):
    candidate.evaluate(args,{'freeze_id':'f'*64},generate)
   generate.assert_not_called()

if __name__=='__main__':unittest.main()
