import tempfile,unittest,time,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import post_eval
import llamafactory_runner
class Tests(unittest.TestCase):
 def test_lf_candidate_uses_monitored_cleanup_path(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cfg={'job_id':'brain-lf','backend':'llamafactory','image_id':'sha256:'+'a'*64}
   with patch.object(llamafactory_runner,'evaluation_process',return_value={'job_id':'brain-lf','comparison_complete':True}) as owned,patch.object(post_eval.subprocess,'run') as legacy:
    self.assertTrue(post_eval.run(p,cfg,p,time.time()+3600));legacy.assert_not_called()
   self.assertEqual(owned.call_args.kwargs,{'phase':'candidate','budget':1740})
   self.assertIn('--job-id',owned.call_args.args[2]);self.assertFalse((p/'evaluation-pending.json').exists())
 def test_lf_candidate_interrupt_remains_pending(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cfg={'job_id':'brain-lf','backend':'llamafactory','image_id':'sha256:'+'a'*64}
   with patch.object(llamafactory_runner,'evaluation_process',side_effect=RuntimeError('cleanup awaited')):
    self.assertFalse(post_eval.run(p,cfg,p,time.time()+3600))
   self.assertTrue((p/'evaluation-pending.json').exists());self.assertEqual(json.loads((p/'post-eval-status.json').read_text())['status'],'deferred')
 def test_short_window_defers_without_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with patch.object(post_eval.subprocess,'run') as command:
    self.assertFalse(post_eval.run(p,{'job_id':'brain-test'},p,time.time()+60));command.assert_not_called()
   self.assertTrue((p/'evaluation-pending.json').exists())
 def test_real_complete_clears_pending(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);response={'job_id':'brain-test','comparison_complete':True,'outcome':'regressed'}
   with patch.object(post_eval.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(response))):
    self.assertTrue(post_eval.run(p,{'job_id':'brain-test'},p,time.time()+3600))
   self.assertFalse((p/'evaluation-pending.json').exists())
 def test_inconclusive_is_not_complete(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with patch.object(post_eval.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='{"comparison_complete":false}')):
    self.assertFalse(post_eval.run(p,{'job_id':'brain-test'},p,time.time()+3600))
   self.assertTrue((p/'evaluation-pending.json').exists())

 def test_explicit_legacy_pending_is_evaluated(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'evaluation-pending.json').write_text(json.dumps({'job_id':'legacy-test'}))
   response={'job_id':'legacy-test','comparison_complete':True,'outcome':'unchanged'}
   with patch.object(post_eval.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(response))):
    self.assertTrue(post_eval.run(p,{'job_id':'legacy-test'},p,time.time()+3600))
   self.assertFalse((p/'evaluation-pending.json').exists())

 def test_preserves_queued_freeze_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);request={'job_id':'legacy-test','freeze_id':'a'*64,'baseline_evaluation_id':'b'*64}
   (p/'evaluation-pending.json').write_text(json.dumps(request))
   with patch.object(post_eval.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='{}')) as command:
    self.assertFalse(post_eval.run(p,{'job_id':'legacy-test'},p,time.time()+3600))
    self.assertIn('--freeze-file',command.call_args.args[0])
   self.assertEqual(json.loads((p/'evaluation-pending.json').read_text())['baseline_evaluation_id'],'b'*64)
