"""Admission binds administrator-approved profiles without changing legacy jobs."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from . import common as c
from . import test_worker

class ProfileTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'jobs').mkdir()
  self.source=self.root/'recipe';self.source.mkdir();(self.source/'train.py').write_text('# approved recipe\n')
  self.cfg={'image_id':'sha256:'+'a'*64,'recipe_path':str(self.source),'model_revision':c.REVISION,'orchestration_mode':'tekton'}
  self.entry={'image_id':'sha256:'+'b'*64,'recipe_path':str(self.source),'recipe_sha256':{'train.py':c.sha((self.source/'train.py').read_bytes())}}
  self.cfg['training_profiles']={'lf-v1':self.entry}
  self.body=test_worker.Tests().body()
  self.patches=[patch.object(c,'ROOT',self.root),patch.object(c,'config',return_value=self.cfg),patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3))]
  for p in self.patches:p.start()
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
  self.tmp.cleanup()
 def job(self,result):return json.loads((self.root/'jobs'/result['job_id']/'job.json').read_text())
 def test_default_preserves_exact_legacy_identity(self):
  result=c.admit(self.body);meta={k:v for k,v in self.body.items() if k not in ('data','prompts','knowledge_prompts')}
  meta.update(image_id=self.cfg['image_id'],model_revision=c.REVISION,orchestrator='tekton',recipe_sha256=self.entry['recipe_sha256'])
  self.assertEqual(self.job(result),meta);self.assertEqual(result['job_id'],'l20-'+c.sha(c.canonical(meta))[:24])
  self.assertNotIn('profile_id',meta)
 def test_selected_profile_snapshot_idempotent_and_old_job_unchanged(self):
  old=c.admit(self.body);old_meta=self.job(old);c.atomic(self.root/'jobs'/old['job_id']/'state.json',{'phase':'complete'})
  body={**self.body,'profile_id':'lf-v1'};new=c.admit(body)
  self.assertNotEqual(old['job_id'],new['job_id']);self.assertEqual(self.job(new)['image_id'],self.entry['image_id']);self.assertEqual(self.job(new)['profile_id'],'lf-v1')
  self.assertEqual(c.admit(body)['status'],'already_admitted');self.assertEqual(self.job(old),old_meta)
  self.assertEqual((self.root/'jobs'/new['job_id']/'recipe/train.py').read_bytes(),(self.source/'train.py').read_bytes())
 def test_unknown_profile_and_arbitrary_execution_fields_rejected(self):
  for extra in ({'profile_id':'unknown'},{'profile_id':'../escape'},{'profile_id':None},{'image_id':self.entry['image_id']},{'recipe_path':'/tmp/other'}):
   with self.subTest(extra=extra),self.assertRaises(ValueError):c.admit({**self.body,**extra})
  self.assertFalse(list((self.root/'jobs').iterdir()))
 def test_hash_drift_missing_hash_and_extra_recipe_rejected(self):
  for change in ('drift','missing','extra'):
   with self.subTest(change=change):
    original=dict(self.entry['recipe_sha256'])
    if change=='drift':self.entry['recipe_sha256']['train.py']='c'*64
    elif change=='missing':self.entry.pop('recipe_sha256')
    else:(self.source/'extra.py').write_text('# unexpected')
    with self.assertRaises(ValueError):c.admit({**self.body,'profile_id':'lf-v1'})
    self.entry['recipe_sha256']=original;(self.source/'extra.py').unlink(missing_ok=True)
 def test_symlink_recipe_and_mutable_image_tag_rejected(self):
  self.entry['image_id']='mutable:latest'
  with self.assertRaises(ValueError):c.admit({**self.body,'profile_id':'lf-v1'})
  self.entry['image_id']='sha256:'+'b'*64
  (self.source/'train.py').unlink();(self.root/'outside.py').write_text('# approved recipe\n');(self.source/'train.py').symlink_to(self.root/'outside.py')
  with self.assertRaises(ValueError):c.admit({**self.body,'profile_id':'lf-v1'})
 def test_profile_never_bypasses_single_active_job_gate(self):
  c.admit(self.body)
  with self.assertRaises(ValueError):c.admit({**self.body,'profile_id':'lf-v1'})

if __name__=='__main__':unittest.main()
