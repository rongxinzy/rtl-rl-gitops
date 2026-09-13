import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import lf_profile as lf

class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        for name in ('scheduling','state','recipe','model'):(self.root/name).mkdir()
        (self.root/'recipe/train.py').write_text('# test')
        self.profile=dict(backend='llamafactory',image_id='sha256:'+'a'*64,recipe_path='recipe',recipe_sha256={'train.py':hashlib.sha256(b'# test').hexdigest()},model_path='model',model_revision=lf.REVISION,llamafactory_commit=lf.COMMIT)
        self.write_profile()
    def write_profile(self):
        (self.root/'scheduling/llamafactory-profile.json').write_text(json.dumps(self.profile))
    def test_missing_preserves_default(self):
        (self.root/'scheduling/llamafactory-profile.json').unlink()
        self.assertIsNone(lf.load_profile(self.root))
    def test_profile(self):self.assertEqual(lf.load_profile(self.root),self.profile)
    def test_recipe_drift(self):
        (self.root/'recipe/train.py').write_text('changed')
        with self.assertRaises(ValueError):lf.load_profile(self.root)
    def test_extra_file(self):
        (self.root/'recipe/extra.py').touch()
        with self.assertRaises(ValueError):lf.load_profile(self.root)
    def test_symlink(self):
        (self.root/'alias').symlink_to(self.root/'recipe',target_is_directory=True)
        self.profile['recipe_path']='alias';self.write_profile()
        with self.assertRaises(ValueError):lf.load_profile(self.root)
    def test_escape(self):
        for path in ('../recipe','/tmp/recipe'):
            self.profile['recipe_path']=path;self.write_profile()
            with self.assertRaises(ValueError):lf.load_profile(self.root)
    def fixture_data(self):
        folder=self.root/'research/data_factory/state/datasets'/'d123'
        folder.mkdir(parents=True)
        (folder/'rl_train.jsonl').write_bytes(b'original')
        (folder/'sft_train.jsonl').write_bytes(b'sft')
        (folder/'registry_train.jsonl').write_bytes(b'registry')
        old=dict(data=str((folder/'rl_train.jsonl').relative_to(self.root)),data_sha256=hashlib.sha256(b'original').hexdigest(),dataset_id='d123',max_steps=20,baseline_evaluation_id='baseline1',evaluation_freeze_id='freeze1')
        manifest=dict(dataset_id='d123',files={'sft_train.jsonl':hashlib.sha256(b'sft').hexdigest()})
        return folder,old,manifest
    def test_sft_and_baseline_binding(self):
        folder,old,manifest=self.fixture_data();calls=[]
        def runner(args,**kwargs):
            calls.append(args);return subprocess.CompletedProcess(args,0,'{"status":"clean"}')
        with patch('research.data_factory.cli.read_dataset',return_value=manifest) as reader:
            result=lf.config(self.root,old,runner=runner)
        reader.assert_called_once_with(folder)
        self.assertTrue(result['data'].endswith('/sft_train.jsonl'))
        self.assertEqual(result['source_baseline_evaluation_id'],'baseline1')
        self.assertNotIn('baseline_evaluation_id',result)
        self.assertIn(str(folder/'sft_train.jsonl'),calls[0])
    def test_sft_drift_rejected(self):
        folder,old,manifest=self.fixture_data();(folder/'sft_train.jsonl').write_bytes(b'drift')
        with patch('research.data_factory.cli.read_dataset',return_value=manifest):
            with self.assertRaises(ValueError):lf.config(self.root,old)
    def test_original_data_drift_rejected(self):
        folder,old,_=self.fixture_data();(folder/'rl_train.jsonl').write_bytes(b'drift')
        with self.assertRaises(ValueError):lf.config(self.root,old)
    def test_sft_contamination_rejected(self):
        folder,old,manifest=self.fixture_data()
        with patch('research.data_factory.cli.read_dataset',return_value=manifest):
            with self.assertRaises(ValueError):lf.config(self.root,old,runner=lambda *a,**k:subprocess.CompletedProcess(a[0],0,'{"status":"overlap"}'))
    def setup_job(self):
        ident='brain-'+'b'*24
        (self.root/'state/scheduler-hold').touch()
        (self.root/'scheduling/job.json').write_text(json.dumps({'job_id':ident}))
        return ident
    def test_hold_required(self):
        with self.assertRaises(ValueError):lf.migrate_queued(self.root,'brain-'+'b'*24)
    def test_started_rejected(self):
        ident=self.setup_job();(self.root/'runs'/ident).mkdir(parents=True)
        with self.assertRaises(ValueError):lf.migrate_queued(self.root,ident)
    def test_cas(self):
        self.setup_job()
        with self.assertRaises(ValueError):lf.migrate_queued(self.root,'brain-'+'c'*24)
    def test_active_rejected(self):
        ident=self.setup_job()
        with self.assertRaises(ValueError):lf.migrate_queued(self.root,ident,lambda *a,**k:subprocess.CompletedProcess(a[0],0,'active'))
    def test_migration_journal(self):
        ident=self.setup_job();new={'job_id':'brain-'+'c'*24,'backend':'llamafactory'}
        (self.root/'state/scheduler.json').write_text(json.dumps({'started_window':'old','business':'keep'}))
        runner=lambda args,**k:subprocess.CompletedProcess(args,0,'inactive' if args[0]=='systemctl' else '')
        with patch.object(lf,'config',return_value=new):lf.migrate_queued(self.root,ident,runner)
        self.assertEqual(json.loads((self.root/'scheduling/job.json').read_text()),new)
        journal=json.loads((self.root/'state/superseded'/f'{ident}.json').read_text())
        self.assertEqual(journal['previous_job'],{'job_id':ident})
        self.assertEqual(json.loads((self.root/'state/scheduler.json').read_text())['business'],'keep')
        self.assertFalse((self.root/'runs').exists())

if __name__=='__main__':unittest.main()
