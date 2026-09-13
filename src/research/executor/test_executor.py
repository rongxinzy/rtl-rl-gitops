import json
import hashlib
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
from .core import Executor


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = []
        self.executor = Executor(self.root, self.root / 'executor', lambda a, p: self.calls.append(a) or {'ok': True})
    def tearDown(self):
        self.temp.cleanup()
    def stage(self, action='inspect', params=None):
        body = {'action_id': 'test', 'action': action, 'params': params or {}, 'stage': 'stage'}
        staged = self.executor.submit(body)
        return {**body, 'stage': 'apply', 'plan_hash': staged['plan_hash']}
    def test_stage_no_execution(self):
        self.stage()
        self.assertEqual(self.calls, [])
    def test_apply_exact_once(self):
        apply = self.stage()
        self.assertEqual(self.executor.submit(apply)['state'], 'completed')
        self.executor.submit(apply)
        self.assertEqual(self.calls, ['inspect'])
    def test_id_rebinding_forbidden(self):
        self.stage()
        with self.assertRaises(ValueError):
            self.executor.submit({'action_id': 'test', 'action': 'data.status'})
    def test_hash_required(self):
        body = self.stage()
        body['plan_hash'] = 'wrong'
        with self.assertRaises(ValueError):
            self.executor.submit(body)
    def test_paths_and_shell_rejected(self):
        for body in [{'action_id': '../escape', 'action': 'inspect'}, {'action_id': 'x', 'action': 'shell'}, {'action_id': 'x', 'action': 'inspect', 'params': {'cmd': 'reboot'}}, {'action_id': 'x', 'action': 'data.run', 'params': {'plan_id': '../../escape'}}]:
            with self.assertRaises(ValueError):
                self.executor.validate(body)
    def test_fixed_aliases(self):
        body = self.executor.validate({'action_id': 'x', 'action': 'eval.baseline'})
        self.assertEqual(body['action'], 'evaluation.baseline')
    def test_training_requires_evidence(self):
        executor = Executor(self.root, self.root / 'real')
        with self.assertRaises(FileNotFoundError):
            executor.propose({'dataset_id': 'a' * 64, 'evaluation_id': 'b' * 64})
    def test_failed_result_no_sensitive_error(self):
        self.executor.runner = lambda *args: (_ for _ in ()).throw(RuntimeError('secret-token'))
        result = self.executor.submit(self.stage())
        self.assertEqual(result['state'], 'failed')
        self.assertNotIn('secret-token', json.dumps(result))
    def test_running_is_not_replayed(self):
        body = self.stage()
        path = self.executor.state / 'test.json'
        state = json.loads(path.read_text());state['state'] = 'running';path.write_text(json.dumps(state))
        self.assertEqual(self.executor.submit(body)['state'], 'running')
        self.assertFalse(self.calls)
    def test_admission_preserves_incomplete_job(self):
        from .admission import admit
        executor = Executor(self.root, self.root / 'real')
        folder = self.root / 'research/data_factory/state/datasets' / ('a' * 64)
        folder.mkdir(parents=True)
        data = folder / 'rl_train.jsonl';data.write_text(''.join(json.dumps({'task_id':'new' + str(i), 'spec':'spec'}) + '\n' for i in range(8)))
        verified = executor.state / 'verified';verified.mkdir()
        (verified / ('dataset-' + 'a' * 64 + '.json')).write_text(json.dumps({'data': str(data.relative_to(self.root)), 'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}))
        (self.root / 'state').mkdir()
        (self.root / 'scheduling').mkdir()
        job = self.root / 'scheduling/job.json';job.write_text('{"job_id":"current-100"}')
        before = job.read_bytes()
        with patch.object(executor, 'propose', return_value={}), patch('research.executor.admission.shutil.disk_usage', return_value=type('Disk', (), {'free':100 * 1024**3})()):
            with self.assertRaisesRegex(ValueError, 'not completed'):
                admit(executor, {'dataset_id':'a' * 64, 'evaluation_id':'b' * 64})
        self.assertEqual(job.read_bytes(), before)
    def test_registry_rejects_heldout(self):
        from .registry import registry_rows
        path = self.root / 'registry.jsonl';path.write_text('{"task_id":"heldout","split":"val"}\n')
        with self.assertRaises(ValueError):
            registry_rows(path)
    def test_dead_worker_is_interrupted(self):
        body = self.stage()
        path = self.executor.state / 'test.json'
        record = json.loads(path.read_text());record.update(state='running', worker_pid=99999999)
        path.write_text(json.dumps(record))
        self.executor.recover()
        self.assertEqual(json.loads(path.read_text())['state'], 'interrupted')
    def test_live_worker_not_interrupted(self):
        import os
        self.stage()
        path = self.executor.state / 'test.json'
        record = json.loads(path.read_text());record.update(state='running', worker_pid=os.getpid())
        path.write_text(json.dumps(record))
        self.executor.recover()
        self.assertEqual(json.loads(path.read_text())['state'], 'running')
    def test_autonomous_job_requires_comparison(self):
        from .iteration import gate
        with self.assertRaisesRegex(ValueError, 'comparison'):
            gate(self.executor, {'job_id':'brain-test'}, {'dataset_id':'a' * 64})
    def test_nonimprovement_requires_different_dataset(self):
        from .iteration import gate
        current = {'job_id':'brain-test', 'dataset_id':'a' * 64}
        with patch('research.executor.iteration.outcomes', return_value=[{'outcome':'unchanged'}]):
            with self.assertRaisesRegex(ValueError, 'new validated dataset'):
                gate(self.executor, current, {'dataset_id':'a' * 64, 'max_steps':200})
            directory = self.executor.state / 'verified';directory.mkdir()
            (directory / ('dataset-' + 'b' * 64 + '.json')).write_text('{"data_sha256":"new"}')
            with patch('research.data_factory.cli.read_dataset', side_effect=[{'tasks':[{'split':'train','semantic_sha256':'old'}]}, {'tasks':[{'split':'train','semantic_sha256':'new'}]}]):
                self.assertEqual(gate(self.executor, current, {'dataset_id':'b' * 64})['outcome'], 'unchanged')
    def test_legacy_job_also_requires_comparison(self):
        from .iteration import gate
        with self.assertRaisesRegex(ValueError, 'comparison'):
            gate(self.executor, {'job_id':'grpo-night-100'}, {'dataset_id':'a' * 64})
    def test_catalog_guides_active_training(self):
        (self.root / 'scheduling').mkdir()
        (self.root / 'scheduling/job.json').write_text('{"job_id":"brain-current","max_steps":100}')
        info = self.executor.job_info()
        self.assertFalse(info['complete'])
        self.assertEqual(info['admission_hint'], 'wait_for_training')
        done = self.root / 'runs/brain-current';done.mkdir(parents=True)
        (done / 'job-complete.json').write_text('{}')
        self.assertEqual(self.executor.job_info()['admission_hint'], 'evaluate_current_job')
    def test_adapter_hash_ignores_unrelated_metadata(self):
        from .iteration import adapter_hash
        folder = self.root / 'adapter';folder.mkdir()
        (folder / 'adapter_config.json').write_text('{}')
        (folder / 'adapter_model.safetensors').write_bytes(b'weights')
        before = adapter_hash(folder)
        (folder / 'README.md').write_text('metadata')
        self.assertEqual(adapter_hash(folder), before)
        (folder / 'adapter_model.safetensors').write_bytes(b'changed')
        self.assertNotEqual(adapter_hash(folder), before)
    def test_verified_baseline_reuse_avoids_subprocess(self):
        executor = Executor(self.root, self.root / 'reuse')
        record = {'status':'baseline_reused', 'baseline_complete':True, 'freeze_id':'a' * 64, 'evaluation_id':'b' * 64}
        with patch('research.executor.evidence.reusable_baseline', return_value=record), patch('research.executor.core.subprocess.run') as run:
            self.assertEqual(executor.execute('evaluation.baseline', {}), record)
        run.assert_not_called()
    def test_reuse_requires_current_freeze(self):
        from .evidence import reusable_baseline
        art = self.root / 'research/evaluation/artifacts';art.mkdir(parents=True)
        (art / 'latest.json').write_text(json.dumps({'freeze_id':'a' * 64}))
        with patch('research.executor.evidence.verified_freeze', return_value=True), patch('research.executor.evidence.evaluations', return_value=[{'freeze_id':'b' * 64, 'baseline_complete':True}]):
            self.assertIsNone(reusable_baseline(self.executor))
    def test_reuse_rejects_incomplete_measurement(self):
        from .evidence import reusable_baseline
        art = self.root / 'research/evaluation/artifacts';art.mkdir(parents=True)
        (art / 'latest.json').write_text(json.dumps({'freeze_id':'a' * 64}))
        with patch('research.executor.evidence.verified_freeze', return_value=True), patch('research.executor.evidence.evaluations', return_value=[{'freeze_id':'a' * 64, 'baseline_complete':False}]):
            self.assertIsNone(reusable_baseline(self.executor))
    def test_canary_regression_wins_over_net_gain(self):
        from .iteration import comparison_outcome
        self.assertEqual(comparison_outcome({'a':'pass','b':'fail','c':'fail'}, {'a':'fail','b':'pass','c':'pass'}), 'regressed')
        self.assertEqual(comparison_outcome({'a':'pass','b':'fail'}, {'a':'pass','b':'pass'}), 'improved')
    def test_typed_proposal_rejects_arbitrary_code(self):
        body = {'action_id':'dsl', 'action':'data.propose', 'params':{'proposal':{'input_widths':[2], 'output_width':2, 'expression':{'op':'input','index':0}}}}
        self.assertEqual(self.executor.validate(body)['action'], 'data.propose')
        for bad in [{'proposal':body['params']['proposal'], 'code':'reboot'}, {'proposal':{'input_widths':[2], 'output_width':2, 'expression':{'op':'shell','command':'reboot'}}}]:
            with self.assertRaises(ValueError):
                self.executor.validate({**body, 'params':bad})
    def test_proposal_passes_only_json_stdin(self):
        import subprocess
        executor = Executor(self.root, self.root / 'propose')
        payload = {'input_widths':[2], 'output_width':2, 'expression':{'op':'input','index':0}}
        response = subprocess.CompletedProcess([], 0, '{"status":"planned","plan_id":"abc"}', '')
        with patch('research.executor.core.subprocess.run', return_value=response) as run:
            executor.execute('data.propose', {'proposal':payload})
        self.assertEqual(run.call_args.args[0][-2:], ['propose','--stdin'])
        self.assertEqual(json.loads(run.call_args.kwargs['input']), payload)
        self.assertNotIn('shell', run.call_args.kwargs)
    def test_runtime_identity_mismatch_rejected(self):
        from .iteration import runtime_valid
        image = 'sha256:' + 'a' * 64
        result = {'metadata':{'training_image_id':image}, 'judge_runner_sha256':'judge'}
        self.assertTrue(runtime_valid(result, {'image_id':image}, {'judge_runner_sha256':'judge'}, image))
        self.assertFalse(runtime_valid(result, {'image_id':'sha256:' + 'b' * 64}, {'judge_runner_sha256':'judge'}))
        self.assertFalse(runtime_valid(result, {'image_id':image}, {'judge_runner_sha256':'changed'}))
        self.assertFalse(runtime_valid(result, {'image_id':image}, {'judge_runner_sha256':'judge'}, 'sha256:' + 'c' * 64))
    def test_training_resource_gates(self):
        from .admission import resource_gate
        rows = [{'task_id':str(i)} for i in range(8)]
        with patch('research.executor.admission.shutil.disk_usage', return_value=type('Disk', (), {'free':50 * 1024**3})()):
            resource_gate(self.root, rows)
            with self.assertRaisesRegex(ValueError, '8 distinct'):
                resource_gate(self.root, rows[:7])
        with patch('research.executor.admission.shutil.disk_usage', return_value=type('Disk', (), {'free':50 * 1024**3 - 1})()):
            with self.assertRaisesRegex(ValueError, '50 GiB'):
                resource_gate(self.root, rows)
    def test_compose_ids_are_bounded_distinct(self):
        body = {'action_id':'mix', 'action':'data.compose', 'params':{'dataset_ids':['a'*64,'b'*64]}}
        self.assertEqual(self.executor.validate(body)['action'], 'data.compose')
        for ids in [['a'*64], ['a'*64]*2, ['../bad','b'*64]]:
            with self.assertRaises(ValueError):
                self.executor.validate({**body, 'params':{'dataset_ids':ids}})
    def test_candidate_reuses_verified_comparison_without_gpu(self):
        from .iteration import request_evaluation
        (self.root / 'scheduling').mkdir()
        (self.root / 'scheduling/job.json').write_text('{"job_id":"legacy"}')
        with patch('research.executor.iteration.outcomes', return_value=[{'outcome':'unchanged'}]), patch('research.executor.core.subprocess.run') as run:
            result = request_evaluation(self.executor)
        self.assertTrue(result['comparison_complete'])
        run.assert_not_called()
    def test_candidate_queues_completed_legacy_without_gpu(self):
        from .iteration import request_evaluation
        (self.root / 'scheduling').mkdir()
        (self.root / 'scheduling/job.json').write_text('{"job_id":"legacy"}')
        folder = self.root / 'runs/legacy';folder.mkdir(parents=True)
        (folder / 'job-config.json').write_text('{"job_id":"legacy","max_steps":100}')
        (folder / 'last-exit.json').write_text('{"returncode":0,"step":100}')
        (folder / 'trainer_state_final.json').write_text('{"global_step":100}')
        adapter = folder / 'adapter';adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
        baseline = {'baseline_complete':True,'freeze_id':'a'*64,'evaluation_id':'b'*64}
        with patch('research.executor.iteration.outcomes', return_value=[]), patch('research.executor.evidence.evaluations', return_value=[baseline]), patch('research.executor.core.subprocess.run') as run:
            result = request_evaluation(self.executor)
            self.assertTrue(result['evaluation_pending'])
            self.assertFalse(result['comparison_complete'])
            self.assertTrue((folder / 'evaluation-pending.json').exists())
            (folder / 'evaluation-pending.json').unlink()
            (folder / 'last-exit.json').write_text('{"returncode":0,"step":99}')
            with self.assertRaisesRegex(ValueError, 'not completed'):
                request_evaluation(self.executor)
            self.assertFalse((folder / 'evaluation-pending.json').exists())
        run.assert_not_called()
    def test_new_id_same_bytes_cannot_bypass_iteration_gate(self):
        from .iteration import gate
        folder = self.executor.state / 'verified';folder.mkdir()
        (folder / ('dataset-' + 'b' * 64 + '.json')).write_text('{"data_sha256":"same"}')
        with patch('research.executor.iteration.outcomes', return_value=[{'outcome':'unchanged'}]):
            with self.assertRaisesRegex(ValueError, 'changed training content'):
                gate(self.executor, {'job_id':'brain-x','dataset_id':'a'*64,'data_sha256':'same'}, {'dataset_id':'b'*64})
    def test_reordered_same_semantics_cannot_bypass_iteration_gate(self):
        from .iteration import gate
        folder = self.executor.state / 'verified';folder.mkdir()
        (folder / ('dataset-' + 'b' * 64 + '.json')).write_text('{"data_sha256":"newbytes"}')
        def rows(values):
            return {'tasks':[{'split':'train','semantic_sha256':value} for value in values]}
        old, reordered = rows(['one','two']), rows(['two','one','one'])
        with patch('research.executor.iteration.outcomes', return_value=[{'outcome':'regressed'}]), patch('research.data_factory.cli.read_dataset', side_effect=[old,reordered]):
            with self.assertRaisesRegex(ValueError, 'different training semantics'):
                gate(self.executor, {'job_id':'brain-x','dataset_id':'a'*64,'data_sha256':'oldbytes'}, {'dataset_id':'b'*64})
    def test_limit_type_and_bounds(self):
        for limit in [True, 0, 9, '4']:
            with self.assertRaises(ValueError):
                self.executor.validate({'action_id': 'x', 'action': 'data.plan', 'params': {'limit': limit}})

if __name__ == '__main__':
    unittest.main()
