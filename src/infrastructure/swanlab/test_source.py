import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

spec = importlib.util.spec_from_file_location('source', Path(__file__).with_name('source.py'))
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / ('l20-' + 'a' * 24)
        (self.folder / 'run').mkdir(parents=True)
        self.identity = {'model_revision': source.REVISION, 'dataset_id': 'rtl-test', 'max_steps': 2}
        self.put('run/job.json', self.identity)
        self.put('state.json', {'phase': 'training', 'secret': 'must-not-appear'})
        self.row = {'step': 1, 'loss': 0.5, 'gradient_norm': 1.2, 'peak_allocated_bytes': 123, 'task_id': 'must-not-appear'}

    def put(self, name, value):
        (self.folder / name).write_text(json.dumps(value))

    def metrics(self, rows, tail=''):
        (self.folder / 'run/metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows) + tail)

    def test_partial_line_and_whitelist(self):
        self.metrics([self.row], '{"step":2')
        snap = source.snapshot(self.root)
        self.assertEqual(len(snap['jobs'][0]['steps']), 1)
        self.assertNotIn('must-not-appear', json.dumps(snap))
        self.assertFalse(snap['jobs'][0]['training_complete'])

    def test_complete_binding(self):
        self.metrics([self.row, {**self.row, 'step': 2}])
        binding = hashlib.sha256((self.folder / 'run/job.json').read_bytes()).hexdigest()
        for name in ('training_metrics.json', 'trainer_state_final.json'):
            self.put('run/' + name, {'global_step': 2, 'job_sha256': binding})
        self.put('state.json', {'phase': 'complete'})
        self.assertTrue(source.collect(self.folder)['training_complete'])
        self.put('run/training_metrics.json', {'global_step': 2, 'job_sha256': 'wrong'})
        self.assertEqual(len(source.snapshot(self.root)['rejected']), 1)

    def test_sequence_and_nonfinite_rejected(self):
        for rows in ([self.row, self.row], [{**self.row, 'step': 2}], [{**self.row, 'loss': float('nan')}], [{**self.row, 'gradient_norm': False}]):
            self.metrics(rows)
            self.assertEqual(len(source.snapshot(self.root)['rejected']), 1)

    def test_wrong_revision_and_terminal_incomplete(self):
        self.identity['model_revision'] = 'huihui'
        self.put('run/job.json', self.identity)
        self.assertEqual(len(source.snapshot(self.root)['rejected']), 1)
        self.identity['model_revision'] = source.REVISION
        self.put('run/job.json', self.identity)
        self.put('state.json', {'phase': 'complete'})
        self.assertEqual(len(source.snapshot(self.root)['rejected']), 1)

    def test_symlink_not_read(self):
        (self.folder / 'run/job.json').unlink()
        (self.folder / 'run/job.json').symlink_to('/etc/passwd')
        self.assertEqual(len(source.snapshot(self.root)['rejected']), 1)

    def test_metadata_whitelist_and_safe_events(self):
        self.identity.update(max_length=512, rank=8, seed=42, learning_rate=5e-5,
                             data_sha256='b' * 64, model_manifest_sha256='c' * 64,
                             recipe_sha256={'train.py': 'd' * 64, 'secret': 'must-not-appear'},
                             api_key='<TEST_ONLY>')
        self.put('run/job.json', self.identity)
        self.put('job.json', {'image_id': 'sha256:' + 'e' * 64, 'orchestrator': 'tekton',
                             'freeze_id': 'freeze-1', 'knowledge_freeze_id': 'knowledge-2',
                             'prompt': 'must-not-appear', 'env': {'KEY': 'must-not-appear'}})
        self.put('run/data-preflight.json', {'examples': 12, 'families': ['adder', 'counter'],
                  'dropped_overlength': 3, 'rows': ['must-not-appear']})
        self.put('run/model-report.json', {'architecture': 'Qwen3_5ForConditionalGeneration',
                 'target_modules': ['model.layers.0.q_proj', 'secret/must-not-appear'],
                 'trainable_parameters': 1200, 'quantized_modules': 10, 'bitsandbytes': '0.48.0',
                 'credential': 'must-not-appear'})
        self.metrics([{**self.row, 'seconds': 4.5, 'checkpoint': 'checkpoint-1',
                       'stdout': 'must-not-appear'}])
        job = source.collect(self.folder)
        self.assertEqual(job['metadata_version'], 2)
        self.assertEqual(job['metadata']['data_preflight']['examples'], 12)
        self.assertEqual(job['metadata']['model_report']['target_module_types'], ['q_proj'])
        self.assertEqual(job['events'][0]['checkpoint'], 'checkpoint-1')
        self.assertEqual(job['steps'][0]['seconds'], 4.5)
        self.assertNotIn('must-not-appear', json.dumps(job))

    def test_optional_metadata_missing_or_bad_does_not_reject_old_run(self):
        self.metrics([self.row])
        (self.folder / 'run/model-report.json').write_text('{bad')
        job = source.collect(self.folder)
        self.assertEqual(job['metadata']['model_report'], {})
        self.assertEqual(len(job['steps']), 1)
        self.metrics([{**self.row, 'seconds': float('nan'), 'checkpoint': '../../secret'}])
        job = source.collect(self.folder)
        self.assertNotIn('seconds', job['events'][0])
        self.assertNotIn('checkpoint', job['events'][0])

    def test_device_current_observation_and_no_shell(self):
        output = '0, NVIDIA L20, 46068, 580.65.06\n1, NVIDIA L20, 46068, 580.65.06\n'
        with patch.object(source.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output)) as call:
            device = source.observe_device()
        self.assertEqual(device['host_gpu_count'], 2)
        self.assertEqual(device['training_gpu']['index'], 0)
        self.assertFalse(device['historical_training_hardware_verified'])
        self.assertEqual(device['scope'], 'current_source_host_observation')
        self.assertEqual(call.call_args.args[0][0], '/usr/bin/querygpu')
        self.assertNotIn('shell', call.call_args.kwargs)
        self.assertNotIn('uuid', json.dumps(device))
        self.assertNotIn('hostname', json.dumps(device))
        with patch.object(source.subprocess, 'run', side_effect=OSError('must-not-appear')):
            device = source.observe_device()
        self.assertIsNone(device['training_gpu'])
        self.assertNotIn('must-not-appear', json.dumps(device))


if __name__ == '__main__':
    unittest.main()
