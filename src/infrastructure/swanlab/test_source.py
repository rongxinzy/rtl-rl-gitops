import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
