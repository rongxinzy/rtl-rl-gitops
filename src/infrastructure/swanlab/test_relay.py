"""Lifecycle checks with an SDK double; no credentials or network required."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('relay', Path(__file__).with_name('relay.py'))
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = {'job_id': 'l20-' + 'a' * 24, 'job_sha256': 'b' * 64,
                    'model_revision': 'official-pin', 'dataset_id': 'test', 'max_steps': 2,
                    'phase': 'training', 'training_complete': False,
                    'steps': [{'step': 1, 'loss': 0.5, 'gradient_norm': 1.0, 'peak_allocated_bytes': 123}]}
        self.sdk = SimpleNamespace(init=Mock(side_effect=lambda **kw: SimpleNamespace(id=kw['id'], url='https://swanlab.cn/example')),
                                   log=Mock(), finish=Mock(), Settings=Mock())
        for context in (patch.object(relay, 'STATE', self.root), patch.dict('sys.modules', {'swanlab': self.sdk}),
                        patch.object(relay.signal, 'signal')):
            context.start()
            self.addCleanup(context.stop)

    def snapshot(self, **changes):
        job = copy.deepcopy(self.job)
        job.update(changes)
        return {'schema_version': 1, 'jobs': [job]}

    def receipt(self):
        return json.loads((self.root / (self.job['job_id'] + '.json')).read_text())

    def test_resume_uses_same_id_and_replays_metrics(self):
        final = self.snapshot(training_complete=True, phase='complete')
        with patch.object(relay, 'snapshot', return_value=final):
            self.assertEqual(relay.worker(self.job['job_id']), 0)
            self.assertEqual(relay.worker(self.job['job_id']), 0)
        configs = [call.kwargs for call in self.sdk.init.call_args_list]
        self.assertEqual(configs[0]['id'], configs[1]['id'])
        self.assertEqual(configs[0]['resume'], 'allow')
        self.assertEqual(self.sdk.log.call_count, 2)
        self.assertNotEqual(relay.run_id(self.job), relay.run_id({**self.job, 'job_sha256': 'c' * 64}))

    def test_partial_stays_running_then_finishes_success(self):
        final = self.snapshot(training_complete=True, steps=self.job['steps'] + [{**self.job['steps'][0], 'step': 2}])
        def during_sleep(_):
            self.sdk.finish.assert_not_called()
            self.assertEqual(self.receipt()['status'], 'running')
            self.assertEqual(self.receipt()['step'], 1)
        with patch.object(relay, 'snapshot', side_effect=[self.snapshot(), final]), patch.object(relay.time, 'sleep', side_effect=during_sleep):
            self.assertEqual(relay.worker(self.job['job_id']), 0)
        self.sdk.finish.assert_called_once_with(state='success')
        self.assertEqual([call.kwargs['step'] for call in self.sdk.log.call_args_list], [1, 2])
        self.assertEqual(self.receipt()['status'], 'completed')
        self.assertTrue(self.receipt()['sdk_finish_returned'])

    def test_failed_job_is_crashed(self):
        with patch.object(relay, 'snapshot', return_value=self.snapshot(phase='failed')):
            self.assertEqual(relay.worker(self.job['job_id']), 0)
        self.sdk.finish.assert_called_once_with(state='crashed')
        self.assertEqual(self.receipt()['status'], 'failed')

    def test_stale_source_does_not_finish_success(self):
        with patch.object(relay, 'snapshot', side_effect=[self.snapshot(), ValueError('source_stale')]), patch.object(relay.time, 'sleep'):
            self.assertEqual(relay.worker(self.job['job_id']), 1)
        self.sdk.finish.assert_called_once_with(state='aborted')
        self.assertEqual(self.receipt()['status'], 'retry')
        self.assertNotIn('sdk_finish_returned', self.receipt())
        path = self.root / 'snapshot.json'
        path.write_text(json.dumps(self.snapshot()))
        os.utime(path, (1, 1))
        with patch.object(relay, 'SNAPSHOT', path), self.assertRaisesRegex(ValueError, 'source_stale'):
            relay.snapshot()

    def test_sdk_finish_failure_does_not_record_completion(self):
        self.sdk.finish.side_effect = RuntimeError('sensitive provider details')
        with patch.object(relay, 'snapshot', return_value=self.snapshot(training_complete=True)):
            self.assertEqual(relay.worker(self.job['job_id']), 1)
        self.assertEqual(self.receipt()['status'], 'retry')
        self.assertNotIn('sensitive', json.dumps(self.receipt()))
        self.assertNotIn('sdk_finish_returned', self.receipt())


if __name__ == '__main__':
    unittest.main()
