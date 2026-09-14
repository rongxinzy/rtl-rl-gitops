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

    def test_native_job_never_initializes_relay_run(self):
        self.job['metadata']={'telemetry':'swanlab-native-v1'}
        with patch.object(relay,'snapshot',return_value={'jobs':[self.job]}):
            relay.worker(self.job['job_id'])
        self.sdk.init.assert_not_called()
        self.sdk.log.assert_not_called()
        self.assertEqual(json.loads((self.root/(self.job['job_id']+'.json')).read_text())['status'],'native_telemetry')

    def test_lf_label_alone_does_not_disable_legacy_relay(self):
        self.job['metadata']={'backend':'llamafactory'}
        self.assertFalse(relay.native_telemetry(self.job))

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

    def test_stale_source_never_uploads_or_finishes_from_cached_progress(self):
        stale = self.snapshot(training_complete=True, source_stale=True)
        fresh = self.snapshot(training_complete=True, source_stale=False)
        def during_sleep(_):
            self.sdk.init.assert_not_called()
            self.sdk.log.assert_not_called()
            self.sdk.finish.assert_not_called()
            self.assertEqual(self.receipt()['status'], 'source_stale')
        with patch.object(relay, 'snapshot', side_effect=[stale, fresh]), patch.object(relay.time, 'sleep', side_effect=during_sleep):
            self.assertEqual(relay.worker(self.job['job_id']), 0)
        self.sdk.log.assert_called_once()

    def test_pro_pause_finishes_aborted_and_releases_worker_slot(self):
        paused = self.snapshot(source='pro6000d', phase='paused', attempt_started_at=100)
        with patch.object(relay, 'snapshot', return_value=paused):
            self.assertEqual(relay.worker(self.job['job_id']), 0)
        self.sdk.finish.assert_called_once_with(state='aborted')
        receipt=self.receipt()
        self.assertEqual(receipt['status'], 'paused')
        self.assertEqual(receipt['attempt_started_at'], 100)
        self.assertTrue(relay.receipt_terminal(paused['jobs'][0], receipt))

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

    def test_rich_presentation_records_training_host_and_task(self):
        job = {**self.job, 'secret': 'must-not-appear',
               'metadata': {'orchestrator': 'tekton', 'rank': 8,
                            'data_preflight': {'examples': 12}}}
        device = {'scope': 'current_source_host_observation',
                  'historical_training_hardware_verified': False,
                  'training_gpu': {'index': 0, 'name': 'NVIDIA L20'}, 'host_gpu_count': 2}
        config, description, tags = relay.presentation(job, device)
        self.assertEqual(config['training_device_observation'], device)
        self.assertEqual(config['rank'], 8)
        self.assertEqual(config['metadata_version'], 2)
        self.assertIn('supervised', config['task_type'])
        self.assertIn('validated metric files', config['log_source'])
        self.assertIn('not the relay host', description)
        self.assertIn('Tekton', tags)
        self.assertNotIn('must-not-appear', json.dumps(config))
        data = {'schema_version': 1, 'device_observation': device,
                'jobs': [{**job, 'training_complete': True}]}
        with patch.object(relay, 'snapshot', return_value=data):
            self.assertEqual(relay.worker(job['job_id']), 0)
        args = self.sdk.init.call_args.kwargs
        self.assertEqual(args['job_type'], 'rtl-qlora-sft')
        self.assertEqual(args['group'], 'Qwen3.8-27B-RTL-L20')
        self.assertTrue(args['description'])
        self.assertIn('QLoRa'.lower(), [x.lower() for x in args['tags']])


if __name__ == '__main__':
    unittest.main()
