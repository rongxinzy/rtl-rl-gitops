import tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from .catalog_cache import CatalogSnapshot
from .core import Executor
from . import l20_branch

class CatalogTests(unittest.TestCase):
    def wait_idle(self, cache):
        deadline=time.monotonic()+2
        while cache.refreshing and time.monotonic()<deadline:
            time.sleep(0.005)
        self.assertFalse(cache.refreshing)

    def test_cold_request_and_concurrent_reads_do_not_wait_for_validation(self):
        entered=threading.Event();release=threading.Event();calls=[]
        def build():
            calls.append(1);entered.set();release.wait(2);return {'items':[1]}
        cache=CatalogSnapshot(build)
        try:
            first=cache.read({'items':[]})
            self.assertEqual(first['catalog_snapshot']['status'],'warming')
            self.assertTrue(entered.wait(1))
            for _ in range(20):
                self.assertTrue(cache.read({})['catalog_snapshot']['refreshing'])
            self.assertEqual(len(calls),1)
        finally:release.set()
        self.wait_idle(cache)
        value=cache.read({});self.assertEqual(value['items'],[1]);value['items'].append(2)
        self.assertEqual(cache.read({})['items'],[1])

    def test_expired_snapshot_remains_explicitly_stale_during_refresh(self):
        clock=[0];release=threading.Event();calls=[]
        def build():
            calls.append(1)
            if len(calls)>1:release.wait(2)
            return {'version':len(calls)}
        cache=CatalogSnapshot(build,ttl=60,monotonic=lambda:clock[0],wall_time=lambda:123)
        cache.read({});self.wait_idle(cache);clock[0]=61
        try:
            old=cache.read({})
            self.assertEqual(old['version'],1);self.assertTrue(old['catalog_snapshot']['stale'])
            self.assertFalse(old['catalog_snapshot']['authoritative_for_admission'])
        finally:release.set()
        self.wait_idle(cache);self.assertEqual(cache.read({})['version'],2)

    def test_refresh_failure_preserves_snapshot_without_exception_text(self):
        clock=[0];calls=[]
        def build():
            calls.append(1)
            if len(calls)>1:raise ValueError('sensitive producer payload')
            return {'version':1}
        cache=CatalogSnapshot(build,monotonic=lambda:clock[0]);cache.read({});self.wait_idle(cache)
        clock[0]=61;cache.read({});self.wait_idle(cache)
        value=cache.read({});self.assertEqual(value['version'],1)
        self.assertEqual(value['catalog_snapshot']['error_type'],'ValueError')
        self.assertNotIn('sensitive',str(value));self.assertEqual(len(calls),2)

    def test_catalog_builder_is_async_and_inspect_does_not_repeat_job_verification(self):
        entered=threading.Event();release=threading.Event()
        def build(_):entered.set();release.wait(2);return {'current_job':{'available':False}}
        with tempfile.TemporaryDirectory() as temp,patch.object(Executor,'_build_catalog',build):
            ex=Executor(Path(temp),Path(temp)/'state')
            try:
                with patch.object(ex,'job_info',side_effect=AssertionError('synchronous verification')):
                    self.assertEqual(ex.inspect()['catalog']['catalog_snapshot']['status'],'warming')
                    self.assertTrue(entered.wait(1))
            finally:release.set();self.wait_idle(ex._catalog_snapshot)

    def test_admission_never_uses_stale_catalog_as_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            ex=Executor(Path(temp),Path(temp)/'state')
            ex._catalog_snapshot.value={'dataset_summary':[{'validated':True}]}
            params={'dataset_id':'a'*64,'evaluation_id':'b'*64}
            with patch.object(ex,'catalog',side_effect=AssertionError('catalog is not admission')),patch('research.executor.evidence.refresh',side_effect=ValueError('live integrity failure')) as refresh:
                for action in ('training.admit','l20.admit'):
                    with self.assertRaisesRegex(ValueError,'live integrity failure'):ex.execute(action,params)
                self.assertEqual(refresh.call_count,2)

if __name__=='__main__':unittest.main()
