import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error

os.environ['ROUTER_TEST']='1'
spec=importlib.util.spec_from_file_location('rtl_router',Path(__file__).with_name('router.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class Response(io.BytesIO):
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class Tests(unittest.TestCase):
    def test_only_created_at_changes(self):
        value={'created_at':123.0,'temperature':1.0,'fraction':1.25,'text':'created_at: 123.0 中文','nested':[{'created_at':124.0}], 'created':125.0}
        out=r.normalize(json.dumps(value).encode()); decoded=json.loads(out)
        self.assertIsInstance(decoded['created_at'],int)
        self.assertIsInstance(decoded['created'],float)
        self.assertIsInstance(decoded['temperature'],float)
        self.assertEqual(decoded['text'],value['text'])
        self.assertEqual(decoded['nested'][0]['created_at'],124)
    def test_fractional_created_at_preserved(self):
        self.assertEqual(json.loads(r.normalize(b'{"created_at":1.5}'))['created_at'],1.5)
    def test_non_json_passes_through(self):
        self.assertEqual(r.normalize(b'[DONE]'),b'[DONE]')
    def status(self,epochs=('e','e'),ages=(0,0),counts=(0,0),terminating=False):
        pods=[{'metadata':{},'status':{'podIP':str(i),'conditions':[{'type':'Ready','status':'True'}]}} for i in range(2)]
        if terminating:pods.append({'metadata':{'deletionTimestamp':'now'},'status':{'podIP':'old'}})
        with patch.object(r,'config',return_value={'data':{'backend':'backup','epoch':'e'}}),patch.object(r,'api',return_value={'items':pods}),patch.object(r,'healthy',return_value=True),patch.object(r.urllib.request,'urlopen',side_effect=[Response(json.dumps({'backend':'backup','epoch':epochs[i],'age':ages[i],'primary_inflight':counts[i],'backup_inflight':0}).encode()) for i in range(2)]):
            return r.cluster_status()
    def test_epoch_consistency(self):
        self.assertFalse(self.status(epochs=('e','old'))['converged'])
    def test_stale_replica(self):
        self.assertFalse(self.status(ages=(0,4))['converged'])
    def test_inflight_sum(self):
        self.assertEqual(self.status(counts=(2,3))['primary_inflight'],5)
    def test_terminating_replica_failclosed(self):
        try:out=self.status(terminating=True)
        except RuntimeError:return
        self.assertFalse(out['converged'],'Terminating replicas may hold streams; cannot exclude from drain gate')
    def test_stream_normalization_and_no_replay(self):
        calls=[]
        class Backend(r.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                calls.append(self.path)
                self.rfile.read(int(self.headers.get('Content-Length',0)))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(b'data: {"created_at":123.0,"choices":[{"delta":{"content":"value 1.0"}}]}\n\n')
                self.wfile.flush();self.close_connection=True
        upstream=r.ThreadingHTTPServer(('127.0.0.1',0),Backend)
        router=r.ThreadingHTTPServer(('127.0.0.1',0),r.Handler)
        threads=[threading.Thread(target=server.serve_forever,daemon=True) for server in (upstream,router)]
        for thread in threads:thread.start()
        try:
            with patch.dict(r.BACKENDS,{'primary':'http://127.0.0.1:'+str(upstream.server_address[1]),'backup':'http://127.0.0.1:1'}),patch.dict(r.CACHE,{'backend':'primary'}):
                req=urllib.request.Request('http://127.0.0.1:'+str(router.server_address[1])+'/v1/chat/completions',data=b'{}',headers={'Authorization':'Bearer test'})
                with urllib.request.urlopen(req) as response:data=response.read()
            self.assertIn(b'"created_at":123,',data)
            self.assertIn(b'"content":"value 1.0"',data)
            self.assertEqual(len(calls),1)
            self.assertEqual(r.INFLIGHT['primary'],0)
        finally:
            for server in (router,upstream):server.shutdown();server.server_close()
            for thread in threads:thread.join()
    def test_business_activity_updates_on_start_and_end(self):
        with patch.dict(r.CACHE, {'backend':'primary'}), patch.object(r, 'LAST_BUSINESS_AT', 0):
            with patch.object(r.time, 'time', return_value=100):
                backend=r.begin_request(True)
            self.assertEqual(r.LAST_BUSINESS_AT,100)
            self.assertEqual(r.INFLIGHT['primary'],1)
            with patch.object(r.time, 'time', return_value=500):
                r.end_request(backend,True)
            self.assertEqual(r.LAST_BUSINESS_AT,500)
            self.assertEqual(r.INFLIGHT['primary'],0)
    def test_models_requests_do_not_reset_idle_timer(self):
        with patch.dict(r.CACHE, {'backend':'primary'}), patch.object(r, 'LAST_BUSINESS_AT', 123):
            backend=r.begin_request(False);r.end_request(backend,False)
            self.assertEqual(r.LAST_BUSINESS_AT,123)
    def test_maintenance_denies_new_request_without_touching_existing(self):
        with patch.dict(r.CACHE, {'backend':'maintenance'}), patch.dict(r.INFLIGHT, {'primary':1,'backup':0}), patch.object(r,'LAST_BUSINESS_AT',123):
            self.assertEqual(r.begin_request(True),'maintenance')
            self.assertEqual(r.INFLIGHT['primary'],1)
            self.assertEqual(r.LAST_BUSINESS_AT,123)
    def test_auth_live_http(self):
        server=r.ThreadingHTTPServer(('127.0.0.1',0),r.Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_address[1])+'/admin/local-status'
            with self.assertRaises(urllib.error.HTTPError) as exc:urllib.request.urlopen(url)
            self.assertEqual(exc.exception.code,401)
            req=urllib.request.Request(url,headers={'Authorization':'Bearer test'})
            with urllib.request.urlopen(req) as resp:self.assertIn('primary_inflight',json.load(resp))
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()

class WorkloadAuthTests(unittest.TestCase):
    def test_background_http_permissions_and_spoof(self):
        time_value=1789401600  # Explicit window patch below keeps fixture date irrelevant.
        p=r.TrafficPolicy('test','fixture-background',clock=lambda:time_value)
        server=r.ThreadingHTTPServer(('127.0.0.1',0),r.Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(path,key,method='GET',extra=None):
            req=urllib.request.Request('http://127.0.0.1:'+str(server.server_address[1])+path,
                data=b'{}' if method=='POST' else None,method=method,
                headers={'Authorization':'Bearer '+key,**(extra or {})})
            try:
                with urllib.request.urlopen(req) as resp:return resp.status,resp.read()
            except urllib.error.HTTPError as exc:return exc.code,exc.read()
        try:
            with patch.object(r,'POLICY',p),patch.object(p,'in_night_window',return_value=True),patch.dict(r.CACHE,{'backend':'maintenance'}):
                self.assertEqual(call('/admin/local-status','fixture-background')[0],403)
                status,body=call('/v1/chat/completions','fixture-background','POST')
                self.assertEqual(status,503);self.assertIn(b'background_paused',body)
                self.assertNotIn(b'fixture-background',body)
                status,body=call('/v1/chat/completions','test','POST',{'X-Workload':'background','User-Agent':'bot'})
                self.assertEqual(status,503);self.assertIn(b'scheduled training maintenance',body)
                self.assertNotIn(b'background_paused',body)
                self.assertEqual(call('/v1/chat/completions','unknown','POST',{'X-Workload':'background'})[0],401)
                self.assertEqual(call('/admin/local-status','test')[0],200)
                self.assertEqual(p.snapshot()['background_rejected_total'],1)
                self.assertEqual(r.INFLIGHT['primary'],0)
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_background_backend_accounting_preserves_protected_idle(self):
        with patch.dict(r.CACHE,{'backend':'primary'}),patch.object(r,'LAST_BUSINESS_AT',123):
            backend=r.begin_request(True,'background')
            self.assertEqual(r.INFLIGHT['primary'],1)
            self.assertEqual(r.LAST_BUSINESS_AT,123)
            r.end_request(backend,True,'background')
            self.assertEqual(r.INFLIGHT['primary'],0)
            self.assertEqual(r.LAST_BUSINESS_AT,123)

class ForcedMaintenanceTests(unittest.TestCase):
    def test_authenticated_force_gate_and_convergence(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now=datetime(2026,9,14,23,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        p=r.TrafficPolicy('test','fixture-background',force_training_at='23:30',clock=lambda:now)
        server=r.ThreadingHTTPServer(('127.0.0.1',0),r.Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(payload,key='test'):
            req=urllib.request.Request('http://127.0.0.1:'+str(server.server_address[1])+'/admin/backend',
                data=json.dumps(payload).encode(),method='PUT',headers={'Authorization':'Bearer '+key})
            try:
                with urllib.request.urlopen(req) as resp:return resp.status
            except urllib.error.HTTPError as exc:return exc.code
        try:
            with patch.object(r,'POLICY',p),patch.object(r,'cluster_status',return_value={
                'converged':True,'primary_inflight':2,'business_idle_seconds':0}) as status,patch.object(r,'config',
                return_value={'data':{'backend':'primary','epoch':'old'}}),patch.object(r,'api') as api:
                self.assertEqual(call({'backend':'maintenance','force':True},'fixture-background'),403)
                self.assertEqual(call({'backend':'maintenance','force':True},'unknown'),401)
                self.assertEqual(call({'backend':'maintenance'}),409)
                self.assertEqual(call({'backend':'maintenance','force':False}),409)
                for value in ['true',1,None,[],{}]:
                    self.assertEqual(call({'backend':'maintenance','force':value}),400)
                self.assertEqual(call({'backend':'primary','force':True}),400)
                api.assert_not_called()
                self.assertEqual(call({'backend':'maintenance','force':True}),200)
                self.assertEqual(api.call_args.args[2]['data']['backend'],'maintenance')
                api.reset_mock()
                status.return_value['converged']=False
                self.assertEqual(call({'backend':'maintenance','force':True}),409)
                api.assert_not_called()
                status.return_value['converged']=True
                with patch.object(p,'force_allowed',return_value=False):
                    self.assertEqual(call({'backend':'maintenance','force':True}),409)
                api.assert_not_called()
        finally:
            server.shutdown();server.server_close();thread.join()
