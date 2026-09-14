import base64
import socket
import threading
import unittest
from unittest.mock import patch
from proxy import Policy, Server

TOKEN='fixture-proxy-token-never-production-123456'

class Tests(unittest.TestCase):
    def setUp(self):self.p=Policy(['api.swanlab.cn'],TOKEN)
    def test_exact_authority(self):
        self.assertEqual(self.p.destination('api.swanlab.cn:443'),'api.swanlab.cn')
        for v in ('api.swanlab.cn:80','api.swanlab.cn.evil:443','API.SWANLAB.CN:443','api.swanlab.cn.:443','127.0.0.1:443','api.swanlab.cn@evil:443','[::1]:443'):
            with self.assertRaises(ValueError):self.p.destination(v)
    def test_authentication(self):
        self.assertTrue(self.p.authenticate('Basic '+base64.b64encode(('telemetry:'+TOKEN).encode()).decode()))
        self.assertFalse(self.p.authenticate('Basic wrong'))
        self.assertFalse(self.p.authenticate(''))
    def test_private_dns_denied(self):
        for address in ['127.0.0.1','10.0.0.1','169.254.169.254','::1','192.168.1.1','224.0.0.1']:
            with patch('proxy.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(address,443))]):
                with self.assertRaises(ValueError):self.p.resolve('api.swanlab.cn')
    def test_mixed_dns_denied(self):
        with patch('proxy.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('1.1.1.1',443)),(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',443))]):
            with self.assertRaises(ValueError):self.p.resolve('api.swanlab.cn')
    def test_public_dns_pinned(self):
        values=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('1.1.1.1',443))]
        with patch('proxy.socket.getaddrinfo',return_value=values):self.assertEqual(self.p.resolve('api.swanlab.cn')[0][-1],('1.1.1.1',443))
    def test_http_rejections_do_not_echo(self):
        srv=Server(('127.0.0.1',0),self.p);thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
        auth='Basic '+base64.b64encode(('telemetry:'+TOKEN).encode()).decode()
        try:
            requests=[('CONNECT api.swanlab.cn:443 HTTP/1.1\r\n\r\n',407),
                      ('GET https://api.swanlab.cn/secret HTTP/1.1\r\n\r\n',403),
                      ('CONNECT evil.example:443 HTTP/1.1\r\nProxy-Authorization: '+auth+'\r\n\r\n',403),
                      ('CONNECT api.swanlab.cn:443 HTTP/1.1\r\nProxy-Authorization: '+auth+'\r\nProxy-Authorization: '+auth+'\r\n\r\n',400)]
            for raw,status in requests:
                with socket.create_connection(srv.server_address,timeout=2) as sock:
                    sock.sendall(raw.encode());reply=sock.recv(4096)
                self.assertTrue(reply.startswith(f'HTTP/1.1 {status}'.encode()))
                self.assertNotIn(TOKEN.encode(),reply);self.assertNotIn(b'secret',reply)
        finally:srv.shutdown();srv.server_close();thread.join()

if __name__=='__main__':unittest.main()

class SNITests(unittest.TestCase):
    def hello(self, host):
        import ssl
        incoming,outgoing=ssl.MemoryBIO(),ssl.MemoryBIO()
        client=ssl.create_default_context().wrap_bio(incoming,outgoing,server_side=False,server_hostname=host)
        try:client.do_handshake()
        except ssl.SSLWantReadError:pass
        return outgoing.read()
    def test_sni_match(self):
        from proxy import client_hello
        a,b=socket.socketpair()
        try:
            wire=self.hello('api.swanlab.cn');a.sendall(wire)
            self.assertEqual(client_hello(b,'api.swanlab.cn'),wire)
        finally:a.close();b.close()
    def test_sni_mismatch_rejected(self):
        from proxy import client_hello
        a,b=socket.socketpair()
        try:
            a.sendall(self.hello('evil.example'))
            with self.assertRaises(ValueError):client_hello(b,'api.swanlab.cn')
        finally:a.close();b.close()
    def test_plaintext_not_tunneled(self):
        from proxy import client_hello
        a,b=socket.socketpair()
        try:
            a.sendall(b'GET / HTTP/1.1\r\n\r\n')
            with self.assertRaises(ValueError):client_hello(b,'api.swanlab.cn')
        finally:a.close();b.close()
