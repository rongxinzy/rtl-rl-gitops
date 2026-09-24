"""Publish backup only after its complete protocol acceptance; no GPU allocation."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import http.client,hmac,json,os
from pathlib import Path
import urllib.request
KEY=Path('/secrets/backup-key').read_text().strip()
ENGINE='http://172.18.5.123:18001'
def accepted():
 try:
  doc=json.loads(Path('/readiness/READY.json').read_text())
  expected={'chat_text','responses_json','responses_stream','responses_tool_stream','responses_tool_roundtrip'}
  if doc.get('quantization')!='UD-IQ1_S' or not expected.issubset(set(doc.get('protocol_checks',[]))):return None
  if doc.get('model_revision')!='621d456e93e926e4b52f85cff5f634358c1828f9':return None
  if doc.get('engine_revision')!='d94f44e79aa219d8057e8de21f95360a187ebf41':return None
  with urllib.request.urlopen(ENGINE+'/health',timeout=2) as r:
   if r.status==200:return doc
 except Exception:pass
 return None
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def reply(self,code,value):
  data=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_GET(self):self.relay()
 def do_POST(self):self.relay()
 def relay(self):
  ready=accepted()
  if self.path=='/health':self.reply(200 if ready else 503,{'ready':bool(ready)});return
  if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+KEY):self.reply(401,{'error':'unauthorized'});return
  if self.path=='/capabilities':self.reply(200 if ready else 503,ready or {'ready':False});return
  if not ready:self.reply(503,{'error':'backup has not passed acceptance'});return
  if not self.path.startswith('/v1/'):self.reply(404,{'error':'not found'});return
  sent=False;up=None
  try:
   if self.headers.get('Transfer-Encoding'):raise ValueError('unsupported request framing')
   n=int(self.headers.get('Content-Length','0'))
   if not 0<=n<=16*1024*1024:raise ValueError('request too large')
   body=self.rfile.read(n) if self.command=='POST' else None
   if body is not None and len(body)!=n:raise ValueError('short request body')
   up=http.client.HTTPConnection('172.18.5.123',18001,timeout=300)
   headers={'Authorization':'Bearer '+KEY,'Content-Type':self.headers.get('Content-Type','application/json'),'Accept-Encoding':'identity'}
   up.request(self.command,self.path,body=body,headers=headers);r=up.getresponse()
   self.send_response(r.status);self.send_header('Content-Type',r.getheader('Content-Type','application/json'));self.send_header('Connection','close');self.end_headers();sent=True;self.close_connection=True
   while True:
    block=r.read1(65536)
    if not block:break
    self.wfile.write(block);self.wfile.flush()
  except Exception:
   self.close_connection=True
   if not sent:self.reply(502,{'error':'backup unavailable'})
  finally:
   if up:up.close()
ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
