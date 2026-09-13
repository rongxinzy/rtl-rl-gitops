"""Authenticated typed L20 branch API, no shell, URL or model path parameters."""
import json,threading,hmac,os,sys,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
try:
 from .common import ROOT,ID,admit,status,atomic,sha,canonical,LOCK,compare_complete
 from .artifacts import verify_all,adapter_digest
 from . import manager,phase_control
except ImportError:
 from common import ROOT,ID,admit,status,atomic,sha,canonical,LOCK,compare_complete
 from artifacts import verify_all,adapter_digest
 import manager,phase_control
class Handler(BaseHTTPRequestHandler):
 token=None
 def setup(self):
  super().setup();self.connection.settimeout(10)
 def log_message(self,*args):pass
 def reply(self,code,value):
  raw=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def auth(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+self.token)
 def do_GET(self):
  if not LOCK.acquire(timeout=1):return self.reply(503,{'error':'worker_busy'})
  try:return self.get_locked()
  finally:LOCK.release()
 def get_locked(self):
  if not self.auth():return self.reply(401,{'error':'unauthorized'})
  if self.path=='/status':return self.reply(200,status())
  parts=self.path.strip('/').split('/')
  if len(parts)==3 and parts[0]=='jobs' and ID.fullmatch(parts[1]) and parts[2]=='status':
   try:return self.reply(200,phase_control.status(ROOT/'jobs'/parts[1]))
   except (OSError,ValueError,KeyError,TypeError):return self.reply(404,{'error':'invalid_job_state'})
  if len(parts)==3 and parts[0]=='jobs' and ID.fullmatch(parts[1]) and parts[2]=='artifacts':
   folder=ROOT/'jobs'/parts[1]
   try:
    state=json.loads((folder/'state.json').read_text())
    if state['phase'] not in ('awaiting_evaluation','complete'):return self.reply(409,{'error':'not_finished'})
    verify_all(folder)
    value={'job':json.loads((folder/'job.json').read_text()),'state':state}
    for name in ('baseline','candidate','knowledge_baseline','knowledge_candidate'):
     raw=(folder/(name+'.jsonl')).read_bytes()
     if len(raw)>256*1024:raise ValueError('bounded generation artifact')
     value[name]=[json.loads(x) for x in raw.splitlines()];value[name+'_sha256']=sha(raw);value[name+'_raw']=raw.decode()
    value['training_metrics']=json.loads((folder/'run/training_metrics.json').read_text())
    value['adapter_sha256']=adapter_digest(folder)
    value['job_sha256']=sha((folder/'job.json').read_bytes())
    value['model_manifest_sha256']=json.loads((folder/'model-binding.json').read_text())['model_manifest_sha256']
    return self.reply(200,value)
   except (OSError,ValueError,KeyError,TypeError):return self.reply(404,{'error':'missing_artifact'})
  return self.reply(404,{'error':'not_found'})
 def do_POST(self):
  if not self.auth():return self.reply(401,{'error':'unauthorized'})
  try:
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=5*1024*1024 or self.headers.get('Transfer-Encoding'):raise ValueError('body_limit')
   self.connection.settimeout(10);body=json.loads(self.rfile.read(length))
   if not LOCK.acquire(timeout=1):return self.reply(503,{'error':'worker_busy'})
   try:
    if self.path=='/jobs':return self.reply(200,admit(body))
    parts=self.path.strip('/').split('/')
    if len(parts)==3 and parts[0]=='jobs' and ID.fullmatch(parts[1]) and parts[2]=='abort':
     return self.reply(200,phase_control.abort(ROOT/'jobs'/parts[1],body))
    if len(parts)==3 and parts[0]=='jobs' and ID.fullmatch(parts[1]) and parts[2]=='phase':
     return self.reply(200,phase_control.authorize(ROOT/'jobs'/parts[1],body))
    if len(parts)==3 and parts[0]=='jobs' and ID.fullmatch(parts[1]) and parts[2]=='evaluation':
     return self.reply(200,compare_complete(ROOT/'jobs'/parts[1],body))
   finally:LOCK.release()
   return self.reply(404,{'error':'not_found'})
  except (ValueError,TypeError,KeyError,OSError):return self.reply(400,{'error':'invalid_request'})
def main():
 Handler.token=(ROOT/'api-key').read_text().strip()
 if len(Handler.token)<32:raise ValueError('token too short')
 ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'jobs').mkdir(exist_ok=True)
 threading.Thread(target=manager.loop,daemon=True).start()
 ThreadingHTTPServer((os.getenv('L20_WORK_BIND','127.0.0.1'),18766),Handler).serve_forever()
if __name__=='__main__':main()
