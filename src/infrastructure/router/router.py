"""Authenticated OpenAI HTTP/SSE primary/backup router with a fenced drain view."""
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import ssl
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from traffic_policy import TrafficPolicy

NS=os.getenv('NAMESPACE','rtl-system')
SECRET=Path(os.getenv('ROUTER_KEY_FILE','/secrets/key')).read_text().strip() if os.getenv('ROUTER_TEST')!='1' else 'test'
BACKGROUND_KEY_PATH=Path(os.getenv('ROUTER_BACKGROUND_KEY_FILE','/secrets/background-key'))
BACKGROUND_KEY=BACKGROUND_KEY_PATH.read_text().strip() if BACKGROUND_KEY_PATH.exists() and os.getenv('ROUTER_TEST')!='1' else None
POLICY=TrafficPolicy(SECRET,BACKGROUND_KEY,night_start=os.getenv('BACKGROUND_NIGHT_START','22:30'),night_end=os.getenv('BACKGROUND_NIGHT_END','07:30'),timezone=os.getenv('SCHEDULE_TIMEZONE','Asia/Shanghai'),clock=lambda:time.time(),force_training_at=os.getenv('FORCE_TRAINING_AT') or None)
BACKENDS={'primary':os.getenv('PRIMARY_URL','http://glm-primary:8000'), 'backup':os.getenv('BACKUP_URL','http://glm-backup:8000')}
API='https://'+os.getenv('KUBERNETES_SERVICE_HOST','kubernetes.default.svc')+':'+os.getenv('KUBERNETES_SERVICE_PORT','443')
LOCK=threading.Lock()
CACHE={'backend':None,'epoch':None,'updated':0}
INFLIGHT={'primary':0,'backup':0}
LAST_BUSINESS_AT=time.time()
ROUTES=set(BACKENDS)|{'maintenance'}
TERMINATING=False


def api(method,path,body=None):
    token=Path('/var/run/secrets/kubernetes.io/serviceaccount/token').read_text().strip()
    req=urllib.request.Request(API+path,method=method,data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    context=ssl.create_default_context(cafile='/var/run/secrets/kubernetes.io/serviceaccount/ca.crt')
    with urllib.request.urlopen(req,context=context,timeout=3) as r:return json.load(r)


def config(): return api('GET',f'/api/v1/namespaces/{NS}/configmaps/glm-router-state')

def refresh():
    while True:
        try:
            cm=config(); d=cm['data']
            if d['backend'] not in ROUTES:raise ValueError('bad backend')
            with LOCK:CACHE.update(backend=d['backend'],epoch=d['epoch'],updated=time.monotonic())
        except Exception: pass  # Keep serving last observed state; never declare drain converged while stale.
        time.sleep(.25)


def fix_created_at(obj):
    if isinstance(obj,dict):
        return {k:(int(v) if k=='created_at' and isinstance(v,float) and v.is_integer() else fix_created_at(v)) for k,v in obj.items()}
    if isinstance(obj,list):return [fix_created_at(v) for v in obj]
    return obj


def normalize(payload):
    try:return json.dumps(fix_created_at(json.loads(payload)),ensure_ascii=False,separators=(',',':')).encode()
    except (ValueError,UnicodeError):return payload


def healthy(backend):
    try:
        with urllib.request.urlopen(BACKENDS[backend]+'/health',timeout=2) as r:return r.status==200
    except Exception:return False


def begin_request(business,identity="protected"):
    global LAST_BUSINESS_AT
    with LOCK:
        backend=CACHE['backend']
        if backend and backend!='maintenance':
            INFLIGHT[backend]+=1
            if business and identity=="protected":LAST_BUSINESS_AT=time.time()
        return backend

def end_request(backend,business,identity="protected"):
    global LAST_BUSINESS_AT
    with LOCK:
        INFLIGHT[backend]-=1
        if business and identity=="protected":LAST_BUSINESS_AT=time.time()


def local_status():
    with LOCK:return {**CACHE,'age':time.monotonic()-CACHE['updated'],'primary_inflight':INFLIGHT['primary'],'backup_inflight':INFLIGHT['backup'],'last_business_at':LAST_BUSINESS_AT,**POLICY.snapshot()}


def cluster_status():
    desired=config()['data']
    pods=api('GET',f'/api/v1/namespaces/{NS}/pods?labelSelector=app%3Dglm-router')['items']
    replicas=[]
    for pod in pods:
        if pod['metadata'].get('deletionTimestamp'):raise RuntimeError('router replica terminating; drain not yet confirmed')
        ready=any(c['type']=='Ready' and c['status']=='True' for c in pod.get('status',{}).get('conditions',[]))
        if not ready or not pod.get('status',{}).get('podIP'):raise RuntimeError('router replica not ready')
        req=urllib.request.Request('http://'+pod['status']['podIP']+':8000/admin/local-status',headers={'Authorization':'Bearer '+SECRET})
        with urllib.request.urlopen(req,timeout=3) as r:replicas.append(json.load(r))
    converged=len(replicas)>=2 and all(x['epoch']==desired['epoch'] and x['backend']==desired['backend'] and x['age']<3 for x in replicas)
    return {'active_backend':desired['backend'],'epoch':desired['epoch'],'converged':converged,
            'primary_inflight':sum(x['primary_inflight'] for x in replicas),
            'backup_inflight':sum(x['backup_inflight'] for x in replicas),
            'last_business_at':max((x.get('last_business_at',time.time()) for x in replicas),default=time.time()),
            'business_idle_seconds':max(0,time.time()-max((x.get('last_business_at',time.time()) for x in replicas),default=time.time())),
            'protected_inflight':sum(x.get('protected_inflight',x['primary_inflight']+x['backup_inflight']) for x in replicas),
            'background_inflight':sum(x.get('background_inflight',0) for x in replicas),
            'last_protected_at':max((x.get('last_protected_at',x.get('last_business_at',time.time())) for x in replicas),default=time.time()),
            'last_background_at':max((x.get('last_background_at',0) for x in replicas),default=0),
            'background_rejected_total':sum(x.get('background_rejected_total',0) for x in replicas),
            'replicas':len(replicas),'healthy_backends':{b:healthy(b) for b in BACKENDS}}


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def setup(self):
        super().setup();self.connection.settimeout(30)
    def log_message(self,*args):pass  # Never log auth headers, prompts, responses or query strings.
    def reply(self,code,value):
        data=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def authorized(self):return POLICY.authenticate(self.headers.get('Authorization',''))=='protected'
    def body(self):
        if self.headers.get('Transfer-Encoding'):raise ValueError('chunked request bodies not supported')
        n=int(self.headers.get('Content-Length','0'))
        if n<0 or n>16*1024*1024:raise ValueError('request body too large')
        b=self.rfile.read(n)
        if len(b)!=n:raise ValueError('incomplete request body')
        return b
    def do_GET(self):self.route()
    def do_POST(self):self.route()
    def do_PUT(self):self.route()
    def route(self):
        global LAST_BUSINESS_AT
        if self.path=='/health':
            ready=bool(CACHE['backend']) and not TERMINATING
            self.reply(200 if ready else 503,{'ready':ready});return
        identity=POLICY.authenticate(self.headers.get('Authorization',''))
        if identity is None:self.close_connection=True;self.reply(401,{'error':'unauthorized'});return
        if identity=='background' and not self.path.startswith('/v1/'):
            self.close_connection=True;self.reply(403,{'error':'workload credential has no control access'});return
        try:
            if self.path=='/admin/local-status' and self.command=='GET':self.reply(200,local_status());return
            if self.path=='/admin/status' and self.command=='GET':self.reply(200,cluster_status());return
            if self.path=='/admin/backend' and self.command=='PUT':
                payload=json.loads(self.body())
                backend=payload['backend']
                force=payload.get('force',False)
                if type(force) is not bool:raise ValueError('force must be boolean')
                if backend not in ROUTES or (force and backend!='maintenance'):raise ValueError('invalid backend')
                if force and not POLICY.force_allowed():
                    self.reply(409,{'error':'forced maintenance outside configured deadline'});return
                if backend=='maintenance':
                    status=cluster_status()
                    if not status['converged'] or (not force and (status['primary_inflight'] or status['business_idle_seconds']<300)):
                        self.reply(409,{'error':'business not idle'});return
                elif not healthy(backend):self.reply(409,{'error':'backend not healthy'});return
                cm=config()
                if cm['data']['backend']!=backend:
                    cm['data']={'backend':backend,'epoch':str(uuid.uuid4())}
                    api('PUT',f'/api/v1/namespaces/{NS}/configmaps/glm-router-state',cm)
                self.reply(200,{'active_backend':backend,'epoch':cm['data']['epoch']});return
            if self.path.startswith('/admin/') or self.command not in ('GET','POST') or not self.path.startswith('/v1/'):
                self.reply(404,{'error':'not found'});return
            body=self.body() if self.command=='POST' else None
        except (ValueError,KeyError):self.reply(400,{'error':'invalid request'});return
        except Exception:self.reply(503,{'error':'control state unavailable'});return
        if TERMINATING:self.reply(503,{'error':'router draining'});return
        business=POLICY.is_business(self.command,self.path)
        if POLICY.admit(identity,self.command,self.path)==503:
            self.reply(503,{'error':{'type':'background_paused','message':'background inference paused for scheduled training; use fallback'}});return
        backend=begin_request(business,identity)
        if backend=='maintenance':self.reply(503,{'error':'scheduled training maintenance'});return
        if not backend:self.reply(503,{'error':'routing not initialized'});return
        ticket=POLICY.begin(identity,self.command,self.path)
        sent=False;up=None
        try:
            url=urllib.parse.urlsplit(BACKENDS[backend]);cls=http.client.HTTPSConnection if url.scheme=='https' else http.client.HTTPConnection
            up=cls(url.hostname,url.port,timeout=10)
            headers={k:v for k,v in self.headers.items() if k.lower() not in ('host','authorization','content-length','connection','transfer-encoding','accept-encoding','upgrade','keep-alive','te','trailer','proxy-authorization','proxy-authenticate')}
            headers['Accept-Encoding']='identity'
            keyfile=Path('/secrets/'+backend+'-key')
            if keyfile.exists() and keyfile.read_text().strip():headers['Authorization']='Bearer '+keyfile.read_text().strip()
            up.request(self.command,(url.path.rstrip('/')+self.path),body=body,headers=headers)
            response=up.getresponse()
            if up.sock:up.sock.settimeout(300)
            ctype=response.getheader('Content-Type','application/json')
            streaming='text/event-stream' in ctype
            if not streaming:
                data=normalize(response.read()) if 'json' in ctype else response.read()
                self.send_response(response.status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();sent=True;self.wfile.write(data)
            else:
                self.send_response(response.status);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers();sent=True;self.close_connection=True
                for line in response:
                    if line.startswith(b'data:'):
                        value=line[5:].strip()
                        if value!=b'[DONE]':line=b'data: '+normalize(value)+b'\n'
                    self.wfile.write(line);self.wfile.flush()
        except Exception:
            self.close_connection=True
            if not sent:
                try:self.reply(502,{'error':'upstream unavailable'})
                except Exception:pass
            # Never replay a partially streamed response or tool call on another backend.
        finally:
            if up:up.close()
            end_request(backend,business,identity)
            POLICY.end(ticket)

def terminate(signum, frame):
    global TERMINATING
    TERMINATING=True
    def wait_for_requests():
        until=time.monotonic()+300
        while time.monotonic()<until:
            with LOCK: active=sum(INFLIGHT.values())
            if not active:break
            time.sleep(.2)
        os._exit(0)
    threading.Thread(target=wait_for_requests,daemon=True).start()

if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminate)
    threading.Thread(target=refresh,daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
