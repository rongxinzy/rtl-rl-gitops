"""One low-priority daytime teacher call; never preempts business traffic."""
import datetime as dt
import fcntl
import json
import os
import signal
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
from zoneinfo import ZoneInfo


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None
OPENER=urllib.request.build_opener(NoRedirect())

def http(url, payload=None, token=None, timeout=5):
    headers={'Content-Type':'application/json'}
    if token: headers['Authorization']='Bearer '+token
    req=urllib.request.Request(url,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
    # This fixed CLI runs on Linux in its main thread. Bound the whole exchange,
    # including response reads, rather than resetting a timeout per socket read.
    def expired(*_):raise TimeoutError('request_deadline')
    previous=signal.signal(signal.SIGALRM,expired)
    old_timer=signal.setitimer(signal.ITIMER_REAL,timeout)
    started=time.monotonic()
    try:
        with OPENER.open(req,timeout=timeout) as response:
            raw=response.read(131073)
            if len(raw)>131072:raise ValueError('oversize_response')
            return json.loads(raw)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.001,old_timer[0]-(time.monotonic()-started)),old_timer[1])



def daylight(now):
    hour=now.astimezone(ZoneInfo('Asia/Shanghai')).hour
    return 8<=hour<22


def idle(status):
    return (status.get('active_backend')=='primary' and status.get('converged') is True
            and status.get('healthy_backends',{}).get('primary') is True
            and status.get('primary_inflight')==0 and status.get('business_idle_seconds',-1)>=60)


def token_file(path, optional=False):
    file=Path(path)
    if optional and not file.exists():return None
    token=file.read_text().strip()
    if not token:
        if optional:return None
        raise ValueError('empty_credential')
    return token


class Teacher:
    def __init__(self, state, router_url, primary_url, router_token, primary_token=None, call=http, clock=None):
        self.state=Path(state);self.state.mkdir(parents=True,exist_ok=True)
        self.router_url=router_url.rstrip('/');self.primary_url=primary_url.rstrip('/')
        self.router_token=router_token;self.primary_token=primary_token;self.call=call
        self.clock=clock or (lambda:dt.datetime.now(dt.timezone.utc))
    def status(self):
        return self.call(self.router_url+'/admin/status',token=self.router_token)
    def produce(self):
        with (self.state/'teacher.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return {'status':'deferred','reason':'teacher_busy'}
            result=self.once()
            result['recorded_at']=self.clock().isoformat()
            with (self.state/'events.jsonl').open('a') as history:
                history.write(json.dumps(result)+'\n')
            path=self.state/'last-result.json';temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(result));os.replace(temp,path)
            return result
    def once(self):
        now=self.clock()
        if not daylight(now):return {'status':'deferred','reason':'outside_day_window'}
        marker=self.state/'last-attempt.json'
        if marker.exists() and now.timestamp()-json.loads(marker.read_text())['time']<300:
            return {'status':'deferred','reason':'rate_limit'}
        try:
            before=self.status()
        except Exception:return {'status':'deferred','reason':'router_unavailable'}
        if not idle(before):return {'status':'deferred','reason':'business_active'}
        nonce=uuid.uuid4().hex
        # No held-out tasks, DUT references or testbenches enter the teacher prompt.
        from research.data_factory.families import FAMILIES,HOLDOUT
        families=sorted(set(FAMILIES)-set(HOLDOUT))
        prompt=('Propose one original small combinational unsigned bitvector task as JSON only. '
            'Return exactly {"input_widths":[...],"output_width":N,"expression":AST}. '
            '1-4 inputs, each width1-10; total input bits<=10; output width1-10. '
            'AST<=31 nodes, depth<=5. Operators: input(index),const(value),not(arg),'
            'and/or/xor/add/sub(left,right),mux(cond,yes,no),shl/shr(arg,amount0-9). '
            'Each node is an object with op plus the listed fields; constants fit output width. '
            'All intermediate results truncate to output width; mux condition is nonzero. '
            'Use multiple input-dependent operations; avoid constants and duplicate basic families. '
            'Training family names: '+','.join(families)+'. Diversity nonce: '+nonce)
        # Recheck wall clock immediately before issuing the only generation request.
        if not daylight(self.clock()):return {'status':'deferred','reason':'day_window_ended'}
        temp=marker.with_suffix('.tmp');temp.write_text(json.dumps({'time':now.timestamp()}));os.replace(temp,marker)
        started=time.monotonic()
        payload={'model':'GLM-5.3-Flash','messages':[{'role':'user','content':prompt}],
                 'temperature':0.8,'max_tokens':1200,'stream':False,'chat_template_kwargs':{'reasoning_effort':'low'}}
        metrics={'request_count':1}
        try:
            response=self.call(self.primary_url+'/v1/chat/completions',payload,self.primary_token,timeout=60)
            metrics.setdefault('generation_seconds',round(time.monotonic()-started,3))
            usage=response.get('usage',{})
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                if type(usage.get(key)) is int:metrics[key]=usage[key]
            choice=response['choices'][0]
            if choice.get('finish_reason')!='stop':raise ValueError('incomplete_generation')
            proposal=json.loads(choice['message']['content'])
            from research.data_factory.dsl import normalize,make_dsl
            proposal=normalize(proposal);make_dsl(proposal)
            from research.data_factory.cli import propose
            published=propose(proposal)
            result={'status':published['status'],'plan_id':published['plan_id'],'proposal':proposal,'validated':False}
        except (ValueError,TypeError,KeyError,IndexError):
            result={'status':'rejected','reason':'invalid_or_duplicate_dsl'}
        except Exception:
            result={'status':'deferred','reason':'teacher_unavailable'}
        metrics.setdefault('generation_seconds',round(time.monotonic()-started,3))
        result.update(metrics)
        try:result['business_active_after']=not idle(self.status())
        except Exception:result['business_active_after']=True
        result['next_request_allowed']=False  # Every invocation is strictly one generation.
        return result


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['produce']);parser.parse_args()
    try:
        worker=Teacher(os.getenv('RTL_TEACHER_STATE','/root/rtl-rl/research/teacher/state'),
            os.getenv('RTL_TEACHER_ROUTER_URL','http://glm-router.rtl-system.svc.cluster.local:8000'),
            os.getenv('RTL_TEACHER_PRIMARY_URL','http://127.0.0.1:30000'),
            token_file(os.getenv('RTL_TEACHER_ROUTER_TOKEN_FILE','/root/rtl-rl/secrets/teacher-router-token')),
            token_file(os.getenv('RTL_TEACHER_PRIMARY_TOKEN_FILE','/root/rtl-rl/secrets/teacher-primary-token'),optional=True))
        result=worker.produce()
    except Exception:result={'status':'deferred','reason':'teacher_configuration_unavailable'}
    print(json.dumps(result))

if __name__=='__main__':main()
