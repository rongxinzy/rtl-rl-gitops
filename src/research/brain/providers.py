"""Bounded model routing; no credentials or provider bodies in failure logs."""
import json, os, time, urllib.error, urllib.request
from pathlib import Path

class Unavailable(RuntimeError): pass

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

OPENER=urllib.request.build_opener(NoRedirect())

def http(url, body=None, headers=None, timeout=90):
    req=urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json', **(headers or {})})
    with OPENER.open(req, timeout=timeout) as r:
        raw=r.read(512*1024+1)
        if len(raw)>512*1024: raise Unavailable('response_limit')
        return json.loads(raw) if raw else {}

def decode(text):
    text=text.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    result=json.loads(text)
    if not isinstance(result,dict): raise ValueError('object required')
    return result

class Models:
    def __init__(self, state, secret_dir='/secrets'):
        self.state=state; self.root=Path(secret_dir)
    def plan(self, system, snapshot):
        fallback=json.loads((self.root/'fallback.json').read_text())
        choices=[('primary',os.environ.get('PRIMARY_URL','http://glm-primary.rtl-system.svc:8000')),
                 ('backup',os.environ.get('BACKUP_URL','http://glm-backup.rtl-system.svc:8000')),
                 ('fallback',fallback['base_url'])]
        errors=[]
        for name,url in choices:
            if self.state.get('cooldowns',{}).get(name,0)>time.time(): continue
            response=None
            try:
                if name=='fallback':
                    response=http(url.rstrip('/')+'/v1/messages',{'model':fallback['model'],
                        'max_tokens':1800,'thinking':{'type':'disabled'},'system':system,'messages':[{'role':'user','content':json.dumps(snapshot,ensure_ascii=False)}]},
                        {'x-api-key':fallback['token'],'anthropic-version':'2023-06-01'},timeout=100)
                    text=''.join(x.get('text','') for x in response.get('content',[]) if x.get('type')=='text')
                else:
                    key=(self.root/(name+'-key')).read_text().strip()
                    headers={'Authorization':'Bearer '+key} if key else {}
                    http(url.rstrip('/')+'/health',headers=headers,timeout=3)
                    response=http(url.rstrip('/')+'/v1/chat/completions',{'model':'GLM-5.3-Flash',
                        'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(snapshot,ensure_ascii=False)}],
                        'max_tokens':1800,'temperature':0.2,'stream':False,
                        'chat_template_kwargs':{'reasoning_effort':'low'}},headers,timeout=70)
                    text=response['choices'][0]['message']['content']
                proposal=decode(text)
                self.state.setdefault('cooldowns',{}).pop(name,None)
                return proposal,{'provider':name,'model':fallback['model'] if name=='fallback' else 'GLM-5.3-Flash',
                                 'usage':response.get('usage',{}),'failures':errors}
            except Exception as exc:
                code=exc.code if isinstance(exc,urllib.error.HTTPError) else type(exc).__name__
                errors.append({'provider':name,'reason':str(code),'stop_reason':str((response or {}).get('stop_reason',''))[:80]})
                self.state.setdefault('cooldowns',{})[name]=time.time()+300
        raise Unavailable(json.dumps(errors))
