"""Bounded authenticated HTTP; no redirects, prompt logging, or arbitrary endpoints."""
import json
from pathlib import Path
import re
import ssl
import urllib.request

JOB = re.compile(r'^l20-[0-9a-f]{24}$')
UID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
HEX = re.compile(r'^[0-9a-f]{64}$')
WORKER = 'http://172.18.6.123:18766'
EXECUTOR = 'http://172.18.4.199:18765'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None

def request(url, token_file, body=None, method=None, context=None):
    token = Path(token_file).read_text().strip()
    if not token:
        raise ValueError('missing client credential')
    handlers = [urllib.request.ProxyHandler({}), NoRedirect()]
    if context:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
          method=method, headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
    with urllib.request.build_opener(*handlers).open(req, timeout=20) as response:
        raw = response.read(1024*1024+1)
    if len(raw)>1024*1024:
        raise ValueError('oversize control response')
    return json.loads(raw)

def worker(path, body=None):
    return request(WORKER+path, '/secrets/worker/token', body)

def executor(path, body=None):
    return request(EXECUTOR+path, '/secrets/executor/token', body)

def identity(job, uid):
    if not JOB.fullmatch(job) or not UID.fullmatch(uid):
        raise ValueError('invalid fixed job/run identity')

def check_status(value, job, uid, unbound=False):
    identity(job, uid)
    if value.get('job_id') != job or value.get('orchestrator') != 'tekton':
        raise ValueError('job orchestration mismatch')
    if value.get('run_uid') != uid and not (unbound and value.get('run_uid') is None):
        raise ValueError('PipelineRun ownership mismatch')
    if not HEX.fullmatch(value.get('job_sha256','')):
        raise ValueError('missing immutable job binding')
    return value

def result(name, value):
    path = Path('/tekton/results') / name
    path.write_text(json.dumps(value, separators=(',',':')) if not isinstance(value,str) else value)
