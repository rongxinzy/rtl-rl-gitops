"""Create exactly one fixed PipelineRun for a newly admitted Tekton-mode job."""
import json
import os
from pathlib import Path
import ssl
import time
import urllib.error
from client import JOB, request, worker

NAMESPACE='rtl-pipelines'
PIPELINE='rtl-l20-experiment-v1'
SA=Path('/var/run/secrets/kubernetes.io/serviceaccount')

def api(path, body=None):
    base='https://'+os.environ['KUBERNETES_SERVICE_HOST']+':'+os.environ.get('KUBERNETES_SERVICE_PORT','443')
    return request(base+path,SA/'token',body,context=ssl.create_default_context(cafile=str(SA/'ca.crt')))

def manifest(job):
    if not JOB.fullmatch(job):
        raise ValueError('invalid worker job')
    return {'apiVersion':'tekton.dev/v1','kind':'PipelineRun','metadata':{'name':'rtl-'+job,'namespace':NAMESPACE,
            'labels':{'rtl.ai/managed-by':'tekton-bridge','rtl.ai/job':job}},
            'spec':{'pipelineRef':{'name':PIPELINE},'params':[{'name':'job-id','value':job}],
                    'timeouts':{'pipeline':'2h0m0s','tasks':'1h50m0s','finally':'5m0s'},
                    'taskRunTemplate':{'serviceAccountName':'rtl-experiment','podTemplate':{'nodeSelector':{'kubernetes.io/hostname':'rtl-control'},'securityContext':{'fsGroup':65532,'runAsUser':65532,'runAsGroup':65532,'runAsNonRoot':True,'seccompProfile':{'type':'RuntimeDefault'}}}}}}

def reconcile():
    state=worker('/status');job=state.get('current_job') or {}
    if job.get('orchestrator')!='tekton' or job.get('phase') in ('complete','failed'):
        return
    job=worker('/jobs/'+job['job_id']+'/status')
    expected=manifest(job['job_id']);collection='/apis/tekton.dev/v1/namespaces/'+NAMESPACE+'/pipelineruns'
    path=collection+'/'+expected['metadata']['name']
    try:
        current=api(path)
    except urllib.error.HTTPError as exc:
        if exc.code!=404:
            raise
        if job.get('run_uid'):
            worker('/jobs/'+job['job_id']+'/abort',{'run_uid':job['run_uid']})
            return {'job_id':job['job_id'],'condition':'Deleted','action':'checkpoint_abort_requested'}
        try:
            current=api(collection,expected)
            print(json.dumps({'event':'pipeline_created','job_id':job['job_id'],'name':current['metadata']['name']}),flush=True)
        except urllib.error.HTTPError as create_error:
            if create_error.code!=409:
                raise
            current=api(path)
    actual=current['spec']
    if actual.get('pipelineRef',{}).get('name')!=PIPELINE or actual.get('params')!=expected['spec']['params'] or 'pipelineSpec' in actual:
        raise ValueError('existing PipelineRun identity mismatch')
    if job.get('run_uid') and job['run_uid']!=current['metadata']['uid']:
        worker('/jobs/'+job['job_id']+'/abort',{'run_uid':job['run_uid']})
        return {'job_id':job['job_id'],'condition':'Replaced','action':'checkpoint_abort_requested'}
    cond=next((x for x in current.get('status',{}).get('conditions',[]) if x['type']=='Succeeded'),{})
    if cond.get('status')=='False':
        worker('/jobs/'+job['job_id']+'/abort',{'run_uid':current['metadata']['uid']})
    return {'job_id':job['job_id'],'pipelinerun':current['metadata']['name'],'condition':cond.get('status','Unknown')}

def main():
    previous=None
    while True:
        try:
            state=reconcile()
            if state!=previous:
                print(json.dumps({'event':'bridge_state','state':state}),flush=True);previous=state
        except Exception as exc:
            print(json.dumps({'event':'bridge_error','error_type':type(exc).__name__}),flush=True)
        time.sleep(15)

if __name__=='__main__':
    main()
