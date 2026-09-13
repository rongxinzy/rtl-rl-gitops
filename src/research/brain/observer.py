"""Read fixed Kubernetes endpoints and expose status fields only, never pod specs."""
import json,os,ssl,urllib.request
from pathlib import Path

def summarize(kind,items):
    rows=[]
    for item in items[:80]:
        meta=item.get('metadata',{});status=item.get('status',{})
        row={'name':meta.get('name'),'namespace':meta.get('namespace')}
        if kind=='pods':
            row.update(phase=status.get('phase'),containers=[{
                'name':s.get('name'),'ready':s.get('ready'),'restarts':s.get('restartCount'),
                'waiting':s.get('state',{}).get('waiting',{}).get('reason'),
                'exit_code':s.get('state',{}).get('terminated',{}).get('exitCode')}
                for s in status.get('containerStatuses',[])])
        elif kind=='nodes':
            row['conditions']={c['type']:c['status'] for c in status.get('conditions',[])}
        elif kind=='events':
            row.update(type=item.get('type'),reason=item.get('reason'),count=item.get('count'))
        elif kind=='pipelineruns':
            condition=next((c for c in status.get('conditions',[]) if c.get('type')=='Succeeded'),{})
            row.update(uid=meta.get('uid'),condition=condition.get('status','Unknown'),reason=condition.get('reason'),
                tasks=[{'name':ref.get('name'),'kind':ref.get('kind'),'pipeline_task':ref.get('pipelineTaskName')}
                       for ref in status.get('childReferences',[])[:80]])
        else:
            row.update({k:status[k] for k in ('readyReplicas','availableReplicas','numberReady','desiredNumberScheduled','active','failed','succeeded') if k in status})
        rows.append(row)
    return rows

def observe():
    base=Path('/var/run/secrets/kubernetes.io/serviceaccount')
    if not (base/'token').exists():return {'available':False}
    url='https://'+os.environ['KUBERNETES_SERVICE_HOST']+':'+os.environ.get('KUBERNETES_SERVICE_PORT','443')
    context=ssl.create_default_context(cafile=str(base/'ca.crt'))
    headers={'Authorization':'Bearer '+(base/'token').read_text().strip()}
    paths=[('nodes','/api/v1/nodes')]
    for ns in ('rtl-system','rtl-egress','rtl-brain','rtl-pipelines'):
        paths.extend([(ns+'/pods','/api/v1/namespaces/'+ns+'/pods'),(ns+'/events','/api/v1/namespaces/'+ns+'/events')])
    paths.append(('rtl-pipelines/pipelineruns','/apis/tekton.dev/v1/namespaces/rtl-pipelines/pipelineruns'))
    result={}
    for label,path in paths:
        try:
            req=urllib.request.Request(url+path+'?limit=80',headers=headers)
            with urllib.request.urlopen(req,context=context,timeout=3) as response:items=json.load(response).get('items',[])
            if label.endswith('/events'):items=sorted(items,key=lambda x:x.get('lastTimestamp') or '',reverse=True)[:5]
            result[label]=summarize(label.split('/')[-1],items)
        except Exception as exc:result[label]={'available':False,'error_type':type(exc).__name__}
    return result
