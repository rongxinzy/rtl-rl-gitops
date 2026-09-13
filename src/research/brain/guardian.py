"""Liveness only: remove stale brain Jobs; never manipulate research/task state."""
import datetime as dt,json,os,ssl,time,urllib.request
from pathlib import Path
base=Path('/var/run/secrets/kubernetes.io/serviceaccount')
url='https://'+os.environ['KUBERNETES_SERVICE_HOST']+':'+os.environ.get('KUBERNETES_SERVICE_PORT','443')
headers={'Authorization':'Bearer '+(base/'token').read_text().strip()}
context=ssl.create_default_context(cafile=str(base/'ca.crt'))
def call(path,method='GET'):
 with urllib.request.urlopen(urllib.request.Request(url+path,headers=headers,method=method),context=context,timeout=10) as r:return json.load(r)
ns='/apis/batch/v1/namespaces/rtl-brain'
cron=call(ns+'/cronjobs/rtl-brain-loop')
removed=[]
if not cron['spec'].get('suspend',False):
 for job in call(ns+'/jobs?labelSelector=app%3Drtl-brain-loop')['items']:
  start=job.get('status',{}).get('startTime')
  owned=any(x.get('uid')==cron['metadata']['uid'] for x in job['metadata'].get('ownerReferences',[]))
  age=time.time()-dt.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp() if start else 0
  if owned and job.get('status',{}).get('active') and age>900:
   name=job['metadata']['name'];call(ns+'/jobs/'+name+'?propagationPolicy=Foreground','DELETE');removed.append(name)
print(json.dumps({'guardian_seen':time.time(),'stale_jobs_removed':removed,'respects_suspend':True}))
