"""One fresh, durable research iteration. Kubernetes schedules and supervises it."""
import datetime as dt, hashlib, json, os, ssl, sys, time, urllib.request, urllib.error
from pathlib import Path
from providers import Models, Unavailable, http
from observer import observe
from state import State, atomic, fingerprint
from workflow import MISSION, validate, next_bootstrap, resource_guard

class Executor:
    def __init__(self):
        self.base=os.environ.get('EXECUTOR_URL','http://rtl-executor:18765')
        self.headers={'Authorization':'Bearer '+Path('/secrets/executor-token').read_text().strip()}
    def get(self,path): return http(self.base+path,headers=self.headers,timeout=15)
    def submit(self,action_id,action,params):
        body={'action_id':action_id,'action':action,'params':params,'stage':'stage'}
        staged=http(self.base+'/v1/actions',body,self.headers,15)
        if staged.get('state')!='staged': return staged
        body.update(stage='apply',plan_hash=staged['plan_hash'])
        return http(self.base+'/v1/actions',body,self.headers,15)

def patch_job(job_id):
    if not isinstance(job_id,str) or not job_id.replace('-','').replace('_','').isalnum(): raise ValueError('invalid job id')
    base=Path('/var/run/secrets/kubernetes.io/serviceaccount')
    url='https://'+os.environ['KUBERNETES_SERVICE_HOST']+':'+os.environ.get('KUBERNETES_SERVICE_PORT','443')
    url+='/apis/rtl.rongxin.ai/v1alpha1/namespaces/rtl-system/inferenceschedules/glm-nightly'
    headers={'Authorization':'Bearer '+(base/'token').read_text().strip()}
    context=ssl.create_default_context(cafile=str(base/'ca.crt'))
    # CAS precondition preserves user changes and forbids mode/deadline/permission modifications.
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),context=context,timeout=10) as r: current=json.load(r)
    changes=[{'op':'test','path':'/metadata/resourceVersion','value':current['metadata']['resourceVersion']},
             {'op':'replace','path':'/spec/jobId','value':job_id}]
    req=urllib.request.Request(url,data=json.dumps(changes).encode(),method='PATCH',headers={**headers,'Content-Type':'application/json-patch+json'})
    with urllib.request.urlopen(req,context=context,timeout=10) as r: return json.load(r)['spec']['jobId']

def refresh(store,executor):
    progress=False
    for aid,item in list(store.value['actions'].items()):
        if item.get('state') not in ('completed','failed','interrupted','rejected'):
            try:
                current=executor.get('/v1/actions/'+aid)
                if current.get('state')=='staged':
                    current=executor.submit(aid,item['action'],item['params'])
                item.update(state=current.get('state','unknown'),result=current.get('result'),
                            verified=current.get('evidence_verified',False),updated=time.time())
                if item['state'] in ('completed','failed','interrupted','rejected'):
                    store.event('action_finished',{'action_id':aid,**item})
                    if item['verified'] and item['action'] not in ('inspect','data.plan','data.propose','proposal.stage'):
                        store.value['total_verified_actions']+=1; progress=True
            except urllib.error.HTTPError as exc:
                if exc.code==404:
                    # No remote intent exists (e.g. executor was busy): safe same-ID retry.
                    try:
                        current=executor.submit(aid,item['action'],item['params'])
                        item.update(state=current.get('state','unknown'),result=current.get('result'),verified=current.get('evidence_verified',False),updated=time.time())
                    except urllib.error.HTTPError as retry_error:
                        if retry_error.code in (400,401,403,422):item.update(state='rejected',reason='invalid_or_unauthorized_intent')
                    except Exception: pass
                store.event('action_poll_failed',{'action_id':aid,'type':type(exc).__name__,'http':exc.code},'warn')
            except Exception as exc:
                store.event('action_poll_failed',{'action_id':aid,'type':type(exc).__name__},'warn')
        result=item.get('result') or {}
        if item.get('state')=='completed' and item.get('action')=='training.admit' and result.get('operator_patch_required') and not item.get('operator_patched'):
            try:
                item['operator_patched']=patch_job(result['job_id'])
                store.event('operator_job_selected',{'job_id':item['operator_patched']})
            except Exception as exc:
                store.event('operator_patch_pending',{'type':type(exc).__name__},'warn')
    return progress

def execute(store,executor,plan):
    action=plan['action']; params=plan['params']
    if action=='wait':return
    mark=fingerprint(action,{'params':params,'job_id':store.value.get('current_job_id')} if action=='eval.candidate' else ({'params':params,'job_id':store.value.get('l20_job_id')} if action=='l20.compare' else params))
    matches=[x for x in store.value['actions'].values() if x['fingerprint']==mark]
    repeatable=action in ('inspect','data.plan','data.teacher','l20.status')
    if any(x.get('state') not in ('completed','failed','interrupted','rejected') for x in matches) or (not repeatable and any(x.get('state')=='completed' and x.get('verified',False) for x in matches)):
        store.event('duplicate_action_suppressed',{'action':action});return
    same_context=[x for x in matches if x.get('resource_epoch')==store.value.get('resource_epoch')]
    if len(same_context)>=3 and not repeatable:
        store.event('direction_exhausted',{'action':action,'attempts':len(matches)},'warn');return
    aid='brain-'+mark[:16]+'-'+str(store.value['iteration'])
    item={'action':action,'params':params,'fingerprint':mark,'state':'dispatching','created':time.time(),'resource_epoch':store.value.get('resource_epoch')}
    # Intent precedes network call; an ambiguous response is polled, never blindly replayed.
    store.value['actions'][aid]=item;store.save()
    try:
        r=executor.submit(aid,action,params)
        item.update(state=r.get('state','unknown'),result=r.get('result'),verified=r.get('evidence_verified',False))
        store.event('action_dispatched',{'action_id':aid,'action':action,'state':item['state']})
    except urllib.error.HTTPError as exc:
        item['state']='rejected' if exc.code in (400,401,403,422) else 'unknown'
        store.event('dispatch_response_error',{'action_id':aid,'http':exc.code,'state':item['state']},'warn')
    except Exception as exc:
        item['state']='unknown';store.event('dispatch_uncertain',{'action_id':aid,'type':type(exc).__name__},'warn')

def main():
    try:store=State(os.environ.get('BRAIN_STATE','/state'))
    except BlockingIOError:
        print(json.dumps({'status':'skipped','reason':'another iteration holds the state lock'}));return
    store.value['iteration']+=1;store.heartbeat('running');store.save()
    try:
        executor=Executor(); made_progress=refresh(store,executor)
        if made_progress:(store.root/'attention.json').unlink(missing_ok=True)
        catalog=executor.get('/v1/catalog')
        store.value['resource_epoch']=fingerprint('catalog',{'artifacts':catalog.get('artifacts'),'job':catalog.get('current_job'),'l20_job':(catalog.get('l20',{}).get('current_job') or {}).get('job_id')})
        store.value['current_job_id']=catalog.get('current_job',{}).get('job_id')
        store.value['l20_job_id']=(catalog.get('l20',{}).get('current_job') or {}).get('job_id')
        running=[{'action_id':k,**v} for k,v in store.value['actions'].items() if v.get('state') not in ('completed','failed','interrupted','rejected')]
        # Passive waiting on a real long task is not cognitive stagnation.
        store.value['stale_count']=0 if made_progress else store.value['stale_count']+(0 if running else 1)
        snapshot={'mission':'Improve held-out RTL correctness with reproducible and bounded post-training',
                  'iteration':store.value['iteration'],'catalog':catalog,'cluster_health':observe(),'running':running[-5:],
                  'recent_actions':list(store.value['actions'].values())[-8:],
                  'stale_count':store.value['stale_count'],'last_plan':store.value.get('last_plan'),
                  'milestones':['freeze evaluation','measure Qwen baseline','verify training data','bounded train','compare held-out results']}
        # Metadata only; never send evaluation answers, testbench bodies or credentials to planners.
        if len(json.dumps(snapshot))>24000:
            snapshot['recent_actions']=[{k:x.get(k) for k in ('action','params','state','verified')} for x in snapshot['recent_actions']]
        day=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d');daily=store.value['daily']
        if daily.get('day')!=day:daily.update(day=day,calls=0)
        minimum=int(os.environ.get('MODEL_INTERVAL_SECONDS','900'))
        should_plan=not running and time.time()-store.value.get('last_model_call',0)>=minimum and daily['calls']<int(os.environ.get('MAX_MODEL_CALLS_DAY','60'))
        plan=next_bootstrap(store.value['actions'],catalog) if not running else {'action':'wait','params':{},'reason':'A durable worker is already running'}
        if should_plan:
            daily['calls']+=1;store.value['last_model_call']=time.time();store.save()
            try:
                proposal,usage=Models(store.value).plan(MISSION,snapshot)
                plan=validate(proposal);store.event('model_plan',{'plan':plan,**usage},'decision')
                store.value['last_provider']=usage['provider']
            except Exception as exc:
                detail={'type':type(exc).__name__}
                if isinstance(exc,Unavailable):
                    try:detail['providers']=json.loads(str(exc))
                    except ValueError:pass
                store.event('planner_unavailable',detail,'warn')
        plan=resource_guard(validate(plan),catalog,next_bootstrap(store.value['actions'],catalog))
        plan=validate(plan); store.value['last_plan']=plan
        if store.value['stale_count']>=4:
            atomic(store.root/'attention.json',{'time':time.time(),'kind':'no_verified_progress','stale_count':store.value['stale_count'],'plan':plan})
        execute(store,executor,plan);store.value['last_successful_iteration']=time.time();store.save();store.heartbeat('idle')
        print(json.dumps({'iteration':store.value['iteration'],'action':plan['action'],'provider':store.value.get('last_provider'),'running_actions':len(running),'verified_actions':store.value['total_verified_actions']}),flush=True)
    except Exception as exc:
        store.event('iteration_error',{'type':type(exc).__name__},'error');store.save();store.heartbeat('error');raise

if __name__=='__main__':main()
