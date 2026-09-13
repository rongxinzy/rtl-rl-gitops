import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from research.ops_agent import agent as a

NOW=2000000000

def inventory():
 d={'metadata':{'name':'glm-router','uid':'dep','generation':1},'spec':{'replicas':2},'status':{'readyReplicas':1,'updatedReplicas':2,'observedGeneration':1}}
 r={'metadata':{'uid':'rs','ownerReferences':[{'kind':'Deployment','controller':True,'uid':'dep'}]}}
 def p(uid,ready):
  return {'metadata':{'name':uid,'uid':uid,'creationTimestamp':'2020-01-01T00:00:00Z','ownerReferences':[{'kind':'ReplicaSet','controller':True,'uid':'rs'}]},'status':{'conditions':[{'type':'Ready','status':'True' if ready else 'False'}]}}
 return {'rtl-system':([d],[r],[p('bad',False),p('good',True)])}

class Models:
 def __init__(self,extra=False):self.extra=extra;self.calls=0
 def plan(self,system,context):
  self.calls+=1
  x={'action':'recover','candidate_id':context['candidate_ids'][0]} if context['candidate_ids'] else {'action':'observe'}
  if self.extra:x['shell']='forbidden'
  return x,{}
class API:
 def __init__(self):self.deletes=[]
 def request(self,path,body=None):self.deletes.append((path,body))

class Tests(unittest.TestCase):
 def setUp(self):
  self.inv=inventory();self.allow={('rtl-system','glm-router')};self.state={'unready_since':{'bad':NOW-700},'day':a.dt.datetime.fromtimestamp(NOW,a.UTC).date().isoformat(),'attempts':0}
 def options(self):return a.candidates(self.inv,self.allow,self.state,NOW)
 def cycle(self,inv=None,dry=False,models=None):
  api=API()
  with patch.object(a,'collect',return_value=({},inv or self.inv)):
   result=a.cycle(api,models or Models(),self.state,self.allow,NOW,lambda s:None,dry)
  return result,api
 def test_redundant_unhealthy_candidate(self):self.assertEqual(len(self.options()),1)
 def test_first_observation_waits(self):self.state={};self.assertEqual(self.options(),{})
 def test_no_healthy_peer(self):self.inv['rtl-system'][2].pop();self.assertEqual(self.options(),{})
 def test_singleton_refused(self):self.inv['rtl-system'][0][0]['spec']['replicas']=1;self.assertEqual(self.options(),{})
 def test_rollout_refused(self):self.inv['rtl-system'][0][0]['status']['updatedReplicas']=1;self.assertEqual(self.options(),{})
 def test_stale_generation_refused(self):self.inv['rtl-system'][0][0]['metadata']['generation']=2;self.assertEqual(self.options(),{})
 def test_owner_uid_mismatch(self):self.inv['rtl-system'][1][0]['metadata']['ownerReferences'][0]['uid']='other';self.assertEqual(self.options(),{})
 def test_forbidden_deployment(self):self.inv['rtl-system'][0][0]['metadata']['name']='gpu';self.assertEqual(self.options(),{})
 def test_cooldown(self):self.state['target_attempts']={'rtl-system/glm-router':NOW-10};self.assertEqual(self.options(),{})
 def test_delete_uid_precondition(self):
  result,api=self.cycle();self.assertEqual(result['status'],'recovery_submitted');self.assertEqual(api.deletes[0][1]['preconditions'],{'uid':'bad'});self.assertIn('pending',self.state)
 def test_dry_run_no_delete(self):result,api=self.cycle(dry=True);self.assertEqual(result['status'],'dry_run_recovery');self.assertFalse(api.deletes)
 def test_extra_action_keys_rejected(self):result,api=self.cycle(models=Models(True));self.assertEqual(result['status'],'rejected');self.assertFalse(api.deletes)
 def test_budget(self):self.state['attempts']=6;result,api=self.cycle();self.assertEqual(result['status'],'rejected');self.assertFalse(api.deletes)
 def test_model_daily_budget(self):
  self.state['model_calls']=60;m=Models();self.cycle(models=m);self.assertEqual(m.calls,0)
 def test_revalidation_changed_uid(self):
  fresh=copy.deepcopy(self.inv);fresh['rtl-system'][2][0]['metadata']['uid']='replacement';api=API()
  with patch.object(a,'collect',side_effect=[({},self.inv),({},fresh)]):r=a.cycle(api,Models(),self.state,self.allow,NOW,lambda s:None,False)
  self.assertEqual(r['status'],'revalidation_failed');self.assertFalse(api.deletes)
 def test_failure_breaks_continuity(self):
  with patch.object(a,'collect',side_effect=RuntimeError('sensitive')):r=a.cycle(API(),Models(),self.state,self.allow,NOW,lambda s:None,False)
  self.assertEqual(self.state['unready_since'],{});self.assertNotIn('sensitive',json.dumps(r))
 def test_pending_blocks_repeat(self):
  self.state['pending']=next(iter(self.options().values()));r,api=self.cycle();self.assertEqual(r['recovery_progress'],'delayed');self.assertFalse(api.deletes)
 def test_findings_preserve_first_seen(self):
  snap={'schedules':[{'name':'night','phase':'WaitingForBusinessIdle','traffic':{'backup_healthy':False}}]};a.findings(snap,self.state,NOW);f=a.findings(snap,self.state,NOW+900);self.assertEqual(f['schedule/night']['first_seen'],NOW);self.assertIn('backup/night',f)
 def test_production_plan_interval_cannot_reduce(self):
  self.state['last_plan']=NOW-100;m=Models()
  with patch.object(a,'collect',return_value=({},self.inv)):a.cycle(API(),m,self.state,self.allow,NOW,lambda s:None,True,600,10)
  self.assertEqual(m.calls,0)
 def test_canary_plan_interval_can_reduce(self):
  self.state['last_plan']=NOW-11;m=Models()
  with patch.object(a,'collect',return_value=({},{})):a.cycle(API(),m,self.state,{('rtl-brain','rtl-ops-recovery-canary')},NOW,lambda s:None,True,10,10)
  self.assertEqual(m.calls,1)
 def test_mixed_allowlist_cannot_reduce(self):
  self.state['last_plan']=NOW-100;m=Models()
  with patch.object(a,'collect',return_value=({},self.inv)):a.cycle(API(),m,self.state,self.allow|{('rtl-brain','rtl-ops-recovery-canary')},NOW,lambda s:None,True,10,10)
  self.assertEqual(m.calls,0)
 def test_unknown_ready_not_recoverable(self):
  self.inv['rtl-system'][2][0]['status']['conditions'][0]['status']='Unknown';self.assertEqual(self.options(),{})
 def test_missing_ready_not_recoverable(self):
  self.inv['rtl-system'][2][0]['status']['conditions']=[];self.assertEqual(self.options(),{})
 def test_gpu_requests_only_not_recoverable(self):
  self.inv['rtl-system'][2][0]['spec']={'containers':[{'resources':{'requests':{'nvidia.com/gpu':1}}}]};self.assertEqual(self.options(),{})
 def test_gpu_init_container_not_recoverable(self):
  self.inv['rtl-system'][2][0]['spec']={'containers':[],'initContainers':[{'resources':{'limits':{'amd.com/gpu':1}}}]};self.assertEqual(self.options(),{})
 def test_pending_scale_zero_not_success(self):
  self.state['pending']=next(iter(self.options().values()));self.inv['rtl-system'][0][0]['spec']['replicas']=0;self.inv['rtl-system'][2].clear();r,api=self.cycle();self.assertEqual(r['recovery_progress'],'superseded_attention_required');self.assertIn('pending',self.state)
 def test_pending_unowned_replacement_not_success(self):
  self.state['pending']=next(iter(self.options().values()));pods=self.inv['rtl-system'][2];pods[0]['metadata']['uid']='new';pods[0]['status']['conditions'][0]['status']='True';pods[0]['metadata']['ownerReferences'][0]['uid']='other';r,api=self.cycle();self.assertEqual(r['recovery_progress'],'delayed');self.assertIn('pending',self.state)
 def test_pending_real_replacement_success(self):
  self.state['pending']=next(iter(self.options().values()));pods=self.inv['rtl-system'][2];pods[0]['metadata']['uid']='new';pods[0]['status']['conditions'][0]['status']='True';r,api=self.cycle();self.assertEqual(r['recovery_progress'],'recovered');self.assertNotIn('pending',self.state)
 def test_resources_no_secret_read_permissions(self):
  items=json.loads(Path(__file__).with_name('resources.json').read_text())['items']
  for item in items:
   for rule in item.get('rules',[]):self.assertNotIn('secrets',rule['resources']);self.assertNotIn('delete',rule['verbs'])
  cron=next(x for x in items if x['kind']=='CronJob');spec=cron['spec']['jobTemplate']['spec']['template']['spec'];secret=spec['volumes'][1]['secret'];self.assertEqual(secret['secretName'],'rtl-brain-secrets');self.assertEqual(len(secret['items']),4)

if __name__=='__main__':unittest.main()
