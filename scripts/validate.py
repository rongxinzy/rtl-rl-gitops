"""Validate the deployable resource boundary before a revision reaches Argo CD."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DYNAMIC={'Secret','Pod','ReplicaSet','Job','PipelineRun','TaskRun','Lease','Event','PersistentVolume'}
seen=set();count=0
for path in (ROOT/'clusters').rglob('*.json'):
 value=json.loads(path.read_text());items=value.get('items',[value]) if value.get('kind')=='List' else [value]
 for item in items:
  kind=item['kind'];meta=item['metadata'];identity=(kind,meta.get('namespace',''),meta['name'])
  assert identity not in seen, ('duplicate',identity)
  seen.add(identity);count+=1
  assert kind not in DYNAMIC,('runtime_or_secret_resource',identity)
  assert not set(meta)&{'uid','resourceVersion','managedFields','creationTimestamp','ownerReferences'},identity
  assert 'status' not in item,identity
  assert 'kubectl.kubernetes.io/last-applied-configuration' not in meta.get('annotations',{}),identity
  if kind in ('Namespace','PersistentVolumeClaim','CustomResourceDefinition'):
   assert 'Prune=false' in meta['annotations']['argocd.argoproj.io/sync-options'],identity
print(json.dumps({'valid':True,'static_resources':count}))

transport=json.loads((ROOT/"clusters/lab/argocd/resources.json").read_text())
embedded=next(x for x in transport["items"] if x["kind"]=="ConfigMap" and x["metadata"]["name"]=="argocd-git-transport")["data"]["git"]
assert embedded==(ROOT/"src/infrastructure/argocd/git-transport.sh").read_text(), "transport source and ConfigMap differ"

# Both control layers must enforce the same automatic-night deadline.
resources=json.loads((ROOT/'clusters/lab/rtl-system/resources.json').read_text())['items']
night=next(x['spec'] for x in resources if x['kind']=='InferenceSchedule')
router=next(x for x in resources if x['kind']=='Deployment' and x['metadata']['name']=='glm-router')
env={x['name']:x.get('value') for x in router['spec']['template']['spec']['containers'][0]['env']}
for spec_key,env_key in [('forceTrainingAt','FORCE_TRAINING_AT'),('trainingStart','BACKGROUND_NIGHT_START'),('trainingStop','BACKGROUND_NIGHT_END'),('timezone','SCHEDULE_TIMEZONE')]:
 assert night[spec_key]==env[env_key], ('night_policy_mismatch',spec_key)
