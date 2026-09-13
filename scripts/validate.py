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
