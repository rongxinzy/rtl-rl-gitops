"""Run on the control node before stopping L20 backup; read-only, never logs keys."""
import json,subprocess,urllib.request
with urllib.request.urlopen('http://172.18.4.199:30000/health',timeout=5) as response:
 if response.status!=200:raise RuntimeError('pro GLM is unhealthy')
code="""
import json,pathlib,urllib.request
key=pathlib.Path('/secrets/key').read_text().strip()
r=urllib.request.Request('http://127.0.0.1:8000/admin/status',headers={'Authorization':'Bearer '+key})
with urllib.request.urlopen(r,timeout=10) as response:s=json.load(response)
assert s['active_backend']=='primary' and s['converged'] and s['healthy_backends']['primary'] and s['backup_inflight']==0
print(json.dumps({k:s[k] for k in ['active_backend','converged','backup_inflight','replicas']}))
"""
pods=json.loads(subprocess.check_output(['k3s','kubectl','-n','rtl-system','get','pods','-l','app=glm-router','-o','json']))['items']
if len(pods)<2:raise RuntimeError('need redundant routers')
checks=[]
for pod in pods:
 value=subprocess.check_output(['k3s','kubectl','-n','rtl-system','exec',pod['metadata']['name'],'--','python3','-c',code],text=True,timeout=20)
 checks.append(json.loads(value))
print(json.dumps({'pro_health':200,'router_checks':checks,'safe_to_stop_backup':True}))
