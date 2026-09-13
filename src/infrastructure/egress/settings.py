import json,subprocess,sys,time
settings={'routingEnabled':'--disable' not in sys.argv,'preferredGateway':next((x.split('=',1)[1] for x in sys.argv if x.startswith('--gateway=')),'auto')}
assert settings['preferredGateway'] in ['auto','primary','backup']
subprocess.run(['k3s','kubectl','-n','rtl-egress','patch','configmap','egress-settings','--type','merge','-p',json.dumps({'data':{'settings.json':json.dumps(settings)}})],check=True)
subprocess.run(['k3s','kubectl','-n','rtl-egress','annotate','pod','-l','role=client','rtl.ai/config-refresh='+str(time.time()),'--overwrite'],check=True)
print(json.dumps(settings))
