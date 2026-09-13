"""Narrow host-network WireGuard egress: explicit tables, never host default route."""
import concurrent.futures, ipaddress, json, os, pathlib, subprocess, sys, time
PODS='10.42.3.0/24'; NODE='172.18.6.123'; STATE=pathlib.Path('/tmp/egress-state.json')
PRIVATE=['0.0.0.0/8','10.0.0.0/8','100.64.0.0/10','127.0.0.0/8','169.254.0.0/16','172.16.0.0/12','192.168.0.0/16','224.0.0.0/4','240.0.0.0/4']
PROBE_CODES={}
GATEWAYS={'primary':{'iface':'wg-rtl-eg0','server':'172.18.5.188','port':51822,'net':'10.254.199','table':201,'return_table':299,'mark':'0x0e990000'},'backup':{'iface':'wg-rtl-eg1','server':'172.18.4.199','port':51823,'net':'10.254.198','table':202,'return_table':298,'mark':'0x0e980000'}}
def run(*args,check=True):
 return subprocess.run([str(x) for x in args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check,text=True).stdout.strip()
def iprule(pref,*args):
 text=run('ip','rule','show'); wanted=f'{pref}:'
 if not any(x.startswith(wanted) for x in text.splitlines()):run('ip','rule','add','pref',pref,*args)
def rule(table,chain,*args):
 exists=subprocess.run(['iptables','-w','5','-t',table,'-C',chain,*map(str,args)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 if not exists:run('iptables','-w','5','-t',table,'-I',chain,'1',*args)
def setup_link(name,client):
 g=GATEWAYS[name];iface=g['iface'];addr=g['net']+('.2' if client else '.1')
 if subprocess.run(['ip','link','show',iface],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:run('ip','link','add',iface,'type','wireguard')
 run('ip','address','replace',addr+'/30','dev',iface)
 private='/keys/client-private' if client else '/keys/'+name+'-private'
 peer=pathlib.Path('/keys/'+(name+'-public' if client else 'client-public')).read_text().strip()
 args=['wg','set',iface,'private-key',private,'peer',peer,'allowed-ips','0.0.0.0/0' if client else g['net']+'.2/32,'+PODS]
 if client:args += ['endpoint',g['server']+':'+str(g['port']),'persistent-keepalive','25']
 else:args[3:3]=['listen-port',str(g['port'])]
 run(*args);run('ip','link','set',iface,'mtu','1380','up')
 return g

def setup_gateway(name):
 g=setup_link(name,False);iface=g['iface'];health=g['net']+'.2/32';mark=g['mark']+'/0xffff0000'
 default=run('ip','-j','route','show','default');lan=json.loads(default)[0]['dev']
 # Mark only flows that actually entered this tunnel. Return policy leaves Flannel's main route intact.
 for src in [PODS,health]:
  rule('mangle','PREROUTING','-i',iface,'-s',src,'-j','CONNMARK','--set-xmark',mark)
  rule('nat','POSTROUTING','-s',src,'-o',lan,'-m','connmark','--mark',mark,'-j','MASQUERADE')
  rule('filter','FORWARD','-i',iface,'-s',src,'-o',lan,'-j','ACCEPT')
 rule('mangle','FORWARD','-d',health,'-o',iface,'-p','tcp','--tcp-flags','SYN,RST','SYN','-j','TCPMSS','--clamp-mss-to-pmtu')
 rule('mangle','FORWARD','-d',PODS,'-o',iface,'-p','tcp','--tcp-flags','SYN,RST','SYN','-j','TCPMSS','--clamp-mss-to-pmtu')
 rule('mangle','PREROUTING','-i',lan,'-m','connmark','--mark',mark,'-j','MARK','--set-xmark',mark)
 rule('filter','FORWARD','-o',iface,'-m','conntrack','--ctstate','ESTABLISHED,RELATED','-m','connmark','--mark',mark,'-j','ACCEPT')
 rule('filter','INPUT','-p','udp','--dport',g['port'],'!','-s',NODE,'-j','DROP')
 rule('filter','INPUT','-p','udp','--dport',g['port'],'-s',NODE,'-j','ACCEPT')
 run('ip','route','replace',PODS,'dev',iface,'table',g['return_table'])
 iprule(10991,'to',PODS,'fwmark',mark,'lookup',g['return_table'])
 return {'role':name,'lan':lan,'interface':iface,'ready':True}

def table_private(table):
 for subnet in PRIVATE:run('ip','route','replace','throw',subnet,'table',table)

def setup_client():
 for name in GATEWAYS:
  g=setup_link(name,True);table_private(g['table'])
  run('ip','route','replace','default','dev',g['iface'],'table',g['table'])
  iprule(10990+g['table'],'from',g['net']+'.2/32','lookup',g['table'])
  # This ACCEPT terminates NAT before Flannel can mask the original Pod address.
  rule('nat','POSTROUTING','-s',PODS,'-o',g['iface'],'-j','ACCEPT')
  rule('mangle','OUTPUT','-s',g['net']+'.2/32','-o',g['iface'],'-p','tcp','--tcp-flags','SYN,RST','SYN','-j','TCPMSS','--clamp-mss-to-pmtu')
  rule('mangle','FORWARD','-s',PODS,'-o',g['iface'],'-p','tcp','--tcp-flags','SYN,RST','SYN','-j','TCPMSS','--clamp-mss-to-pmtu')

def probe(name):
 g=GATEWAYS[name]
 for url in ['https://www.cloudflare.com/cdn-cgi/trace','https://modelscope.cn']:
  result=subprocess.run(['curl','--head','--noproxy','*','--interface',g['net']+'.2','--fail','--silent','--show-error','--connect-timeout','4','--max-time','8',url],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  PROBE_CODES[name]=result.returncode
  if result.returncode==0:return True
 return False

def activate(selected):
 table_private(199)
 # Permanent lower-priority blackhole makes gateway failure fail closed, with no main-table fallback.
 run('ip','route','replace','blackhole','default','metric','32760','table','199')
 if selected:run('ip','route','replace','default','dev',GATEWAYS[selected]['iface'],'metric','10','table','199')
 else:run('ip','route','del','default','metric','10','table','199',check=False)
 iprule(10900,'from',PODS,'lookup','199')

def save(state):
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(STATE)

def main():
 role=os.environ['ROLE']
 if role!='client':
  while True:
   state=setup_gateway(role);state['updated']=time.time();save(state);time.sleep(15)
 setup_client();good={k:0 for k in GATEWAYS};bad={k:0 for k in GATEWAYS}
 routes=json.loads(run('ip','-j','route','show','table','199',check=False) or '[]')
 selected=next((name for name,g in GATEWAYS.items() if any(r.get('dev')==g['iface'] and r.get('dst')=='default' for r in routes)),None)
 installed=any(x.startswith('10900:') for x in run('ip','rule','show').splitlines());previous=None
 while True:
  cfg=json.loads(pathlib.Path('/config/settings.json').read_text());enabled=cfg.get('routingEnabled',False);preferred=cfg.get('preferredGateway','auto')
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:health=dict(zip(GATEWAYS,pool.map(probe,GATEWAYS)))
  for name,ok in health.items():good[name]=good[name]+1 if ok else 0;bad[name]=0 if ok else bad[name]+1
  if preferred in GATEWAYS:
   selected=preferred if good[preferred]>=2 or (selected==preferred and bad[preferred]<3) else None
  elif selected and bad[selected]<3:
   pass  # Non-preemptive: keep a healthy active gateway to preserve long TCP sessions.
  else:selected=next((n for n in GATEWAYS if good[n]>=2),None)
  if enabled:activate(selected);installed=True
  elif installed:activate(None)
  state={'role':'client','routingEnabled':enabled,'selected':selected,'health':health,'consecutiveGood':good,'consecutiveBad':bad,'policyInstalled':installed,'probeExitCodes':dict(PROBE_CODES),'updated':time.time()};save(state)
  summary=(enabled,selected,tuple(health.items()))
  if summary!=previous:print(json.dumps(state),flush=True);previous=summary
  time.sleep(10)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='ready':
  try:state=json.loads(STATE.read_text());sys.exit(0 if time.time()-state['updated']<60 else 1)
  except Exception:sys.exit(1)
 main()
