"""Owned INPUT chain, no global flush. API still requires a strong bearer token."""
import subprocess

def run(args,check=True):return subprocess.run(['iptables','-w','5']+args,check=check,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def add(chain,args):
 if run(['-C',chain]+args,False).returncode:run(['-A',chain]+args)
run(['-N','RTL_L20_API'],False)
for source in ('172.18.4.199/32','172.18.5.188/32','172.18.6.123/32'):
 add('RTL_L20_API',['-s',source,'-j','ACCEPT'])
add('RTL_L20_API',['-j','DROP'])
args=['-d','172.18.6.123','-p','tcp','--dport','18766','-j','RTL_L20_API']
if run(['-C','INPUT']+args,False).returncode:run(['-I','INPUT','1']+args)
