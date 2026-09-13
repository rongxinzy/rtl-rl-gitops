"""Isolated VerilogEval simulation worker; no model judging."""
import json,pathlib,re,subprocess

def classify(returncode,output):
 matches=re.findall(r'^Mismatches: (\d+) in (\d+) samples\s*$',output,re.M)
 if 'TIMEOUT' in output:return 'unknown'
 if returncode:return 'fail'
 if len(matches)!=1:return 'unknown'
 errors,samples=map(int,matches[0])
 if samples<=0:return 'unknown'
 return 'pass' if errors==0 else 'fail'

def main():
 stages={}
 for name,cmd in [('compile',['iverilog','-g2012','-s','tb','-o','/work/sim','/input/dut.sv','/input/test.sv','/input/ref.sv']),('simulation',['vvp','/work/sim'])]:
  try:
   run=subprocess.run(cmd,capture_output=True,text=True,timeout=20,cwd='/work');output=run.stdout+run.stderr
   stages[name]={'returncode':run.returncode,'output':output[-32768:]}
   if name=='compile' and run.returncode:print(json.dumps({'classification':'fail','stages':stages}));return
  except subprocess.TimeoutExpired:print(json.dumps({'classification':'unknown','timeout_stage':name,'stages':stages}));return
  except OSError as exc:print(json.dumps({'classification':'infrastructure','error':str(exc)}));return
 print(json.dumps({'classification':classify(run.returncode,output),'stages':stages}))
if __name__=='__main__':main()
