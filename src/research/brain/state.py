"""Durable state, a single writer lock, append-only decisions and typed artifacts."""
import fcntl, hashlib, json, os, time
from pathlib import Path

def atomic(path,value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f: json.dump(value,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)

class State:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'loop.lock').open('a'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.path=self.root/'progress.json'
        self.value=json.loads(self.path.read_text()) if self.path.exists() else {
            'iteration':0,'actions':{},'stale_count':0,'total_verified_actions':0,'daily':{},'cooldowns':{}}
    def __del__(self):
        if hasattr(self,"lock"):self.lock.close()
    def save(self):
        # Remote content-addressed evidence remains authoritative; bound the local context.
        terminal=[k for k,v in self.value['actions'].items() if v.get('state') in ('completed','failed','interrupted','rejected')]
        for key in terminal[:-500]:
            self.event('action_archived',{'action_id':key,**self.value['actions'].pop(key)})
        atomic(self.path,self.value)
    def event(self,event,detail,level='info'):
        path=self.root/'events.jsonl'
        if path.exists() and path.stat().st_size>=16*1024*1024:
            for i in range(3,0,-1):
                older=self.root/('events.jsonl.'+str(i))
                if older.exists():older.replace(self.root/('events.jsonl.'+str(i+1)))
            path.replace(self.root/'events.jsonl.1')
        with path.open('a') as f:
            f.write(json.dumps({'ts':time.time(),'source':'brain','level':level,'event':event,'detail':detail},ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
    def heartbeat(self,status): atomic(self.root/'heartbeat.json',{'last_seen':time.time(),'status':status,'iteration':self.value['iteration']})

def fingerprint(action,params):
    return hashlib.sha256(json.dumps([action,params],sort_keys=True).encode()).hexdigest()
