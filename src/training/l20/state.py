"""Immutable job identity and atomically published complete checkpoints."""
import hashlib,json,pathlib,os,re,shutil
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for part in iter(lambda:f.read(16*1024*1024),b''):h.update(part)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def bind(out,identity):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);p=out/'job.json'
 if p.exists():
  if json.loads(p.read_text())!=identity:raise ValueError('immutable job identity mismatch')
 else:
  with p.open('x') as f:f.write(canonical(identity));f.flush();os.fsync(f.fileno())
 return digest(p)
def verify_checkpoint(path,job_sha):
 path=pathlib.Path(path)
 if path.is_symlink():raise ValueError('checkpoint symlink rejected')
 manifest=json.loads((path/'complete.json').read_text())
 if manifest['job_sha256']!=job_sha:raise ValueError('checkpoint job identity mismatch')
 for name,value in manifest['files'].items():
  p=path/name
  if p.is_symlink() or p.resolve().parent!=path.resolve() or digest(p)!=value:raise ValueError('checkpoint integrity failure')
 for name in ['adapter_config.json','adapter_model.safetensors','training-state.pt']:
  if name not in manifest['files']:raise ValueError('incomplete checkpoint')
 return manifest

def prune_checkpoints(out,job_sha):
 complete=[]
 for path in pathlib.Path(out).iterdir():
  if path.is_dir() and not path.is_symlink() and re.fullmatch(r'checkpoint-\d{6}',path.name):
   try:verify_checkpoint(path,job_sha)
   except (ValueError,OSError,KeyError):continue
   complete.append(path)
 for path in sorted(complete)[:-2]:shutil.rmtree(path)

def save_checkpoint(out,step,model,optimizer,job_sha,torch,order,cursor):
 out=pathlib.Path(out);target=out/f'checkpoint-{step:06d}';pending=out/f'.checkpoint-{step:06d}.tmp'
 if target.exists():raise ValueError('checkpoint already published')
 pending.mkdir();model.save_pretrained(pending,safe_serialization=True)
 state={'step':step,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'order':order,'cursor':cursor,'job_sha256':job_sha}
 torch.save(state,pending/'training-state.pt')
 files={p.name:digest(p) for p in pending.iterdir() if p.is_file()}
 (pending/'complete.json').write_text(canonical({'step':step,'job_sha256':job_sha,'files':files}))
 for p in pending.iterdir():
  if p.is_file():
   with p.open('rb') as f:os.fsync(f.fileno())
 pending.rename(target)
 pointer=out/'latest.tmp';pointer.write_text(target.name);pointer.replace(out/'latest')
 prune_checkpoints(out,job_sha)
 return target
