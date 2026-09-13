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
 required=['adapter_config.json','adapter_model.safetensors','training-state.pt']
 if manifest.get('backend')=='llamafactory':
  required=['adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
  if {p.name for p in path.iterdir()} != set(manifest['files'])|{'complete.json'}:raise ValueError('unmanifested checkpoint files')
  trainer=json.loads((path/'trainer_state.json').read_text())
  if type(manifest.get('step')) is not int or trainer.get('global_step')!=manifest['step']:raise ValueError('native trainer step mismatch')
 for name in required:
  if name not in manifest['files']:raise ValueError('incomplete checkpoint')
  if manifest.get('backend')=='llamafactory' and (path/name).stat().st_size==0:raise ValueError('empty native checkpoint file')
 return manifest

def write_latest(out,target):
 out=pathlib.Path(out);pointer=out/'latest.tmp'
 if pointer.is_symlink() or (out/'latest').is_symlink():raise ValueError('checkpoint pointer symlink rejected')
 with pointer.open('w') as f:f.write(target.name);f.flush();os.fsync(f.fileno())
 pointer.replace(out/'latest')
 fd=os.open(out,os.O_RDONLY);os.fsync(fd);os.close(fd)

def recover_latest(out,job_sha):
 """Commit a verified orphan after rename-before-pointer crash; never replay it."""
 out=pathlib.Path(out);complete=[]
 for path in out.iterdir():
  if not re.fullmatch(r'checkpoint-\d{6}',path.name):continue
  manifest=verify_checkpoint(path,job_sha)
  if manifest.get('backend')!='llamafactory' or path.name!=f"checkpoint-{manifest['step']:06d}":raise ValueError('native checkpoint identity mismatch')
  complete.append(path)
 if not complete:raise ValueError('no complete native checkpoint to resume')
 latest=out/'latest'
 if latest.exists():
  if latest.is_symlink() or not re.fullmatch(r'checkpoint-\d{6}',latest.read_text().strip()):raise ValueError('invalid checkpoint pointer')
  if out/latest.read_text().strip() not in complete:raise ValueError('checkpoint pointer target missing')
 target=max(complete,key=lambda p:int(p.name.split('-')[1]));write_latest(out,target)
 return target

def publish_native(out,source,step,job_sha):
 out=pathlib.Path(out);source=pathlib.Path(source)
 target=out/f'checkpoint-{step:06d}';pending=out/f'.checkpoint-{step:06d}.tmp'
 if target.exists():
  verify_checkpoint(target,job_sha)
  raise ValueError('checkpoint already published')
 if pending.exists():shutil.rmtree(pending)
 pending.mkdir()
 for p in source.iterdir():
  if not p.is_file() or p.is_symlink():raise ValueError('native checkpoint must contain regular files only')
  shutil.copyfile(p,pending/p.name)
 files={p.name:digest(p) for p in pending.iterdir()}
 (pending/'complete.json').write_text(canonical({'backend':'llamafactory','step':step,'job_sha256':job_sha,'files':files}))
 verify_checkpoint(pending,job_sha)
 for p in pending.iterdir():
  with p.open('rb') as f:os.fsync(f.fileno())
 pending.rename(target)
 write_latest(out,target)
 prune_checkpoints(out,job_sha)
 return target

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
