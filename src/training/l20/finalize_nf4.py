"""Recover validated saved NF4 output after metadata serialization failure."""
import gc,hashlib,json,pathlib,time,resource
import torch,bitsandbytes as bnb
from transformers import AutoModelForImageTextToText
from safetensors import safe_open
from safetensors.torch import save_file
REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0';PIN='320eb4bf224856fbecd328e8443da24f6ced6d29e2e371f7693fffa851c03fbf'
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safejson(v):
 if isinstance(v,set):return sorted(v)
 raise TypeError(type(v).__name__)
start=time.monotonic();torch.set_num_threads(2)
if torch.cuda.is_available():raise RuntimeError('GPU forbidden')
root=pathlib.Path('/output');source=pathlib.Path('/model');partial=root/(REV+'-bnb0502.partial');target=root/(REV+'-bnb0502')
if target.exists() or digest(partial/'source-verification.json')!=PIN:raise ValueError('source/destination identity mismatch')
original=json.loads((partial/'source-verification.json').read_text());index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map'];shapes={}
for file in sorted(set(index.values())):
 with safe_open(source/file,framework='pt') as f:
  for name in f.keys():shapes[name]=list(f.get_slice(name).get_shape())
# HF intentionally omits optional MTP heads from this inference/training class.
# Preserve their exact official BF16 tensors in the derivative instead of dropping them.
mtp_keys=sorted(name for name in shapes if name.startswith('mtp.'))
if len(mtp_keys)!=15:raise RuntimeError('unexpected optional MTP source layout')
mtp={}
for filename in sorted({index[name] for name in mtp_keys}):
 with safe_open(source/filename,framework='pt') as f:
  for name in mtp_keys:
   if index[name]==filename:mtp[name]=f.get_tensor(name)
mtp_file='model-mtp-preserved.safetensors';save_file(mtp,partial/mtp_file)
output_index=partial/'model.safetensors.index.json';mapping=json.loads(output_index.read_text())
new_bytes=sum(t.numel()*t.element_size() for name,t in mtp.items() if name not in mapping['weight_map'])
mapping['metadata']['total_size']+=new_bytes;mapping['weight_map'].update({name:mtp_file for name in mtp_keys});output_index.write_text(json.dumps(mapping,indent=2));del mtp;gc.collect()
model,info=AutoModelForImageTextToText.from_pretrained(partial,local_files_only=True,trust_remote_code=False,device_map={'':'cpu'},dtype=torch.bfloat16,low_cpu_mem_usage=True,output_loading_info=True)
if any(info.get(k) for k in ['missing_keys','error_msgs','mismatched_keys']):raise RuntimeError('incomplete load')
parameters=dict(model.named_parameters());quantized=[name for name,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)]
for name,p in parameters.items():
 if p.is_meta:raise RuntimeError('meta remains')
 actual=list(p.quant_state.shape) if isinstance(p,bnb.nn.Params4bit) else list(p.shape)
 if name not in shapes or actual!=shapes[name]:raise RuntimeError('source shape mismatch '+name)
if set(shapes)-set(parameters)!=set(mtp_keys) or set(parameters)-set(shapes):raise RuntimeError('source/model parameter key mismatch beyond optional MTP')
validation={'load_verified':True,'cpu_only':True,'quantized_modules':len(quantized),'parameter_keys':len(parameters),'state_keys':len(model.state_dict()),'source_parameter_shapes_verified':True,'preserved_optional_mtp_keys':mtp_keys,'mtp_used_in_forward':False,'reload_loading_info':info}
del model,parameters;gc.collect()
config=json.loads((partial/'config.json').read_text());files=[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(partial.iterdir()) if p.is_file() and p.name not in ['READY.json','verification.json']]
result={'repo':'Qwen/Qwen3.8-27B','revision':REV,'derived_nf4':True,'derived_id':target.name,'source_revision':REV,'source_manifest_sha256':PIN,'source_files':original['files'],'source_config_sha256':digest(source/'config.json'),'quantization':config['quantization_config'],'runtime':{'torch':torch.__version__,'bitsandbytes':bnb.__version__},'files':files,'validation':validation,'finalize_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'conversion_script_sha256':digest('/convert.py'),'finalization_script_sha256':digest('/finalize.py'),'recovery_reason':'first conversion fully loaded and reloaded; loading_info set required JSON normalization'}
(partial/'verification.json').write_text(json.dumps(result,indent=2,default=safejson));value=digest(partial/'verification.json')
(partial/'READY.json').write_text(json.dumps({'verification_sha256':value,'derived_id':target.name}));partial.rename(target)
print(json.dumps({'stage':'complete','verification_sha256':value,'total_bytes':sum(f['bytes'] for f in files),'files':len(files),'validation':validation,'seconds':time.monotonic()-start},default=safejson),flush=True)
