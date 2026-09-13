"""CPU-only official-source NF4 derivative with source and output content manifests."""
import gc,hashlib,json,pathlib,time,resource,shutil,os
import torch,bitsandbytes as bnb
from transformers import AutoModelForImageTextToText,AutoTokenizer,BitsAndBytesConfig
from safetensors import safe_open
from safetensors.torch import save_file
SOURCE_REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SOURCE_MANIFEST='320eb4bf224856fbecd328e8443da24f6ced6d29e2e371f7693fffa851c03fbf'
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as source:
  for part in iter(lambda:source.read(16*1024*1024),b''):h.update(part)
 return h.hexdigest()
def safejson(value):
 if isinstance(value,set):return sorted(value)
 raise TypeError(type(value).__name__)
def event(stage,**values):print(json.dumps({'stage':stage,'seconds':time.monotonic()-start,**values}),flush=True)
def load_checked(path,**kwargs):
 model,info=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,trust_remote_code=False,device_map={'':'cpu'},dtype=torch.bfloat16,low_cpu_mem_usage=True,output_loading_info=True,**kwargs)
 if info.get('missing_keys') or info.get('error_msgs') or info.get('mismatched_keys'):raise RuntimeError('incomplete model load: '+str(info))
 if any(p.is_meta for p in model.parameters()):raise RuntimeError('unmaterialized model parameters')
 return model,info
start=time.monotonic();torch.set_num_threads(2)
if torch.cuda.is_available():raise RuntimeError('CPU-only converter must not see CUDA')
source=pathlib.Path('/model');output=pathlib.Path('/output');target=output/(SOURCE_REV+'-bnb0502');partial=output/(SOURCE_REV+'-bnb0502.partial')
if target.exists() or partial.exists():raise RuntimeError('conversion destination already exists; do not overwrite evidence')
manifest=json.loads((source/'verification.json').read_text())
if digest(source/'verification.json')!=SOURCE_MANIFEST or manifest['repo']!='Qwen/Qwen3.8-27B' or manifest['revision']!=SOURCE_REV:raise RuntimeError('official source identity mismatch')
for number,row in enumerate(manifest['files']):
 path=source/row['file']
 if path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise RuntimeError('source hash mismatch: '+row['file'])
 event('source_file_verified',index=number+1,file=row['file'])
partial.mkdir();shutil.copyfile(source/'verification.json',partial/'source-verification.json');event('quantization_start')
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
model,info=load_checked(source,quantization_config=quant)
quantized=[name for name,module in model.named_modules() if isinstance(module,bnb.nn.Linear4bit)]
if not model.is_loaded_in_4bit or not quantized:raise RuntimeError('NF4 conversion absent')
event('quantization_loaded',quantized_modules=len(quantized),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
expected_shapes={name:list(p.shape) for name,p in model.named_parameters()};expected_keys=sorted(model.state_dict())
model.save_pretrained(partial,max_shard_size='1GB',safe_serialization=True)
AutoTokenizer.from_pretrained(source,local_files_only=True).save_pretrained(partial)
for name in ['preprocessor_config.json','processor_config.json','chat_template.jinja']:
 if (source/name).is_file() and not (partial/name).exists():shutil.copyfile(source/name,partial/name)
del model;gc.collect();event('saved_reloading')
source_index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map'];mtp_keys=sorted(k for k in source_index if k.startswith('mtp.'))
if len(mtp_keys)!=15:raise RuntimeError('unexpected optional MTP layout')
mtp={}
for filename in sorted({source_index[k] for k in mtp_keys}):
 with safe_open(source/filename,framework='pt') as f:
  for key in mtp_keys:
   if source_index[key]==filename:mtp[key]=f.get_tensor(key)
mtp_file='model-mtp-preserved.safetensors';save_file(mtp,partial/mtp_file,metadata={'format':'pt'})
saved_index=partial/'model.safetensors.index.json';mapping=json.loads(saved_index.read_text());mapping['metadata']['total_size']+=sum(t.numel()*t.element_size() for t in mtp.values());mapping['weight_map'].update({key:mtp_file for key in mtp_keys});saved_index.write_text(json.dumps(mapping,indent=2));del mtp;gc.collect()
model,reload_info=load_checked(partial)
actual_shapes={name:list(p.shape) for name,p in model.named_parameters()}
if expected_shapes!=actual_shapes or expected_keys!=sorted(model.state_dict()):raise RuntimeError('prequantized reload parameter/state key mismatch')
if [name for name,module in model.named_modules() if isinstance(module,bnb.nn.Linear4bit)]!=quantized:raise RuntimeError('quantized module list changed')
del model;gc.collect()
files=[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(partial.iterdir()) if p.is_file()]
derived={'repo':'Qwen/Qwen3.8-27B','revision':SOURCE_REV,'derived_nf4':True,'derived_id':target.name,'source_revision':SOURCE_REV,'source_manifest_sha256':SOURCE_MANIFEST,'source_files':manifest['files'],'source_config_sha256':digest(source/'config.json'),'quantization':quant.to_dict(),'runtime':{'torch':torch.__version__,'bitsandbytes':bnb.__version__},'files':files,'validation':{'load_verified':True,'cpu_only':True,'quantized_modules':len(quantized),'parameter_keys':len(expected_shapes),'state_keys':len(expected_keys),'preserved_optional_mtp_keys':mtp_keys,'mtp_used_in_forward':False,'source_loading_info':info,'reload_loading_info':reload_info},'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'conversion_script_sha256':digest('/convert.py')}
(partial/'verification.json').write_text(json.dumps(derived,indent=2,default=safejson))
(partial/'READY.json').write_text(json.dumps({'verification_sha256':digest(partial/'verification.json'),'derived_id':target.name}))
partial.rename(target);event('complete',path=str(target),files=len(files),total_bytes=sum(x['bytes'] for x in files),verification_sha256=digest(target/'verification.json'))
