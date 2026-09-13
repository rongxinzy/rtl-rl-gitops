"""Bounded CPU-only NF4 feasibility probe; never quantizes the full checkpoint."""
import json,pathlib,time,resource,traceback,os
import torch,bitsandbytes as bnb
from transformers import AutoConfig,AutoModelForImageTextToText,BitsAndBytesConfig
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file,load_file
start=time.monotonic();torch.set_num_threads(2);out=pathlib.Path('/output');report={'torch':torch.__version__,'bnb':bnb.__version__,'cuda_visible':torch.cuda.is_available(),'matrices':[]}
try:
 config=AutoConfig.from_pretrained('/model',local_files_only=True)
 with init_empty_weights():meta=AutoModelForImageTextToText.from_config(config)
 linear={name+'.weight':tuple(module.weight.shape) for name,module in meta.named_modules() if isinstance(module,torch.nn.Linear)}
 report['meta_linear_count']=len(linear);report['language_linear_count']=sum('.language_model.' in name for name in linear)
 index=json.loads(pathlib.Path('/model/model.safetensors.index.json').read_text())['weight_map']
 chosen=[]
 for suffix in ['layers.0.linear_attn.in_proj_a.weight','layers.0.linear_attn.out_proj.weight']:
  matching=[n for n in linear if n.endswith(suffix) and n in index]
  if matching:chosen.append(matching[0])
 if len(chosen)!=2:raise ValueError('expected two actual language matrices not found')
 for name in chosen:
  with safe_open('/model/'+index[name],framework='pt',device='cpu') as f:weight=f.get_tensor(name)
  before=time.monotonic();packed,state=bnb.functional.quantize_4bit(weight,blocksize=64,compress_statistics=True,quant_type='nf4')
  elapsed=time.monotonic()-before;restored=bnb.functional.dequantize_4bit(packed,state)
  serialized={name:packed,**{name+'.'+k:v for k,v in state.as_dict(packed=True).items()}}
  path=out/('matrix-'+str(len(report['matrices']))+'.safetensors');save_file(serialized,path)
  tensors=load_file(path);stats={k:v for k,v in tensors.items() if k!=name}
  param=bnb.nn.Params4bit.from_prequantized(data=tensors[name],quantized_stats=stats,requires_grad=False,device='cpu')
  second=bnb.functional.dequantize_4bit(param.data,param.quant_state)
  report['matrices'].append({'name':name,'shape':list(weight.shape),'original_bytes':weight.numel()*weight.element_size(),'serialized_bytes':path.stat().st_size,'quantize_seconds':elapsed,'rmse':((restored.float()-weight.float()).square().mean().sqrt()).item(),'relative_rmse':((restored.float()-weight.float()).square().mean()/weight.float().square().mean()).sqrt().item(),'serialization_exact':torch.equal(restored,second)})
  del weight,packed,state,restored,second,param,tensors,serialized
 # Actual HF serializer/loader on a tiny instance of the same conditional-generation class.
 cfg=config.to_dict();t=cfg['text_config'];t.update(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=256,layer_types=['full_attention'],mtp_num_hidden_layers=0)
 t['rope_parameters']={'rope_type':'default','rope_theta':10000.0,'partial_rotary_factor':0.25,'mrope_section':[1,1,0],'mrope_interleaved':True}
 v=cfg['vision_config'];v.update(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64,num_position_embeddings=16)
 tinyconfig=type(config).from_dict(cfg);tiny=AutoModelForImageTextToText.from_config(tinyconfig).to(torch.bfloat16);tiny.save_pretrained(out/'tiny-original');del tiny
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 small=AutoModelForImageTextToText.from_pretrained(out/'tiny-original',quantization_config=quant,device_map={'':'cpu'},local_files_only=True,dtype=torch.bfloat16)
 small.save_pretrained(out/'tiny-nf4');del small
 reloaded=AutoModelForImageTextToText.from_pretrained(out/'tiny-nf4',device_map={'':'cpu'},local_files_only=True)
 report['hf_tiny_roundtrip']={'class':type(reloaded).__name__,'loaded_in_4bit':reloaded.is_loaded_in_4bit,'quantized_modules':sum(isinstance(m,bnb.nn.Linear4bit) for m in reloaded.modules())}
 report['status']='success'
except Exception as exc:
 report.update(status='unsupported_or_unverified',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()[-4000:])
report.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
