"""Actual checkpoint architecture, NF4 base and language-only LoRA."""
import json,pathlib
from state import REVISION

def load(model_path,rank,resume=None):
 import torch,bitsandbytes as bnb
 from transformers import AutoConfig,AutoModelForImageTextToText,BitsAndBytesConfig
 from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
 config=AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
 if config.model_type!='qwen3_5' or config.architectures!=['Qwen3_5ForConditionalGeneration']:raise ValueError('unexpected checkpoint architecture')
 from provenance import verify_model
 verify=verify_model(model_path)
 if verify.get('revision')!=REVISION or verify.get('repo')!='Qwen/Qwen3.8-27B':raise ValueError('model revision mismatch')
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
 if not getattr(model,'is_loaded_in_4bit',False):raise RuntimeError('NF4 loading was not applied')
 original={n:p.dtype for n,p in model.named_parameters() if p.__class__.__name__!='Params4bit'}
 model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
 # PEFT upcasts every non-4bit tensor. Restore hybrid conv and embeddings to their
 # loaded dtype; keep normalization in FP32. Never cast packed Params4bit storage.
 for name,p in model.named_parameters():
  if name in original and 'norm' not in name and p.dtype!=original[name]:p.data=p.data.to(original[name])
 model.config.use_cache=False
 if hasattr(model.config,'text_config'):model.config.text_config.use_cache=False
 targets=[n for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit) and 'visual' not in n and '.language_model.' in n and n.rsplit('.',1)[-1] in ['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj']]
 if not targets:raise RuntimeError('no quantized language LoRA targets')
 model=PeftModel.from_pretrained(model,resume,is_trainable=True) if resume else get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=0.0,target_modules=targets,bias='none'))
 trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 if not trainable or any('lora_' not in n or 'visual' in n for n,p in trainable):raise RuntimeError('unexpected trainable base/vision parameters')
 torch.cuda.empty_cache()
 return model,{'architecture':type(model.base_model.model).__name__,'target_modules':targets,'trainable_parameters':sum(p.numel() for n,p in trainable),'quantized_modules':sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()),'bitsandbytes':bnb.__version__}
