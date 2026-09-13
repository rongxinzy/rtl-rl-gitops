#!/usr/bin/env python3
"""Pinned LLaMA-Factory SFT entrypoint with native HF resume and worker artifacts."""
import argparse,json,math,os,pathlib,shutil,signal,time
from state import REVISION,digest,canonical,bind,verify_checkpoint,publish_native,recover_latest
LF_COMMIT='100e9a42c6c09f8f7849b70d60f3da445fb2024b'
STOP=False

def stop(signum,frame):
 global STOP
 STOP=True

def prepare_rows(tokenizer,path,limit,template=None):
 rows=[];families=set();ids=set();dropped=0
 for line in pathlib.Path(path).read_text().splitlines():
  if not line.strip():continue
  record=json.loads(line)
  if record.get('split')!='train':raise ValueError('only training split admitted')
  if record.get('validation_level')=='K1-grounded':
   from knowledge_data import verify
   verify(record)
  elif record.get('validation_level')!='Q2':raise ValueError('only Q2 or K1-grounded admitted')
  identity=record.get('task_id');family=record.get('family_id')
  if not isinstance(identity,str) or not identity or not isinstance(family,str) or not family or identity in ids:raise ValueError('missing or duplicate identity')
  ids.add(identity)
  messages=record.get('messages',[])
  if not isinstance(messages,list) or len(messages)<2:raise ValueError('conversation required')
  for i,m in enumerate(messages):
   if not isinstance(m,dict) or set(m)!={'role','content'} or not isinstance(m['content'],str) or not m['content'].strip():raise ValueError('text-only messages required')
   expected='system' if i==0 and m['role']=='system' else ('user' if (i-(messages[0]['role']=='system'))%2==0 else 'assistant')
   if m['role']!=expected:raise ValueError('alternating user/assistant roles required')
  if messages[-1]['role']!='assistant':raise ValueError('assistant target required')
  # Official tokenizer only measures admission length; LF performs actual labels/template.
  token_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
  lf_length=0
  if template is not None:
   system=messages[0]['content'] if messages[0]['role']=='system' else None
   conversation=messages[1:] if system is not None else messages
   lf_length=sum(len(x)+len(y) for x,y in template.encode_multiturn(tokenizer,conversation,system,None))+int(template.efficient_eos)
  if max(len(token_ids),lf_length)>limit:dropped+=1;continue
  rows.append({'messages':messages});families.add(family)
 if len(rows)<5 or len(families)<2:raise ValueError('need >=5 examples and >=2 families')
 return rows,{'examples':len(rows),'families':sorted(families),'dropped_overlength':dropped}

def write_dataset(folder,rows):
 folder.mkdir(parents=True,exist_ok=True)
 (folder/'train.json').write_text(canonical(rows))
 (folder/'dataset_info.json').write_text(canonical({'rtl_train':{'file_name':'train.json','formatting':'sharegpt','columns':{'messages':'messages'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}}))

def config(a,out):
 return dict(model_name_or_path=a.model,trust_remote_code=False,stage='sft',do_train=True,finetuning_type='lora',template='qwen3_5_nothink',enable_thinking=False,dataset='rtl_train',dataset_dir=str(out/'dataset'),output_dir=str(out/'native'),overwrite_output_dir=True,cutoff_len=a.max_length,packing=False,train_on_prompt=False,mask_history=True,val_size=0.0,per_device_train_batch_size=1,gradient_accumulation_steps=1,max_steps=a.max_steps,learning_rate=5e-5,lr_scheduler_type='constant',warmup_steps=0,optim='adamw_torch',weight_decay=0.01,max_grad_norm=1.0,lora_rank=8,lora_alpha=16,lora_dropout=0.0,lora_target='q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj',freeze_vision_tower=True,freeze_multi_modal_projector=True,bf16=True,fp16=False,flash_attn='sdpa',quantization_bit=4,quantization_method='bnb',quantization_type='nf4',double_quantization=True,gradient_checkpointing=True,logging_steps=1,save_strategy='steps',save_steps=1,save_total_limit=2,save_only_model=False,save_safetensors=True,report_to='none',disable_tqdm=True,seed=42,data_seed=42,dataloader_num_workers=0,preprocessing_num_workers=1,plot_loss=False)

def reconcile_metrics(out,resume_step):
 path=out/'metrics.jsonl'
 if not path.exists():return
 kept=[];seen=set()
 for line in path.read_text().splitlines():
  try:row=json.loads(line)
  except json.JSONDecodeError:continue  # partial final write after a crash
  step=row.get('step')
  if type(step) is not int or step<=0:raise ValueError('invalid metric step')
  if step>resume_step:continue
  if step in seen:raise ValueError('duplicate committed metric step')
  seen.add(step);kept.append(row)
 pending=out/'metrics.jsonl.tmp';pending.write_text(''.join(canonical(row)+'\n' for row in kept));pending.replace(path)

def finalize(out,checkpoint,job_sha,max_steps,resume_step=0,peak_allocated_bytes=0,elapsed_seconds=0):
 """Materialize worker outputs solely from an integrity-verified checkpoint."""
 out=pathlib.Path(out);checkpoint=pathlib.Path(checkpoint)
 if checkpoint.resolve().parent!=out.resolve():raise ValueError('checkpoint path escape')
 manifest=verify_checkpoint(checkpoint,job_sha)
 if manifest.get('backend')!='llamafactory':raise ValueError('native checkpoint required')
 step=manifest['step'];adapter=out/'adapter';adapter.mkdir(exist_ok=True)
 for name in ('adapter_config.json','adapter_model.safetensors'):
  pending=adapter/(name+'.tmp');shutil.copyfile(checkpoint/name,pending);pending.replace(adapter/name)
 (out/'trainer_state_final.json').write_text(canonical({'global_step':step,'job_sha256':job_sha,'backend':'llamafactory'}))
 (out/'training_metrics.json').write_text(canonical({'global_step':step,'peak_allocated_bytes':peak_allocated_bytes,'resumed_from_step':resume_step,'elapsed_seconds':elapsed_seconds,'job_sha256':job_sha}))
 status={'status':'complete' if step>=max_steps else 'paused','step':step,'adapter':str(adapter),'checkpoint':str(checkpoint),'job_sha256':job_sha}
 pending=out/'status.json.tmp';pending.write_text(canonical(status));pending.replace(out/'status.json')
 return status

def finalize_completed_resume(out,checkpoint,job_sha,max_steps):
 manifest=verify_checkpoint(checkpoint,job_sha)
 if manifest['step']<max_steps:return False
 # A finished checkpoint needs artifact recovery only, never a Trainer invocation.
 metrics=pathlib.Path(out)/'metrics.jsonl'
 rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
 peak=max((row.get('peak_allocated_bytes',0) for row in rows),default=0)
 elapsed=max((row.get('seconds',0) for row in rows),default=0)
 finalize(out,checkpoint,job_sha,max_steps,resume_step=manifest['step'],peak_allocated_bytes=peak,elapsed_seconds=elapsed)
 return True

def main():
 p=argparse.ArgumentParser()
 for name in ('model','data','output','dataset-id'):p.add_argument('--'+name,required=True)
 p.add_argument('--max-steps',type=int,default=20);p.add_argument('--max-length',type=int,default=1024);p.add_argument('--rank',type=int,default=8);p.add_argument('--stop-after',type=int);p.add_argument('--resume-from');p.add_argument('--stop-file');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 if not 1<=a.max_steps<=20 or a.rank!=8 or a.max_length!=1024:raise ValueError('requires rank8 length1024 and <=20 steps')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 import torch
 from transformers import AutoTokenizer,TrainerCallback
 from provenance import verify_model
 torch.set_num_threads(8)
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
 tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
 from llamafactory.data import get_template_and_fix_tokenizer
 from llamafactory.hparams import DataArguments
 template=get_template_and_fix_tokenizer(tokenizer,DataArguments(template='qwen3_5_nothink',enable_thinking=False))
 rows,report=prepare_rows(tokenizer,a.data,a.max_length,template)
 if a.preflight:
  (out/'data-preflight.json').write_text(canonical(dict(report,data_sha256=digest(a.data),model_weights_verified=False)));return
 provenance=verify_model(a.model)
 model_config=json.loads((pathlib.Path(a.model)/'config.json').read_text())
 if model_config.get('model_type')!='qwen3_5' or model_config.get('architectures')!=['Qwen3_5ForConditionalGeneration']:raise ValueError('unexpected model architecture')
 if provenance.get('derived_nf4'):
  q=model_config.get('quantization_config',{})
  if q.get('bnb_4bit_compute_dtype')!='bfloat16' or q.get('bnb_4bit_quant_type')!='nf4' or not q.get('bnb_4bit_use_double_quant'):raise ValueError('invalid prequantized config')
 marker=pathlib.Path('/opt/llamafactory/COMMIT')
 if not marker.exists() or marker.read_text().strip()!=LF_COMMIT:raise ValueError('LLaMA-Factory source pin mismatch')
 cfg=config(a,out)
 identity={'schema_version':1,'backend':'llamafactory','llamafactory_commit':LF_COMMIT,'model_revision':REVISION,'model_manifest_sha256':digest(pathlib.Path(a.model)/'verification.json'),'dataset_id':a.dataset_id,'data_sha256':digest(a.data),'max_steps':a.max_steps,'max_length':a.max_length,'rank':8,'learning_rate':5e-5,'seed':42,'recipe_sha256':{n:digest(pathlib.Path(__file__).with_name(n)) for n in ['train.py','model.py','state.py','provenance.py','knowledge_data.py']},'training_config':cfg}
 job_sha=bind(out,identity)
 import fcntl
 lock=(out.parent/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 resume_step=0
 # The first checkpoint can be durable even when the worker sees no latest yet.
 # Its immutable job binding permits recovery from that precise crash window.
 if not a.resume_from and not (out/'latest').exists() and any(out.glob('checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9]')):a.resume_from='latest'
 if a.resume_from:
  latest=recover_latest(out,job_sha)
  cp=latest if a.resume_from=='latest' else pathlib.Path(a.resume_from)
  if cp.resolve().parent!=out.resolve():raise ValueError('checkpoint path escape')
  if cp.resolve()!=latest.resolve():raise ValueError('resume must use highest committed native checkpoint')
  manifest=verify_checkpoint(cp,job_sha)
  if manifest.get('backend')!='llamafactory':raise ValueError('native LLaMA-Factory checkpoint required')
  resume_step=manifest['step'];cfg['resume_from_checkpoint']=str(cp)
 elif (out/'latest').exists():raise ValueError('existing checkpoint requires --resume-from')
 reconcile_metrics(out,resume_step)
 if a.resume_from and finalize_completed_resume(out,cp,job_sha,a.max_steps):return
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one CUDA GPU required')
 if torch.cuda.mem_get_info()[0]<40*1024**3:raise RuntimeError('GPU requires >=40GiB free')
 write_dataset(out/'dataset',rows);(out/'data-preflight.json').write_text(canonical(report))
 started=time.monotonic();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 class Bridge(TrainerCallback):
  def on_train_begin(self,args,state,control,model=None,**kwargs):
   trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
   if not getattr(model,'is_loaded_in_4bit',False) or not trainable or any('lora_' not in n or 'visual' in n for n,v in trainable):raise RuntimeError('NF4 language-only LoRA invariant failed')
   (out/'model-report.json').write_text(canonical({'backend':'llamafactory','architecture':type(model).__name__,'trainable_parameters':sum(v.numel() for n,v in trainable),'target_modules':sorted({n.rsplit('.lora_',1)[0] for n,v in trainable}),'quantized_modules':sum(type(m).__name__=='Linear4bit' for m in model.modules())}))
  def on_step_end(self,args,state,control,**kwargs):
   if STOP or pathlib.Path(a.stop_file or out/'pause.request').exists() or (a.stop_after and state.global_step>=a.stop_after):control.should_save=True;control.should_training_stop=True
   return control
  def on_log(self,args,state,control,logs=None,**kwargs):
   if not logs or 'loss' not in logs:return
   loss=float(logs['loss']);norm=float(logs.get('grad_norm',0))
   if not math.isfinite(loss) or not math.isfinite(norm) or norm<=0:raise RuntimeError('invalid loss or gradient')
   event={'step':state.global_step,'loss':loss,'gradient_norm':norm,'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint':f'checkpoint-{state.global_step:06d}'}
   with (out/'metrics.jsonl').open('a') as f:f.write(canonical(event)+'\n');f.flush();os.fsync(f.fileno())
   print(canonical(event),flush=True)
  def on_save(self,args,state,control,**kwargs):
   publish_native(out,pathlib.Path(args.output_dir)/f'checkpoint-{state.global_step}',state.global_step,job_sha)
 from llamafactory.train.tuner import run_exp
 run_exp(args=cfg,callbacks=[Bridge()])
 cp=out/(out/'latest').read_text().strip()
 finalize(out,cp,job_sha,a.max_steps,resume_step,torch.cuda.max_memory_allocated(),time.monotonic()-started)
if __name__=='__main__':main()
