#!/usr/bin/env python3
"""Bounded single-GPU QLoRA SFT with exact optimizer/RNG/data-cursor resume."""
import argparse,json,os,pathlib,signal,sys,time,random
from state import REVISION,digest,canonical,bind,verify_checkpoint,save_checkpoint
STOP=False
def stop(signum,frame):
 global STOP
 STOP=True

def prepare_rows(tokenizer,path,limit):
 rows=[];families=set();ids=set();dropped=0
 for line in pathlib.Path(path).read_text().splitlines():
  if not line.strip():continue
  record=json.loads(line)
  if record.get('validation_level')=='K1-grounded':
   from knowledge_data import verify
   verify(record)
  elif record.get('split')!='train' or record.get('validation_level') not in ['Q2','Q3','functional','formal']:raise ValueError('only validated training split admitted')
  identity=record.get('task_id');family=record.get('family_id')
  if not identity or not family or identity in ids:raise ValueError('missing or duplicate task/family identity')
  messages=record['messages']
  if len(messages)<2 or messages[-1]['role']!='assistant':raise ValueError('assistant target required')
  prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  prompt=tokenizer.encode(prefix,add_special_tokens=False);answer=tokenizer.encode(messages[-1]['content'],add_special_tokens=False)+[tokenizer.eos_token_id]
  if len(prompt)+len(answer)>limit:dropped+=1;continue
  if len(answer)<2:raise ValueError('empty target')
  rows.append({'task_id':identity,'input_ids':prompt+answer,'labels':[-100]*len(prompt)+answer});families.add(family);ids.add(identity)
 if len(rows)<5 or len(families)<2:raise ValueError('need >=5 distinct examples from >=2 families after length filtering')
 return rows,{'examples':len(rows),'families':sorted(families),'dropped_overlength':dropped}

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset-id',required=True);p.add_argument('--max-steps',type=int,default=20);p.add_argument('--max-length',type=int,choices=[512,1024],default=512);p.add_argument('--rank',type=int,default=8);p.add_argument('--stop-after',type=int);p.add_argument('--resume-from');p.add_argument('--stop-file');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 if not 1<=a.max_steps<=20 or not 1<=a.rank<=16:raise ValueError('bounded experiment requires <=20 steps and rank<=16')
 import torch
 from transformers import AutoTokenizer
 torch.set_num_threads(8);torch.manual_seed(42)
 out=pathlib.Path(a.output);tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
 rows,report=prepare_rows(tokenizer,a.data,a.max_length)
 if a.preflight:
  out.mkdir(parents=True,exist_ok=True);report.update(data_sha256=digest(a.data),model_weights_verified=False)
  (out/'data-preflight.json').write_text(canonical(report));print(json.dumps({'event':'data_preflight',**report}),flush=True);return
 identity={'schema_version':1,'model_revision':REVISION,'model_manifest_sha256':digest(pathlib.Path(a.model)/'verification.json'),'dataset_id':a.dataset_id,'data_sha256':digest(a.data),'max_steps':a.max_steps,'max_length':a.max_length,'rank':a.rank,'learning_rate':5e-5,'seed':42,'recipe_sha256':{n:digest(pathlib.Path(__file__).with_name(n)) for n in ['train.py','model.py','state.py','provenance.py','knowledge_data.py']}}
 job_sha=bind(out,identity);(out/'data-preflight.json').write_text(canonical(report));print(json.dumps({'event':'preflight',**report,'job_sha256':job_sha}),flush=True)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 if torch.cuda.device_count()!=1:raise RuntimeError('expose exactly one GPU to this worker')
 import fcntl
 lock=(out.parent/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 free,total=torch.cuda.mem_get_info()
 if free<40*1024**3:raise RuntimeError('GPU not idle with >=40GiB free')
 checkpoint=None;state=None;step=0;cursor=0;order=list(range(len(rows)));random.Random(42).shuffle(order)
 if a.resume_from:
  checkpoint=out/(out/'latest').read_text().strip() if a.resume_from=='latest' else pathlib.Path(a.resume_from)
  if checkpoint.resolve().parent!=out.resolve():raise ValueError('checkpoint path escape')
  verify_checkpoint(checkpoint,job_sha)
  state=torch.load(checkpoint/'training-state.pt',map_location='cpu',weights_only=True)
  if state['job_sha256']!=job_sha:raise ValueError('optimizer identity mismatch')
  step=state['step'];cursor=state['cursor'];order=state['order']
  if sorted(order)!=list(range(len(rows))) or not 0<=cursor<len(rows):raise ValueError('invalid restored data cursor')
 elif (out/'latest').exists():raise ValueError('existing checkpoint requires --resume-from')
 from model import load
 model,model_report=load(a.model,a.rank,checkpoint);model.train();params=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=5e-5)
 if state:
  optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
 (out/'model-report.json').write_text(canonical(model_report))
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 print(json.dumps({'event':'started','resume_step':step,'cursor':cursor,**model_report}),flush=True)
 started=time.monotonic()
 while step<a.max_steps and not STOP:
  row=rows[order[cursor]];inputs={k:torch.tensor([row[k]],device='cuda') for k in ['input_ids','labels']}
  inputs['attention_mask']=torch.ones_like(inputs['input_ids']);optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**inputs,use_cache=False).loss
  if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.0)
  if not torch.isfinite(norm) or norm.item()<=0:raise RuntimeError('missing finite nonzero LoRA gradient')
  optimizer.step();step+=1;cursor=(cursor+1)%len(rows)
  checkpoint=save_checkpoint(out,step,model,optimizer,job_sha,torch,order,cursor)
  event={'step':step,'loss':loss.item(),'gradient_norm':norm.item(),'task_id':row['task_id'],'checkpoint':checkpoint.name,'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(event),flush=True)
  if pathlib.Path(a.stop_file or out/'pause.request').exists() or (a.stop_after and step>=a.stop_after):break
 # The final adapter is separate from resumable checkpoints; optimizer always retained.
 adapter=out/'adapter';model.save_pretrained(adapter,safe_serialization=True)
 probe=rows[0];ids=torch.tensor([probe['input_ids']],device='cuda');model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=ids,use_cache=False).logits[0,-1].float().cpu()
 torch.save({'input_ids':ids.cpu(),'logits':logits,'step':step,'job_sha256':job_sha},out/'reload-probe.pt')
 (out/'trainer_state_final.json').write_text(canonical({'global_step':step,'job_sha256':job_sha,'cursor':cursor}))
 (out/'training_metrics.json').write_text(canonical({'global_step':step,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'resumed_from_step':state['step'] if state else 0,'elapsed_seconds':time.monotonic()-started,'job_sha256':job_sha}))
 (out/'status.json').write_text(canonical({'status':'complete' if step>=a.max_steps else 'paused','step':step,'adapter':str(adapter),'checkpoint':str(checkpoint),'job_sha256':job_sha}))
if __name__=='__main__':main()
