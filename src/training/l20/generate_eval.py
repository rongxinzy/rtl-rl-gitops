"""Offline frozen-prompt generation, no testbench or references admitted."""
import argparse,json,pathlib,hashlib
from state import digest,REVISION
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--prompts',required=True);p.add_argument('--prompts-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--adapter');p.add_argument('--knowledge-prompts');p.add_argument('--knowledge-prompts-sha256');p.add_argument('--knowledge-output');a=p.parse_args()
if digest(a.prompts)!=a.prompts_sha256:raise ValueError('prompt hash mismatch')
rows=[json.loads(x) for x in pathlib.Path(a.prompts).read_text().splitlines()]
if len(rows)!=3 or len({x['task_id'] for x in rows})!=3 or any(set(x)!={'task_id','spec'} for x in rows):raise ValueError('expected three prompt-only records')
if not all((a.knowledge_prompts,a.knowledge_prompts_sha256,a.knowledge_output)):raise ValueError('knowledge arguments required together')
if digest(a.knowledge_prompts)!=a.knowledge_prompts_sha256:raise ValueError('knowledge prompt hash mismatch')
quiz=[json.loads(x) for x in pathlib.Path(a.knowledge_prompts).read_text().splitlines()]
if len(quiz)!=10 or len({x['task_id'] for x in quiz})!=10 or any(set(x)!={'task_id','question','code','output_schema'} for x in quiz):raise ValueError('prompt-only knowledge schema')
job_path=pathlib.Path(a.prompts).parent/'job.json';job=json.loads(job_path.read_text())
if job['prompts_sha256']!=a.prompts_sha256 or job['knowledge_prompts_sha256']!=a.knowledge_prompts_sha256:raise ValueError('job prompt binding mismatch')
adapter_hash=None
if a.adapter:
 adapter_hash=hashlib.sha256(json.dumps({n:digest(pathlib.Path(a.adapter)/n) for n in ('adapter_config.json','adapter_model.safetensors')},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
binding={'job_sha256':digest(job_path),'model_manifest_sha256':digest(pathlib.Path(a.model)/'verification.json'),'generation_recipe_sha256':digest(__file__),'adapter_sha256':adapter_hash}
if binding['generation_recipe_sha256']!=job['recipe_sha256']['generate_eval.py']:raise ValueError('generation recipe changed')
import torch
from transformers import AutoTokenizer
from model import load
model,report=load(a.model,8,a.adapter);model.eval();tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
with pathlib.Path(a.output).open('x') as out:
 for row in rows:
  prompt=tokenizer.apply_chat_template([{'role':'system','content':'Return only one standalone synthesizable Verilog module matching the specification. Do not write a testbench, system tasks, directives, or hierarchical accesses.'},{'role':'user','content':row['spec']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
  if inputs.input_ids.shape[1]>1024:raise ValueError('prompt too long')
  with torch.no_grad():tokens=model.generate(**inputs,max_new_tokens=512,do_sample=False,pad_token_id=tokenizer.eos_token_id)
  completion=tokens[0,inputs.input_ids.shape[1]:]
  out.write(json.dumps({**binding,'task_id':row['task_id'],'content':tokenizer.decode(completion,skip_special_tokens=True),'finish_reason':'length' if len(completion)>=512 else 'stop','model_revision':REVISION,'prompts_sha256':a.prompts_sha256,'quantization':'bnb-nf4','adapter':a.adapter})+'\n');out.flush()

if a.knowledge_prompts:
 if digest(a.knowledge_prompts)!=a.knowledge_prompts_sha256:raise ValueError('knowledge prompt hash mismatch')
 quiz=[json.loads(x) for x in pathlib.Path(a.knowledge_prompts).read_text().splitlines()]
 if len(quiz)!=10 or len({x['task_id'] for x in quiz})!=10 or any(set(x)!={'task_id','question','code','output_schema'} for x in quiz):raise ValueError('prompt-only knowledge schema')
 with pathlib.Path(a.knowledge_output).open('x') as out:
  for row in quiz:
   content=row['question']+'\n'+row['code']+'\nRequired JSON shape: '+json.dumps(row['output_schema'])
   prompt=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
   if inputs.input_ids.shape[1]>1024:raise ValueError('knowledge prompt length')
   with torch.no_grad():tokens=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tokenizer.eos_token_id)
   completion=tokens[0,inputs.input_ids.shape[1]:]
   out.write(json.dumps({**binding,'task_id':row['task_id'],'content':tokenizer.decode(completion,skip_special_tokens=True),'finish_reason':'length' if len(completion)>=256 else 'stop','model_revision':REVISION,'prompts_sha256':a.knowledge_prompts_sha256,'quantization':'bnb-nf4','adapter':a.adapter})+'\n');out.flush()
