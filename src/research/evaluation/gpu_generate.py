"""Fixed offline Qwen base generation; input contains prompts only, never testbenches."""
import json,time,pathlib,torch
from transformers import AutoTokenizer,AutoModelForImageTextToText
start=time.monotonic();torch.manual_seed(42)
tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained('/model',local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa')
model.eval()
with open('/output/generations.jsonl','x') as output:
 for line in pathlib.Path('/input/prompts.jsonl').read_text().splitlines():
  task=json.loads(line);messages=[{'role':'system','content':'Return only one standalone synthesizable Verilog module matching the specification. Do not write a testbench, system tasks, directives, or hierarchical accesses.'},{'role':'user','content':task['spec']}]
  prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inputs=tokenizer(prompt,return_tensors='pt').to('cuda');n=inputs['input_ids'].shape[1]
  if n>2048:raise ValueError('evaluation prompt over fixed token budget')
  with torch.inference_mode():result=model.generate(**inputs,max_new_tokens=512,do_sample=False,pad_token_id=tokenizer.eos_token_id)
  tokens=result[0,n:];row={'task_id':task['task_id'],'content':tokenizer.decode(tokens,skip_special_tokens=True),'generated_tokens':len(tokens),'finish_reason':'length' if len(tokens)>=512 else 'stop'}
  output.write(json.dumps(row)+'\n');output.flush()
pathlib.Path('/output/generation_metrics.json').write_text(json.dumps({'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'greedy':True,'max_new_tokens':512,'seed':42,'torch':torch.__version__}))
