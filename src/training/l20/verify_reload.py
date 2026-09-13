"""Fresh-process NF4 + adapter reload, comparing saved fixed-token logits."""
import argparse,json,pathlib
from state import digest
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
import torch
from model import load
out=pathlib.Path(a.output);identity=json.loads((out/'job.json').read_text());probe=torch.load(out/'reload-probe.pt',map_location='cpu',weights_only=True)
if probe['job_sha256']!=digest(out/'job.json'):raise ValueError('probe/job mismatch')
model,report=load(a.model,identity['rank'],out/'adapter');model.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(input_ids=probe['input_ids'].cuda(),use_cache=False).logits[0,-1].float().cpu()
delta=(actual-probe['logits']).abs().max().item();passed=torch.allclose(actual,probe['logits'],atol=0.01,rtol=0.001)
result={'passed':passed,'max_absolute_logit_delta':delta,'step':probe['step'],'job_sha256':probe['job_sha256'],'tolerance':{'atol':0.01,'rtol':0.001},'quantization':{'format':'nf4','double_quant':True,'compute_dtype':'bfloat16','bitsandbytes':report['bitsandbytes'],'torch':torch.__version__},'model_manifest_sha256':identity['model_manifest_sha256']}
(out/'reload-verification.json').write_text(json.dumps(result));print(json.dumps(result))
if not passed:raise SystemExit(2)
