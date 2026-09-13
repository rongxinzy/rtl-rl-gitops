"""Fixed short greedy cache comparison after the independent reload check."""
import json
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, '/acceptance/qlora-recipe-v1')
sys.argv = ['verify_reload.py', '--model', '/model', '--output', '/acceptance/cuda-512']
loaded = runpy.run_path('/acceptance/qlora-recipe-v1/verify_reload.py')
torch, model = loaded['torch'], loaded['model']
ids = loaded['probe']['input_ids'].cuda()
results = {}
for cache in (False, True):
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                max_new_tokens=8, do_sample=False, use_cache=cache)
    torch.cuda.synchronize()
    results[str(cache)] = {'seconds': time.monotonic()-start,
                          'token_ids': output[0, ids.shape[1]:].cpu().tolist()}
report = {'same_tokens': results['False']['token_ids'] == results['True']['token_ids'],
          'runs': results, 'scope': 'single fixed short greedy probe; not full evaluation equivalence'}
Path('/acceptance/cache-probe.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
