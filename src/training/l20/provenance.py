"""Accept only the pinned official source or a fully verified local NF4 derivative."""
import json,pathlib
from state import digest,REVISION
SOURCE_MANIFEST='320eb4bf224856fbecd328e8443da24f6ced6d29e2e371f7693fffa851c03fbf'
def verify_model(directory):
 directory=pathlib.Path(directory);manifest=directory/'verification.json';data=json.loads(manifest.read_text())
 if data.get('repo')!='Qwen/Qwen3.8-27B' or data.get('revision')!=REVISION:raise ValueError('not pinned official Qwen3.8 source')
 if data.get('derived_nf4'):
  if data.get('source_revision')!=REVISION or data.get('source_manifest_sha256')!=SOURCE_MANIFEST:raise ValueError('derived source pin mismatch')
  if digest(directory/'source-verification.json')!=SOURCE_MANIFEST:raise ValueError('derived original manifest bytes mismatch')
  ready=json.loads((directory/'READY.json').read_text())
  if ready.get('verification_sha256')!=digest(manifest) or data.get('validation',{}).get('load_verified') is not True:raise ValueError('derived checkpoint not completely validated')
  quant=data.get('quantization',{})
  if quant.get('bnb_4bit_quant_type')!='nf4' or quant.get('bnb_4bit_use_double_quant') is not True:raise ValueError('unexpected derivative quantization')
  if quant.get('load_in_4bit') is not True or quant.get('bnb_4bit_compute_dtype')!='bfloat16' or data.get('runtime',{}).get('bitsandbytes')!='0.50.2':raise ValueError('derivative runtime/dtype mismatch')
  source=json.loads((directory/'source-verification.json').read_text())
  if data.get('source_files')!=source['files']:raise ValueError('derived source file manifest mismatch')
 elif digest(manifest)!=SOURCE_MANIFEST:raise ValueError('official manifest hash mismatch')
 for entry in data['files']:
  path=directory/entry['file']
  if path.is_symlink() or path.resolve().parent!=directory.resolve() or path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('model file integrity mismatch: '+entry['file'])
 return data
