"""Local format gate; source and simulation evidence is verified by the admission service."""
import re
HEX=re.compile(r'^[0-9a-f]{64}$')
def verify(record):
 if record.get('validation_level')!='K1-grounded' or record.get('split')!='train':raise ValueError('knowledge training split required')
 if record.get('kind') not in ('explain','predict','repair'):raise ValueError('unsupported knowledge instruction')
 if not re.fullmatch(r'rtl-knowledge-v1:[a-z_]+:w[357]:(explain|predict|repair)',record.get('task_id','')):raise ValueError('outside authored knowledge curriculum')
 for name in ('knowledge_evidence_sha256','knowledge_source_sha256','semantic_sha256'):
  if not isinstance(record.get(name),str) or not HEX.fullmatch(record[name]):raise ValueError('knowledge evidence identity missing')
 if not record.get('verification_scope'):raise ValueError('knowledge evidence scope missing')
 return True
