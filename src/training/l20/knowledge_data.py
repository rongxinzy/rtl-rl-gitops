"""Local format gate; source and simulation evidence is verified by the admission service."""
import re
HEX=re.compile(r'^[0-9a-f]{64}$')
def verify(record):
 if record.get('validation_level')!='K1-grounded' or record.get('split')!='train':raise ValueError('knowledge training split required')
 if record.get('kind') not in ('explain','predict','repair'):raise ValueError('unsupported knowledge instruction')
 task_id=record.get('task_id','')
 legacy=re.fullmatch(r'(?:rtl-knowledge-v1|rtl-knowledge-remediation-v1):[a-z_]+:w[357]:(explain|predict|repair)',task_id)
 trace_json=re.fullmatch(r'rtl-knowledge-remediation-v1:[a-z_]+:w[46]:predict',task_id)
 if not (legacy or trace_json):raise ValueError('outside authored knowledge curriculum')
 for name in ('knowledge_evidence_sha256','knowledge_source_sha256','semantic_sha256'):
  if not isinstance(record.get(name),str) or not HEX.fullmatch(record[name]):raise ValueError('knowledge evidence identity missing')
 if not record.get('verification_scope'):raise ValueError('knowledge evidence scope missing')
 return True
