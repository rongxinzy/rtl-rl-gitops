"""Bounded bitvector expressions: independently evaluated and rendered as RTL."""
from .families import canonical, sha, make_task, HOLDOUT, WIDTHS
VERSION='bitvector-dsl-v1'
BINARY={'and','or','xor','add','sub'}
COMMUTATIVE={'and','or','xor','add'}
def normalize(payload):
 if not isinstance(payload,dict) or set(payload)!={'input_widths','output_width','expression'}:raise ValueError('invalid proposal fields')
 widths=payload['input_widths'];w=payload['output_width']
 integer=lambda x:type(x) is int
 if not isinstance(widths,list) or not 1<=len(widths)<=4 or not all(integer(x) and 1<=x<=10 for x in widths) or sum(widths)>10:raise ValueError('input bit budget')
 if not integer(w) or not 1<=w<=10:raise ValueError('output width')
 count=0
 def visit(n,depth):
  nonlocal count
  count+=1
  if depth>5 or count>31 or not isinstance(n,dict):raise ValueError('AST complexity')
  op=n.get('op');fields={'input':{'index'},'const':{'value'},'not':{'arg'},'mux':{'cond','yes','no'},'shl':{'arg','amount'},'shr':{'arg','amount'}}
  keys={'left','right'} if op in BINARY else fields.get(op)
  if keys is None or set(n)!={'op'}|keys:raise ValueError('invalid AST node')
  out={'op':op}
  for k in sorted(keys):
   v=n[k]
   if k in ('index','value','amount'):
    bound=len(widths)-1 if k=='index' else (9 if k=='amount' else (1<<w)-1)
    if not integer(v) or not 0<=v<=bound:raise ValueError('invalid AST scalar')
    out[k]=v
   else:out[k]=visit(v,depth+1)
  if op in COMMUTATIVE and canonical(out['left'])>canonical(out['right']):out['left'],out['right']=out['right'],out['left']
  return out
 return {'input_widths':widths[:],'output_width':w,'expression':visit(payload['expression'],1)}

def evaluate(node,values,width):
 mask=(1<<width)-1;op=node['op']
 ev=lambda n:evaluate(n,values,width)
 if op=='input':value=values[node['index']]
 elif op=='const':value=node['value']
 elif op=='not':value=~ev(node['arg'])
 elif op=='mux':value=ev(node['yes']) if ev(node['cond']) else ev(node['no'])
 elif op=='shl':value=ev(node['arg'])<<node['amount']
 elif op=='shr':value=ev(node['arg'])>>node['amount']
 else:
  a,b=ev(node['left']),ev(node['right'])
  value={'and':lambda:a&b,'or':lambda:a|b,'xor':lambda:a^b,'add':lambda:a+b,'sub':lambda:a-b}[op]()
 return value&mask

def render(p):
 w=p['output_width'];lines=[]
 def emit(n):
  op=n['op']
  if op=='input':expr=f"i{n['index']}"
  elif op=='const':expr=f"{w}'d{n['value']}"
  elif op=='not':expr='~'+emit(n['arg'])
  elif op=='mux':expr=f"({emit(n['cond'])} != {w}'d0) ? {emit(n['yes'])} : {emit(n['no'])}"
  elif op in ('shl','shr'):expr=f"{emit(n['arg'])} {'<<' if op=='shl' else '>>'} {n['amount']}"
  else:expr=f"{emit(n['left'])} "+{'and':'&','or':'|','xor':'^','add':'+','sub':'-'}[op]+f" {emit(n['right'])}"
  name=f'n{len(lines)}';lines.append(f'wire [{w-1}:0] {name}; assign {name} = {expr};');return name
 result=emit(p['expression'])
 interface='module dut('+', '.join(f'input [{v-1}:0] i{i}' for i,v in enumerate(p['input_widths']))+f', output [{w-1}:0] y);'
 return interface+'\n'+'\n'.join(lines)+f'\nassign y = {result};\nendmodule\n'

def make_dsl(payload):
 p=normalize(payload);w=p['output_width'];bits=sum(p['input_widths']);vectors=[];tb=[]
 for packed in range(1<<bits):
  values=[];offset=0
  for size in p['input_widths']:values.append((packed>>offset)&((1<<size)-1));offset+=size
  y=evaluate(p['expression'],values,w);vectors.append((packed,y))
  tb.append(' '.join(f"i{i}={size}'d{a};" for i,(size,a) in enumerate(zip(p['input_widths'],values)))+f' #1; if(y !== {w}\'d{y}) $fatal(1,"vector {packed} mismatch");')
 # Match the legacy unary digest even when inputs are partitioned or renamed.
 semantic=sha(canonical({'width':w,'vectors':vectors})) if bits==w else sha(canonical({'input_bits':bits,'output_width':w,'vectors':vectors}))
 for offset in range(max(0,bits-w+1)):
  if all(y==(((a>>offset)&((1<<w)-1)) ^ (((a>>offset)&((1<<w)-1))>>1)) for a,y in vectors):raise ValueError('held-out Gray function forbidden')
 for size in WIDTHS:
  for family in HOLDOUT:
   if semantic==make_task(family,size)['semantic_sha256']:raise ValueError('held-out semantic overlap')
 if len({y for _,y in vectors})==1:raise ValueError('constant functions are not useful proposals')
 reference=render(p)
 # XOR bit-flips are guaranteed wrong on every binary vector and compile independently.
 mutants=[reference.replace('assign y = ',f"assign y = {w}'d{v} ^ ") for v in (1,(1<<w)-1 if w>1 else 1)]
 if w==1:mutants[1]=reference.replace('assign y = ','assign y = ~')
 decl=' '.join(f'reg [{size-1}:0] i{i};' for i,size in enumerate(p['input_widths']))
 ports=','.join(f'.i{i}(i{i})' for i in range(len(p['input_widths'])))
 testbench=f'module tb; {decl} wire [{w-1}:0] y; dut u({ports},.y(y));\ninitial begin\n'+'\n'.join(tb)+'\n$display("RTL_JUDGE_PASS"); $finish; end\nendmodule\n'
 spec=('Implement a standalone unsigned combinational module dut with '+', '.join(f"input [{size-1}:0] i{i}" for i,size in enumerate(p['input_widths']))+f' and output [{w-1}:0] y. '
       'No clock, state or latches. Evaluate the following expression AST. Each node, including input and constant nodes, is zero-extended or truncated to the output width; every operation wraps modulo 2**output_width. '
       'not is bitwise complement; and/or/xor are bitwise; add/sub are unsigned modular arithmetic; shl/shr are logical constant shifts; mux selects yes when cond is nonzero, otherwise no. Input index refers to iN. '
       +canonical(p))
 return {'task_id':VERSION+':'+sha(canonical(p)),'family_id':VERSION,'width':w,'top':'dut','split':'train','dsl':p,'spec':spec,'reference':reference,'mutants':mutants,'testbench':testbench,'semantic_sha256':semantic,
         'coverage':{'kind':'exhaustive_binary_inputs','vectors':len(vectors),'unknown_inputs_tested':False,'formal_proof':False}}
