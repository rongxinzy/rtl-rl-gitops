"""Finite authored unary RTL families; Python truth tables are the judge oracle."""
import hashlib,json
FAMILIES=('bit_reverse','population_count','rotate_left','gray_encode')
HOLDOUT=frozenset({'gray_encode'})
WIDTHS=tuple(range(2,7))
VERSION='authored-unary-v1'
def sha(v):
 return hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def make_task(family,width):
 if family not in FAMILIES or width not in WIDTHS:raise ValueError('unsupported finite family/width')
 w=width;bits=lambda a:[(a>>i)&1 for i in range(w)]
 def oracle(a):
  b=bits(a)
  if family=='bit_reverse':return int(''.join(map(str,b)),2)
  if family=='population_count':return sum(b)
  if family=='rotate_left':return sum(b[i]<<(i+1) for i in range(w-1))+b[-1]
  return sum((b[i] ^ (b[i+1] if i+1<w else 0))<<i for i in range(w))
 reference_expr={
  'bit_reverse':"{"+','.join(f'a[{i}]' for i in range(w))+"}",
  'population_count':'+'.join("{"+f"{w-1}'b0,a[{i}]"+"}" for i in range(w)),
  'rotate_left':f'{{a[{w-2}:0],a[{w-1}]}}',
  'gray_encode':'a ^ (a >> 1)',
 }[family]
 interface=f'module dut(input [{w-1}:0] a, output [{w-1}:0] y);'
 def rtl(expr):return interface+'\nassign y = '+expr+';\nendmodule\n'
 meanings={'bit_reverse':f'Reverse the order of the {w} input bits: output bit i equals input bit {w-1}-i.',
 'population_count':'Output the unsigned number of input bits that equal one; zero-extend that count to the output width.',
 'rotate_left':f'Rotate a left by exactly one bit within {w} bits; the input MSB wraps into output bit 0.',
 'gray_encode':'Convert an unsigned binary input to reflected Gray code: output MSB equals input MSB; every lower output bit i is input bit i XOR input bit i+1.'}
 vectors=[(a,oracle(a)) for a in range(1<<w)]
 lines=[f"a={w}'d{a}; #1; if(y !== {w}'d{y}) $fatal(1,\"vector {a} mismatch\");" for a,y in vectors]
 tb=f'module tb; reg [{w-1}:0] a; wire [{w-1}:0] y; dut u(.a(a),.y(y));\ninitial begin\n'+'\n'.join(lines)+'\n$display("RTL_JUDGE_PASS"); $finish;\nend\nendmodule\n'
 return {'task_id':f'{VERSION}:{family}:w{w}','family_id':family,'width':w,'top':'dut',
 'split':'val' if family in HOLDOUT else 'train','spec':f'Implement a standalone synthesizable combinational unsigned RTL module. Exact interface: {interface}\n{meanings[family]} No clock, state or latches. Drive y for every binary a.',
 'reference':rtl(reference_expr),'mutants':[rtl(f"{w}'d0"),rtl(f'({reference_expr}) ^ {w}\'d1')],
 'testbench':tb,'semantic_sha256':sha(canonical({'width':w,'vectors':vectors})),
 'coverage':{'kind':'exhaustive_binary_inputs','vectors':len(vectors),'unknown_inputs_tested':False,'formal_proof':False}}
