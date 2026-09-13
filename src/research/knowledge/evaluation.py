"""Private closed-book application benchmark; export prompts, never answer keys.

API: freeze() -> private immutable bundle; prompts(bundle) -> model-safe rows;
score(bundle, [{task_id, content}, ...], model_revision=...) -> strict result.
No network/model execution occurs here. Sources are pinned by immutable revision URLs and verified byte hashes.
"""
import copy
import hashlib
import json
import re

VERSION='rtl-closed-book-application-v1'
SOURCES=[
 {'id':'lowrisc','revision':'9c15ff5dce23eef969e00ab3715153967419eadd','url':'https://raw.githubusercontent.com/lowRISC/style-guides/9c15ff5dce23eef969e00ab3715153967419eadd/VerilogCodingStyle.md','sha256':'f16c75f95ddca539c179057f9e6ea732574478d7000d8ddcf7787ecec1c79833'},
 {'id':'yosys_primer','revision':'d0e71cfb7bcafe2b437f3edc1789b69e99ecd55a','url':'https://raw.githubusercontent.com/YosysHQ/yosys/d0e71cfb7bcafe2b437f3edc1789b69e99ecd55a/docs/source/appendix/primer.rst','sha256':'ea359ddb0e793d9d753727aee91ae0f27dd35e2934f710d118ce224baf13eda1'},
 {'id':'yosys_binary','revision':'d0e71cfb7bcafe2b437f3edc1789b69e99ecd55a','url':'https://raw.githubusercontent.com/YosysHQ/yosys/d0e71cfb7bcafe2b437f3edc1789b69e99ecd55a/docs/source/cell/word_binary.rst','sha256':'dc052b7d43731c43371d007c3d781fee12b9b184cb3989c9fedbffd00df740c8'},
]


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def sha(value):return hashlib.sha256((value if isinstance(value,str) else canonical(value)).encode()).hexdigest()
def normalized(value):return ''.join(re.sub(r'/\*.*?\*/|//[^\n]*','',value,flags=re.S).split()).casefold()

def task(name,transfer,question,code,answer,sources):
 return {'task_id':'knowledge-heldout-'+name,'transfer':transfer,'question':question,'code':code,'answer':answer,'source_ids':sources}

# Authored applications of documented language rules, not quoted training examples.
TASKS=[
 task('concat-carry','concept_application',
  'All expressions have settled. Give unsigned decimal values of narrow_concat and explicit_extension. The braces around a+b are a one-item concatenation.',
  "logic [3:0] a=4'he,b=4'h5; logic [4:0] narrow_concat,explicit_extension;\nassign narrow_concat={a+b};\nassign explicit_extension={1'b0,a}+{1'b0,b};",
  {'narrow_concat':3,'explicit_extension':19},['lowrisc','yosys_binary']),
 task('mixed-comparison','concept_application',
  'Give Boolean values of mixed_less and signed_less after settling.',
  "logic signed [3:0] s=-4'sd3; logic [3:0] u=4'd2;\nassign mixed_less=(s<u);\nassign signed_less=($signed(s)<$signed(u));",
  {'mixed_less':False,'signed_less':True},['lowrisc','yosys_binary']),
 task('shift-cast','concept_application',
  'Give logical_pattern and signed_pattern as unsigned decimal interpretations of their eight-bit patterns.',
  "logic [7:0] x=8'hd8; logic [7:0] logical_pattern,signed_pattern;\nassign logical_pattern=x>>>2;\nassign signed_pattern=$signed(x)>>>2;",
  {'logical_pattern':54,'signed_pattern':246},['yosys_binary']),
 task('nba-swap-priority','structural_transfer',
  'Initially a=2,b=5. Three rising edges sample en=[1,0,1]. Return states as three [a,b] pairs after all nonblocking updates at each edge.',
  "logic [3:0] a,b;\nalways @(posedge clk) begin a<=b; b<=a; if(en) b<=b+1'b1; end",
  {'states':[[5,6],[6,5],[5,6]]},['yosys_primer','lowrisc']),
 task('independent-three-stage','structural_transfer',
  'Initially a=1,b=2,c=3. Edges sample [din,ea,eb,ec] rows [[9,1,0,1],[4,0,1,0],[7,1,1,1]]. Return states as [a,b,c] after each edge. Each enable independently controls its stage.',
  'logic [3:0] a,b,c;\nalways @(posedge clk) begin if(ea) a<=din; if(eb) b<=a; if(ec) c<=b; end',
  {'states':[[9,2,2],[9,9,2],[7,9,9]]},['yosys_primer','lowrisc']),
 task('elastic-stall-replace','structural_transfer',
  'Initially valid=1,data=5. Inputs [ready,in_valid,in_data] per edge are [[0,1,9],[1,1,7],[1,0,3],[0,1,2]]. A downstream transfer occurs when pre-edge valid and ready are both 1 and uses pre-edge data. Give transferred payloads and post-edge [valid,data] states. This transfer convention is part of this problem.',
  'logic valid; logic [3:0] data;\nalways @(posedge clk) if(!valid || ready) begin valid<=in_valid; if(in_valid) data<=in_data; end',
  {'payloads':[5,7],'states':[[1,5],[1,7],[0,7],[1,2]]},['yosys_primer']),
 task('reset-overridden','concept_application',
  'Initially q=6. Inputs [rst,en,d,flag] per edge are [[1,1,9,1],[1,1,9,0],[0,1,4,1]]. Give q after each edge and Boolean reset_has_priority indicating whether this code always makes rst win over flag.',
  "logic [3:0] q;\nalways @(posedge clk) begin if(rst) q<=4'd0; else if(en) q<=d; if(flag) q<=q+1'b1; end",
  {'q':[7,0,1],'reset_has_priority':False},['yosys_primer','lowrisc']),
 task('partial-nba-old-slice','structural_transfer',
  'Initially q=8\'ha5. Two rising edges sample [lo,hi,nib] rows [[1,1,3],[0,1,9]]. Give the unsigned decimal q after each edge.',
  'logic [7:0] q;\nalways @(posedge clk) begin if(lo) q[3:0]<=nib; if(hi) q[7:4]<=q[3:0]; end',
  {'q':[83,51]},['yosys_primer']),
 task('missing-branch-storage','concept_application',
  'Apply [sel,a,b]=[1,6,3], let the process settle, then [0,9,1] and settle. Give outputs as [y,z] for both events, and the alphabetically ordered names among y,z that require storage because assignments are missing on some paths.',
  'reg [3:0] y,z;\nalways @* begin if(sel) y=a; z=y^b; end',
  {'outputs':[[6,5],[6,7]],'latched':['y']},['yosys_primer','lowrisc']),
 task('saturation-priority-zero','structural_transfer',
  'Initially count=3. Edges sample [inc,dec] rows [[1,0],[1,1],[1,0],[0,1],[0,1],[0,1],[1,1]]. Give unsigned count after each edge; interpret the exact guard order, including when count is zero.',
  "logic [1:0] count;\nalways @(posedge clk) begin if(dec && count!=0) count<=count-1'b1; else if(inc && count!=3) count<=count+1'b1; end",
  {'count':[3,2,3,2,1,0,1]},['yosys_primer','yosys_binary']),
]


def freeze():
 body={'version':VERSION,'sources':copy.deepcopy(SOURCES),'tasks':copy.deepcopy(TASKS),
       'scope':'Ten authored closed-book semantic applications; no broad capability or improvement claim.',
       'transfer_note':'Concept overlap with teaching is intentional. Structural transfer labels denote new composed code structures, not proof that concepts were unseen.'}
 for row in body['tasks']:
  row['question_sha256']=sha(row['question']);row['code_sha256']=sha(row['code']);row['answer_sha256']=sha(row['answer'])
 body['freeze_id']=sha(body)
 return body


def verify(bundle):
 if canonical(bundle)!=canonical(freeze()):raise ValueError('Unknown or modified frozen knowledge benchmark')
 return True


def shape(value):
 if type(value) is bool:return 'boolean'
 if type(value) is int:return 'integer'
 if isinstance(value,str):return 'string'
 if isinstance(value,list):return ['array',shape(value[0]) if value else 'empty']
 return {key:shape(item) for key,item in value.items()}


def prompts(bundle):
 verify(bundle)
 return [{'task_id':row['task_id'],'question':row['question']+' Return only a JSON object, no explanation or markdown. All stated inputs are stable before each edge; there are no X/Z values.',
          'code':row['code'],'output_schema':shape(row['answer'])} for row in bundle['tasks']]


def parse_answer(text):
 if not isinstance(text,str) or len(text.encode())>16384:raise ValueError('Invalid answer size/type')
 def pairs(items):
  result={}
  for key,value in items:
   if key in result:raise ValueError('Duplicate JSON key')
   result[key]=value
  return result
 def bad_constant(_):raise ValueError('Nonfinite JSON constant')
 value=json.loads(text,object_pairs_hook=pairs,parse_constant=bad_constant)
 if not isinstance(value,dict):raise ValueError('Answer must be a JSON object')
 return value


def score(bundle,responses,model_revision):
 verify(bundle)
 if not isinstance(model_revision,str) or not re.fullmatch('[0-9a-f]{40,64}',model_revision):raise ValueError('Immutable model revision required')
 expected={row['task_id']:row for row in bundle['tasks']}
 if not isinstance(responses,list) or any(not isinstance(row,dict) or set(row)!={'task_id','content'} or not isinstance(row.get('content'),str) for row in responses):raise ValueError('Expected task_id/content rows only')
 ids=[row['task_id'] for row in responses]
 if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('Missing, duplicate, or unknown tasks')
 results=[]
 for row in responses:
  try:answer=parse_answer(row['content']);correct=canonical(answer)==canonical(expected[row['task_id']]['answer'])
  except (ValueError,TypeError,RecursionError):correct=False
  results.append({'task_id':row['task_id'],'passed':correct,'response_sha256':sha(row['content'])})
 result={'freeze_id':bundle['freeze_id'],'model_revision':model_revision,'complete':True,'passed':sum(row['passed'] for row in results),'total':len(results),'results':results,'scope':bundle['scope']}
 result['result_id']=sha(result)
 return result


def reject_training_overlap(bundle,rows=None):
 """Exact normalized code/question/ID guard; not a semantic leakage detector."""
 if rows is None:rows,bundle=bundle,freeze()
 verify(bundle);ids={t['task_id'] for t in bundle['tasks']}
 forbidden={normalized(t[k]) for t in bundle['tasks'] for k in ('code','question')}
 answers={canonical(t['answer']) for t in bundle['tasks']}
 for row in rows:
  if row.get('task_id') in ids:raise ValueError('Held-out task ID in training')
  texts=[v for v in row.values() if isinstance(v,str)]
  texts += [m.get('content','') for m in row.get('messages',[]) if isinstance(m,dict)]
  for text in texts:
   for match in re.finditer(r'\{',text):
    try:
     value,_=json.JSONDecoder().raw_decode(text[match.start():])
     if canonical(value) in answers:raise RuntimeError('answer contamination')
    except (ValueError,TypeError):pass
    except RuntimeError:raise ValueError('Held-out answer in training')
   if any(value in normalized(text) for value in forbidden) or any(value in ''.join(text.split()) for value in answers):raise ValueError('Held-out question/code/answer in training')
 return True
