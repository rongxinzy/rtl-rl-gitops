"""Authored, executable knowledge lessons; natural-language claims retain provenance."""
import hashlib,json
VERSION='rtl-knowledge-v1'
WIDTHS=(3,5,7)
FAMILIES=('sign_extension','truncation','complete_mux','enable_register','pipeline','reset_counter')
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def lesson(family,w):
 if family not in FAMILIES or w not in WIDTHS:raise ValueError('outside authored curriculum')
 mask=(1<<w)-1;sequential=family in FAMILIES[3:];trace=[]
 if family=='sign_extension':
  interface=f'module dut(input [{w-1}:0] a, output [{w+1}:0] y);'
  expression='{{2{a['+str(w-1)+']}},a}'
  body='assign y = '+expression+';';bad="assign y = {2'b00,a};"
  declarations=f'reg [{w-1}:0] a; wire [{w+1}:0] y; dut u(.a(a),.y(y));'
  vectors=[{'a':a,'y':a+((3<<w) if a&(1<<(w-1)) else 0)} for a in range(1<<w)]
  rule='符号扩展保留补码数值：新增高位复制原最高位。输出位向量可按有符号数解释；端口本身未声明signed，不能据此假定后续表达式进行有符号运算。'
  stimulus=lambda v:f"a={w}'d{v['a']}; #1;"
  output_width=w+2
 elif family=='truncation':
  interface=f'module dut(input [{w+1}:0] a, output [{w-1}:0] y);'
  body=f'assign y = a[{w-1}:0];';bad=f'assign y = a[{w+1}:2];'
  declarations=f'reg [{w+1}:0] a; wire [{w-1}:0] y; dut u(.a(a),.y(y));'
  vectors=[{'a':a,'y':a&mask} for a in range(1<<(w+2))]
  rule='此切片保留最低若干位，高位丢弃；对无符号输入等价于对2的输出位宽次方取模，既不是饱和运算，也不是右移。'
  stimulus=lambda v:f"a={w+2}'d{v['a']}; #1;"
  output_width=w
 elif family=='complete_mux':
  interface=f'module dut(input sel, input [{w-1}:0] a,b, output reg [{w-1}:0] y);'
  body='always @* begin y = b; if (sel) y = a; end';bad='always @* begin y = b; if (!sel) y = a; end'
  declarations=f'reg sel; reg [{w-1}:0] a,b; wire [{w-1}:0] y; dut u(.sel(sel),.a(a),.b(b),.y(y));'
  vectors=[{'sel':s,'a':a,'b':mask-a,'y':a if s else mask-a} for s in (0,1) for a in range(1<<w)]
  rule='组合过程在所有路径上为输出赋值。这里默认选b，再由sel选择a，没有保持旧值的路径。若省掉默认赋值且没有else，则会要求保持旧值，可能推断锁存器。'
  stimulus=lambda v:f"sel=1'b{v['sel']}; a={w}'d{v['a']}; b={w}'d{v['b']}; #1;"
  output_width=w
 else:
  interface=f'module dut(input clk,rst,en, input [{w-1}:0] d, output reg [{w-1}:0] y);'
  declarations=f'reg clk=0,rst,en; reg [{w-1}:0] d; wire [{w-1}:0] y; dut u(.clk(clk),.rst(rst),.en(en),.d(d),.y(y));'
  if family=='enable_register':
   body="always @(posedge clk) begin if (rst) y <= 0; else if (en) y <= d; end"
   bad=body.replace('else if (en)','else if (!en)')
   rule='时钟上升沿先检查同步复位，否则使能有效才装载输入。使能无效时触发器保持旧值；时钟过程缺少else可以表示保持，不等同于组合过程缺赋值推断锁存器。'
  elif family=='pipeline':
   body=f'reg [{w-1}:0] stage; always @(posedge clk) begin if (rst) begin stage <= 0; y <= 0; end else begin stage <= d; y <= stage; end end'
   bad=body.replace('stage <= d; y <= stage;','stage <= d; y <= d;')
   rule='两个非阻塞赋值都根据该上升沿更新前的状态求右值；y接收旧stage，stage接收当前d。因此d需经过两个寄存器采样，不能把第二个赋值理解成读取刚更新的stage。'
  else:
   body=f"always @(posedge clk) begin if (rst) y <= 0; else if (en) y <= y + {w}'d1; end"
   bad=f"always @(posedge clk) begin if (en) y <= y + {w}'d1; else if (rst) y <= 0; end"
   rule='if复位、else if使能规定复位优先；两者同时为1时输出清零。有限位宽无符号计数溢出后回到零。这里复位是同步的，只有时钟上升沿才更新状态。'
  inputs=[(1,0,0),(0,1,1),(0,1,mask),(0,0,mask-1),(1,1,mask),(0,1,2)]
  inputs += [(0,1,(i*3)&mask) for i in range((1<<w)+1)]
  vectors=[];y=0;stage=0
  for rst,en,d in inputs:
   old=stage
   if rst:y=0;stage=0
   elif family=='pipeline':y=old;stage=d
   elif en:y=d if family=='enable_register' else (y+1)&mask
   vectors.append({'rst':rst,'en':en,'d':d,'y':y})
  stimulus=lambda v:f"rst=1'b{v['rst']}; en=1'b{v['en']}; d={w}'d{v['d']}; #5; clk=1; #1;"
  output_width=w
 reference=interface+'\n'+body+'\nendmodule\n';mutant=interface+'\n'+bad+'\nendmodule\n'
 checks=[]
 for i,v in enumerate(vectors):
  checks.append(stimulus(v)+f" if(y !== {output_width}'d{v['y']}) $fatal(1,\"sample {i}\");"+(' #4; clk=0;' if sequential else ''))
 tb='module tb; '+declarations+'\ninitial begin\n'+'\n'.join(checks)+'\n$display("RTL_JUDGE_PASS"); $finish; end endmodule\n'
 indexes=([0,1,2,3,4,5] if sequential else [0,1,len(vectors)//2-1,len(vectors)//2,len(vectors)-2,len(vectors)-1])
 trace=[vectors[i] for i in indexes]
 question_inputs=[{k:v for k,v in x.items() if k!='y'} for x in trace]
 return {'lesson_id':f'{VERSION}:{family}:w{w}','family':family,'width':w,'rule':rule,'reference':reference,'mutant':mutant,'testbench':tb,'trace_inputs':question_inputs,'trace_outputs':[x['y'] for x in trace],
  'sampling':'输入按表中次序在时钟低电平时施加；每行产生一个上升沿，等待非阻塞更新完成后读取y。首行复位建立初态。' if sequential else '每行独立施加输入，等待组合逻辑稳定，按无符号十进制读取y。',
  'coverage':{'vectors':len(vectors),'kind':'bounded_clocked_trace' if sequential else ('exhaustive_single_input' if family!='complete_mux' else 'selector_and_complementary_data_sweep'),'formal_proof':False},
  'semantic_sha256':sha(canonical({'family':family,'width':w,'vectors':vectors}))}

def messages(item):
 code='```verilog\n'+item['reference']+'```'
 trace=canonical(item['trace_inputs'])
 return [
  ('explain','解释下面电路依赖的RTL规则、输出更新条件和常见误用。\n'+code,item['rule']),
  ('predict','根据下面RTL预测y，返回JSON数组。'+item['sampling']+'\n'+code+'\n输入序列：'+trace,canonical(item['trace_outputs'])),
  ('repair','修复下面RTL，使其满足规则：'+item['rule']+' 保持接口不变，简述问题并给出完整模块。\n```verilog\n'+item['mutant']+'```',item['rule']+'\n'+code)]
