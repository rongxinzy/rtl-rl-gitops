"""Authored, executable knowledge lessons; natural-language claims retain provenance."""
import hashlib,json
VERSION='rtl-knowledge-remediation-v1'
WIDTHS=(3,5,7)
# This build is deliberately a new, remedial-only batch. The previous 54-row
# course remains in artifact storage and is not silently replayed.
FAMILIES=('concat_width','mixed_signed_compare','arithmetic_shift','nba_old_state','partial_nba_slice','reset_priority','incomplete_comb_storage','saturating_guard','elastic_stall')
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def lesson(family,w):
 if family not in FAMILIES or w not in WIDTHS:raise ValueError('outside authored curriculum')
 mask=(1<<w)-1;sequential=family in {'nba_old_state','partial_nba_slice','reset_priority','saturating_guard','elastic_stall'};trace=[]
 if family=='concat_width':
  output_fields=('narrow','wide')
  interface=f'module dut(input [{w-1}:0] a,b, output [{w-1}:0] narrow, output [{w}:0] wide);'
  body="assign narrow = {a+b}; assign wide = {1'b0,a}+{1'b0,b};"
  bad="assign narrow = a+b; assign wide = {1'b0,(a+b)};"
  declarations=f'reg [{w-1}:0] a,b; wire [{w-1}:0] narrow; wire [{w}:0] wide; dut u(.a(a),.b(b),.narrow(narrow),.wide(wide));'
  vectors=[{'a':a,'b':b,'narrow':(a+b)&mask,'wide':a+b} for a in range(1<<w) for b in range(1<<w)]
  rule='拼接会确定表达式宽度；单项拼接不会保留加法进位。若需要更宽结果，应先扩展操作数，再执行加法。'
  stimulus=lambda v:f"a={w}'d{v['a']}; b={w}'d{v['b']}; #1;"
  output_width=w
 elif family=='mixed_signed_compare':
  output_fields=('mixed','signed_cmp')
  interface=f'module dut(input [{w-1}:0] a,b, output mixed, output signed_cmp);'
  body='assign mixed = ($signed(a) < b); assign signed_cmp = ($signed(a) < $signed(b));'
  bad='assign mixed = ($signed(a) < b); assign signed_cmp = (a < b);'
  declarations=f'reg [{w-1}:0] a,b; wire mixed,signed_cmp; dut u(.a(a),.b(b),.mixed(mixed),.signed_cmp(signed_cmp));'
  vectors=[]
  for a in range(1<<w):
   for b in range(1<<w):
    sa=a-(1<<w) if a&(1<<(w-1)) else a; sb=b-(1<<w) if b&(1<<(w-1)) else b
    vectors.append({'a':a,'b':b,'mixed':int(a<b),'signed_cmp':int(sa<sb)})
  rule='比較運算的符號性受兩側操作數共同影響；一側為無符號時，另一側可能轉成無符號。需要有符號比較時，兩側都必須明確按有符號值解讀。'
  stimulus=lambda v:f"a={w}'d{v['a']}; b={w}'d{v['b']}; #1;"
  output_width=1
 elif family=='arithmetic_shift':
  output_fields=('logical_y','signed_y')
  interface=f'module dut(input [{w-1}:0] a, output [{w-1}:0] logical_y, signed_y);'
  body='assign logical_y = a >>> 1; assign signed_y = $signed(a) >>> 1;'
  bad='assign logical_y = a >> 1; assign signed_y = a >>> 1;'
  declarations=f'reg [{w-1}:0] a; wire [{w-1}:0] logical_y,signed_y; dut u(.a(a),.logical_y(logical_y),.signed_y(signed_y));'
  vectors=[]
  for a in range(1<<w):
   sign=a&(1<<(w-1)); vectors.append({'a':a,'logical_y':a>>1,'signed_y':((a>>1)|(1<<(w-1))) if sign else a>>1})
  rule='>>>只有左操作數為signed時才做符號填充；unsigned左操作數即使使用>>>仍以零填充。需要算術右移時應明確轉成signed。'
  stimulus=lambda v:f"a={w}'d{v['a']}; #1;"
  output_width=w
 elif family=='incomplete_comb_storage':
  output_fields=('y',)
  interface=f'module dut(input sel, input [{w-1}:0] a,b, output reg [{w-1}:0] y);'
  body='always @* begin if (sel) y = a; else y = b; end'
  bad='always @* begin if (sel) y = a; end'
  declarations=f'reg sel; reg [{w-1}:0] a,b; wire [{w-1}:0] y; dut u(.sel(sel),.a(a),.b(b),.y(y));'
  vectors=[{'sel':s,'a':a,'b':(mask-a),'y':a if s else mask-a} for s in (0,1) for a in range(1<<w)]
  rule='組合always區塊每條控制路徑都必須賦值。缺少else會要求保留先前y，因而推斷鎖存器；用預設值加條件覆寫可表達完整組合選擇。'
  stimulus=lambda v:f"sel=1'b{v['sel']}; a={w}'d{v['a']}; b={w}'d{v['b']}; #1;"
  output_width=w
 elif family in ('nba_old_state','partial_nba_slice','reset_priority','saturating_guard','elastic_stall'):
  if family=='nba_old_state':
   output_fields=('x','y')
   interface=f'module dut(input clk, input [{w-1}:0] a,b, output reg [{w-1}:0] x,y);'
   body='always @(posedge clk) begin x <= y; y <= x; end'
   bad='always @(posedge clk) begin x <= y; y <= x + 1\'b1; end'
   declarations=f'reg clk=0; reg [{w-1}:0] a,b; wire [{w-1}:0] x,y; dut u(.clk(clk),.a(a),.b(b),.x(x),.y(y));'
   inputs=[(2,5),(0,0),(1,3),(mask,2),(4,1)]; x=0;y=0; vectors=[]
   for a,b in inputs: x,y=y,x; vectors.append({'a':a,'b':b,'x':x,'y':y})
   # Use initial values so each trace begins from a defined state.
   body=f'initial begin x={w}\'d0; y={w}\'d0; end\n'+body
   rule='同一时钟沿中的非阻塞赋值都读取更新前的寄存器值；互换赋值实现状态交换，不会让第二句读取第一句刚写入的值。'
   stimulus=lambda v:f"a={w}'d{v['a']}; b={w}'d{v['b']}; #5; clk=1; #1; clk=0; #1;"
   output_width=w
  elif family=='partial_nba_slice':
   output_fields=('q',)
   interface=f'module dut(input clk,lo,hi, input [{w-1}:0] nib, output reg [{2*w-1}:0] q);'
   body=f'always @(posedge clk) begin if(lo) q[{w-1}:0] <= nib; if(hi) q[{2*w-1}:{w}] <= q[{w-1}:0]; end'
   bad=f'always @(posedge clk) begin if(lo) q[{w-1}:0] <= nib; if(hi) q[{2*w-1}:{w}] <= nib; end'
   declarations=f'reg clk=0,lo,hi; reg [{w-1}:0] nib; wire [{2*w-1}:0] q; dut u(.clk(clk),.lo(lo),.hi(hi),.nib(nib),.q(q));'
   q=(1<<(2*w-1))+5; inputs=[(1,1,3),(0,1,1),(1,0,mask),(1,1,2)]; vectors=[]
   for lo,hi,nib in inputs:
    old=q; low=nib if lo else old&mask; high=((old&mask) if hi else old>>w); q=(high<<w)|low; vectors.append({'lo':lo,'hi':hi,'nib':nib,'q':q})
   body=f'initial q={2*w}\'d{(1<<(2*w-1))+5};\n'+body
   rule='同一时钟沿对不同切片的非阻塞赋值仍读取旧状态；高位切片若读取低位切片，应得到该沿开始前的低位，而不是本沿新写入的nib。'
   stimulus=lambda v:f"lo=1'b{v['lo']}; hi=1'b{v['hi']}; nib={w}'d{v['nib']}; #5; clk=1; #1; clk=0; #1;"
   output_width=2*w
  elif family=='reset_priority':
   output_fields=('q',)
   interface=f'module dut(input clk,rst,en, input [{w-1}:0] d, output reg [{w-1}:0] q);'
   body=f'always @(posedge clk) begin if(rst) q<=0; if(en) q<=d; end'
   bad=f'always @(posedge clk) begin if(rst) q<=0; else if(en) q<=d; end'
   declarations=f'reg clk=0,rst,en; reg [{w-1}:0] d; wire [{w-1}:0] q; dut u(.clk(clk),.rst(rst),.en(en),.d(d),.q(q));'
   q=2; inputs=[(1,1,3),(1,0,1),(0,1,mask),(0,0,0)]; vectors=[]
   for rst,en,d in inputs:
    if rst:q=0
    if en:q=d
    vectors.append({'rst':rst,'en':en,'d':d,'q':q})
   body=f'initial q={w}\'d2;\n'+body
   rule='同一过程对同一寄存器安排多个非阻塞赋值时，最后执行且条件成立的赋值生效。独立if中的使能赋值可能覆盖前面的复位；if/else if才明确给出复位优先级。'
   stimulus=lambda v:f"rst=1'b{v['rst']}; en=1'b{v['en']}; d={w}'d{v['d']}; #5; clk=1; #1; clk=0; #1;"
   output_width=w
  elif family=='saturating_guard':
   output_fields=('q',)
   interface=f'module dut(input clk,inc,dec, output reg [{w-1}:0] q);'
   body=f'always @(posedge clk) begin if(dec && q!=0) q<=q-1\'b1; else if(inc && q!={w}\'d{mask}) q<=q+1\'b1; end'
   bad=f'always @(posedge clk) begin if(inc && q!={w}\'d{mask}) q<=q+1\'b1; else if(dec && q!=0) q<=q-1\'b1; end'
   declarations=f'reg clk=0,inc,dec; wire [{w-1}:0] q; dut u(.clk(clk),.inc(inc),.dec(dec),.q(q));'
   q=0; inputs=[(1,0),(1,1),(0,1),(1,0),(1,1)]; vectors=[]
   for inc,dec in inputs:
    if dec and q!=0:q-=1
    elif inc and q!=mask:q+=1
    vectors.append({'inc':inc,'dec':dec,'q':q})
   body=f'initial q=0;\n'+body
   rule='飽和計數器的分支次序定義同時請求時的優先權；條件不成立會進入else if。到達邊界後仍須按原始guard次序處理另一方向請求。'
   stimulus=lambda v:f"inc=1'b{v['inc']}; dec=1'b{v['dec']}; #5; clk=1; #1; clk=0; #1;"
   output_width=w
  else:
   output_fields=('valid','data')
   interface=f'module dut(input clk,ready,in_valid, input [{w-1}:0] in_data, output reg valid, output reg [{w-1}:0] data);'
   body='always @(posedge clk) if(!valid || ready) begin valid<=in_valid; if(in_valid) data<=in_data; end'
   bad='always @(posedge clk) begin valid<=in_valid; if(in_valid) data<=in_data; end'
   declarations=f'reg clk=0,ready,in_valid; reg [{w-1}:0] in_data; wire valid; wire [{w-1}:0] data; dut u(.clk(clk),.ready(ready),.in_valid(in_valid),.in_data(in_data),.valid(valid),.data(data));'
   valid=1;data=2; inputs=[(0,1,3),(1,1,5),(1,0,1),(0,1,6)];vectors=[]
   for ready,in_valid,in_data in inputs:
    if not valid or ready:
     valid=in_valid
     if in_valid:data=in_data
    vectors.append({'ready':ready,'in_valid':in_valid,'in_data':in_data,'valid':valid,'data':data})
   body=f'initial begin valid=1; data={w}\'d2; end\n'+body
   rule='ready為0且valid為1時，彈性暫存級必須保持valid與data；當級為空或下游接收時才可替換內容。有效負載只能在valid握手下解讀。'
   stimulus=lambda v:f"ready=1'b{v['ready']}; in_valid=1'b{v['in_valid']}; in_data={w}'d{v['in_data']}; #5; clk=1; #1; clk=0; #1;"
   output_width=w
  sequential=True
 else:
  raise ValueError('family has no authored implementation')
 reference=interface+'\n'+body+'\nendmodule\n';mutant=interface+'\n'+bad+'\nendmodule\n'
 signal_widths={field:w for field in output_fields}
 if family=='concat_width':signal_widths['wide']=w+1
 if family=='partial_nba_slice':signal_widths['q']=2*w
 if family=='mixed_signed_compare':signal_widths={'mixed':1,'signed_cmp':1}
 if family=='elastic_stall':signal_widths['valid']=1
 checks=[]
 for i,v in enumerate(vectors):
  checks.append(stimulus(v)+''.join(f" if({field} !== {signal_widths[field]}'d{v[field]}) $fatal(1,\"sample {i} {field}\");" for field in output_fields)+(' #4; clk=0;' if sequential else ''))
 tb='module tb; '+declarations+'\ninitial begin\n'+'\n'.join(checks)+'\n$display("RTL_JUDGE_PASS"); $finish; end endmodule\n'
 indexes=([0,1,2,3] if sequential else [0,1,len(vectors)//2-1,len(vectors)//2,len(vectors)-2,len(vectors)-1])
 trace=[vectors[i] for i in indexes]
 question_inputs=[{k:v for k,v in x.items() if k not in output_fields} for x in trace]
 return {'lesson_id':f'{VERSION}:{family}:w{w}','family':family,'width':w,'rule':rule,'reference':reference,'mutant':mutant,'testbench':tb,'trace_inputs':question_inputs,'trace_outputs':[{k:x[k] for k in output_fields} for x in trace],
  'sampling':'输入按表中次序在时钟低电平时施加；每行产生一个上升沿，等待非阻塞更新完成后读取y。首行复位建立初态。' if sequential else '每行独立施加输入，等待组合逻辑稳定，按无符号十进制读取y。',
  'coverage':{'vectors':len(vectors),'kind':'bounded_clocked_trace' if sequential else ('exhaustive_single_input' if family!='complete_mux' else 'selector_and_complementary_data_sweep'),'formal_proof':False},
  'semantic_sha256':sha(canonical({'family':family,'width':w,'vectors':vectors}))}

def messages(item):
 code='```verilog\n'+item['reference']+'```'
 trace=canonical(item['trace_inputs'])
 return [
  ('explain','解释下面电路依赖的RTL规则、输出更新条件和常见误用。\n'+code,item['rule']),
  ('predict','根据下面RTL预测所有输出信号，返回JSON数组；每个数组元素是一个仅含输出信号名和值的JSON对象。'+item['sampling']+'\n'+code+'\n输入序列：'+trace,canonical(item['trace_outputs'])),
  ('repair','修复下面RTL，使其满足规则：'+item['rule']+' 保持接口不变，简述问题并给出完整模块。\n```verilog\n'+item['mutant']+'```',item['rule']+'\n'+code)]
