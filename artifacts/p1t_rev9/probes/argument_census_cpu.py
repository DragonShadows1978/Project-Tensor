"""PASS iff all native argument reads/refusals run zero Python conversion methods, both call styles.
Uses freshly built full add-on and core current-source dispatch slice, no launches or GPU work.
Prior art: FIX6 recorder census (refusal with no user callbacks), CPython3.12 C-API and DLPack Array API.
Taken: method recorder. Ours: cross-type and positional/keyword census generated from each actual CallArgs signature.
"""
import ast,json,re,sys
from pathlib import Path
import numpy as np
import _p1t_rev7_core as n
from tc_signal import _tc_signal as c
root=Path(__file__).resolve().parents[3];rows=[];hits=[]
class R:
    def hit(self,s):hits.append(s)
    def __index__(self):self.hit('index');return 1
    def __int__(self):self.hit('int');return 1
    def __float__(self):self.hit('float');return 1.
    def __bool__(self):self.hit('bool');return True
    def __len__(self):self.hit('len');return 2
    def __getitem__(self,k):self.hit('getitem');return 1
    def __iter__(self):self.hit('iter');return iter([1,2])
    def __repr__(self):self.hit('repr');return 'R'
    def __str__(self):self.hit('str');return 'R'
class I(int):
    def __index__(self):hits.append('I.index');return 2
    def __int__(self):hits.append('I.int');return 2
    def __bool__(self):hits.append('I.bool');return True
class F(float):
    def __float__(self):hits.append('F.float');return 2.
class T(tuple):
    def __len__(self):hits.append('T.len');return 2
    def __getitem__(self,i):hits.append('T.getitem');return 1
    def __iter__(self):hits.append('T.iter');return iter([1,2])
class L(list):
    def __len__(self):hits.append('L.len');return 2
    def __getitem__(self,i):hits.append('L.getitem');return 1
    def __iter__(self):hits.append('L.iter');return iter([1,2])
class S(str):
    def __str__(self):hits.append('S.str');return 'x'
    def __eq__(self,o):hits.append('S.eq');return str.__eq__(self,o)
    __hash__=str.__hash__
# Exact published FIX6 base args only (no executing its GPU guard or tests).
src=(root/'tc_signal/tests/test_fix6_findings.py').read_text();tree=ast.parse(src)
base=eval(compile(ast.Expression(next(x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_BINDINGS' for t in x.targets))),'<literal base args>','eval'),{'__builtins__':{}})
signatures={}
for p in ['bindings.cpp','lstm_bindings.cpp','m4_bindings.cpp']:
    txt=(root/'tc_signal/src'/p).read_text()
    for m in re.finditer(r'const CallArgs c\("([^"]+)"(.*?);',txt,re.S):
        signatures[m[1]]=re.findall(r'\{"([^"]+)"',m[2])
assert len(signatures)==28,(len(signatures),signatures) # raw exit is the 29th
signatures['_exit_close_admission']=['first_diag_s']
def check(label,fn,a,k):
    hits.clear()
    try: result=fn(*a,**k);outcome='accepted'
    except (TypeError,ValueError,OverflowError,RuntimeError,BufferError) as e:outcome=type(e).__name__
    assert not hits,(label,hits)
    rows.append({'case':label,'outcome':outcome,'calls':list(hits)})
values=[R(),I(1),F(1.),T((1,2)),L([1,2]),S('text'),np.int64(1),np.float32(1),np.float64(1),np.bool_(False)]
for name,args,typed in base:
    names=signatures[name];assert len(names)==len(args),(name,names,args)
    for i in typed:
        for v in values:
            # Prevent successful close or hook arming in the census process.
            if name in ['_exit_close_admission','_test_inject'] and isinstance(v,(int,float,list,tuple)):
                if name=='_exit_close_admission' and isinstance(v,(int,float)):continue
                if name=='_test_inject' and isinstance(v,(list,tuple)):v=[R()]
            a=list(args);a[i]=v
            for style in ['pos','kw']:
                check(f'{name}:{i}:{type(v).__name__}:{style}',getattr(c,name),a if style=='pos' else [],{} if style=='pos' else dict(zip(names,a)))
for name,names in signatures.items():
    for style in ['extra','unknown','missing']:
        if style=='missing' and not names:continue
        if style=='missing' and name in ['pending_stats','cublas_config','lstm_cublas_config']:continue
        check(name+':arity:'+style,getattr(c,name),[R()]*20 if style=='extra' else [],{'no_such_parameter':R()} if style=='unknown' else {})
# C-level keyword matching of a str subclass bypasses __eq__.
check('str-subclass-key',c.pending_stats,[],{S('reset_high_water'):False})
t=n.zero_cuda_label()
for arg in ['stream','max_version','dl_device','copy']:
    for v in values:
        # Valid int streams >1 need a GPU fence: avoid them entirely.
        if arg=='stream' and isinstance(v,int):v=I(1)
        check('core:'+arg+':'+type(v).__name__,t.__dlpack__,[],{arg:v})
for seq in [T((I(1),0)),L([I(1),0])]:check('core:subclass-maxversion',t.__dlpack__,[],{'max_version':seq})
for seq in [T((I(2),I(0))),L([I(2),I(0)])]:check('core:subclass-device',t.__dlpack__,[],{'dl_device':seq})
check('core:str-subclass-key',t.__dlpack__,[],{S('copy'):False})
for fn in [t.__dlpack__,t.__dlpack_device__,n.empty,n._dlpack_begin_shutdown,n._dlpack_raise_unraisable]:
    check('core:arity:'+fn.__name__,fn,[R()]*20,{})
for shape in [T((I(1),I(2))),L([I(1),I(2)])]:check('core:shape-subclass',n.empty,[shape,S('float32'),S('cpu')],{})
(root/'artifacts/p1t_rev9/argument_census.json').write_text(json.dumps({'cases':rows,'count':len(rows),'python_calls':0},indent=2)+'\n')
print('PASS',len(rows),'argument checks; 29 addon bindings, all typed positions both call styles; zero conversion calls')
