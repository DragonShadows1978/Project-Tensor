"""Drive real post-device handlers on faithful constructed binding outputs.
Prior art: DS1 A3 L48 and C++ Tensor/NDArray export ABI (2026), boundary
analysis (Myers, 1979). Taken: actual tuple indices, optional None slots,
dtype/shape/offset/storage metadata; ours: CPU-owned backing bytes only.
This checks harness readback/scoring, never GPU numerics.
"""
from gpu_common import *
import numpy as np
from types import SimpleNamespace
from worker_core import read_attention,check_layout,finish_arrays
from RUN import compare_bitwise
from schema import fields,payload_bytes
from layout import layout

class Export:
 def __init__(self,a,dt,storage=None,offset=0):self.a=a;self.dt=dt;self.storage=a if storage is None else storage;self.offset=offset
class Info:
 @staticmethod
 def info(t):return (t.a.ctypes.data,t.a.shape,t.dt,t.storage.nbytes,t.storage.ctypes.data,False,False,0,t.offset)
tc=SimpleNamespace(synchronize=lambda:(_ for _ in ()).throw(AssertionError('CPU readback must not synchronize CUDA')))
I=Info()
def pattern(shape,dtype):
 a=np.empty(shape,dtype=dtype)
 # Bounded temporary chunks; every reference field is nonzero and nonconstant.
 flat=a.reshape(-1)
 for i in range(0,flat.size,1000000):flat[i:i+1000000]=np.arange(i+1,min(flat.size,i+1000000)+1,dtype=np.uint64)%127+1
 return a
def standalone(shape,dtype):
 a=pattern(shape,dtype);return Export(a,{'uint8':4,'uint16':5,'uint32':0,'uint64':2}[dtype])
def result_tuple(c,f):
 plan=layout(f,c,True);arena=np.empty(plan['total'],dtype=np.uint8);assert arena.ctypes.data%16==0
 shapes={name:r['shape'] for name,r in fields(c).items()}
 lookup={}
 for slot in plan['views']:
  name=slot['name'];dtype='uint8' if name in ('mask','flags','used') else 'uint32'
  shape={'pq':(c['B'],c['H'],1,128),'mask':(c['B'],c['H'],1,c['S']),'bulk':(c['B'],c['H'],1,c['S']),'prefix':(c['B'],c['H'],1,c['S']),'cm':(c['B'],c['H'],(c['S']+255)//256),'cz':(c['B'],c['H'],(c['S']+255)//256),'flags':(c['B'],c['KV'],(c['S']+63)//64),'used':(c['B'],c['KV'],(c['S']+63)//64),'scores':(c['B'],c['KV'],(c['S']+63)//64),'logz':(c['B'],c['H']),'blog':(c['B'],c['H'],(c['S']+63)//64)}.get(name,(slot['bytes']//4,))
  a=np.ndarray(shape,dtype=dtype,buffer=arena,offset=slot['offset']);a[:]=pattern(shape,dtype)
  lookup[name]=Export(a,4 if dtype=='uint8' else 0,arena,slot['offset']//np.dtype(dtype).itemsize)
 output=standalone((c['B'],c['H'],1,128),'uint16')
 foot=np.array([plan['total'],c['B']*c['H']*128*2]+[0]*(4 if f=='eb3' else 5),dtype=np.uint64);foot=Export(foot,2)
 if f=='eb3':return (output,lookup['pq'],lookup['mask'],lookup['bulk'],lookup['prefix'],None,None,foot)
 return (output,lookup['flags'],lookup['used'],lookup['scores'],lookup['logz'],lookup['cm'],lookup['cz'],lookup['blog'],lookup['mask'],lookup['bulk'],lookup['prefix'],lookup['pq'],foot)

def arrays(c):
 out={k:pattern(r['shape'],r['dtype']) for k,r in fields(c).items()}
 for n in c['append_lengths']:
  if n<c['S']-3:continue
  current=dict(c,S=n)
  r3=result_tuple(current,'eb3');r4=result_tuple(current,'eb4')
  vi=standalone((c['B'],c['KV'],n,128),'uint8');vs=standalone((c['B'],c['KV'],n),'uint32')
  read_attention(tc,I,r3,r4,vi,vs,n,c['S'],out)
  if n==c['S']:check_layout(tc,I,r3,r4,c)
 assert set(out)==set(fields(c))
 for name,spec in fields(c).items():assert list(out[name].shape)==spec['shape'] and str(out[name].dtype)==spec['dtype'],name
 return out

if __name__=='__main__':
 A=ART;cases=load(A/'INPUT_CENSUS.json');chosen=[min(cases,key=payload_bytes),max(cases,key=payload_bytes)];rows=[]
 for c in chosen:
  for scenario in ('equal','one_bit_difference'):
   dst=A/'constructed_readback'/c['id']/scenario
   for arm in ('lab','port'):
    out=dst/arm;out.mkdir(parents=True)
    aa=arrays(c)
    if scenario=='one_bit_difference' and arm=='port':aa[f'step_{c["S"]}_eb3_out'].flat[0]^=np.uint16(1)
    finish_arrays(out,c,aa,c['append_lengths'],dict(constructed=True,device_calls=0))
   got=compare_bitwise(dst,c);want=scenario=='equal'
   rows.append(dict(case=c['id'],shape=[c[x] for x in ('B','H','KV','S')],scenario=scenario,expected_pass=want,observed_pass=got,status='PASS' if got==want else 'FAIL',fields=len(fields(c)),unsigned_payload_bytes_per_arm=payload_bytes(c)))
  # Exercise the absent-field failure that escaped DS1's old readback gate.
  bad=list(result_tuple(c,'eb3'));bad[0]=None
  try:read_attention(tc,I,bad,result_tuple(c,'eb4'),standalone((c['B'],c['KV'],c['S'],128),'uint8'),standalone((c['B'],c['KV'],c['S']),'uint32'),c['S'],c['S'],{})
  except AssertionError:rows.append(dict(case=c['id'],scenario='missing_output_rejected',status='PASS'))
  else:rows.append(dict(case=c['id'],scenario='missing_output_rejected',status='FAIL'))
 result=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',scope='Constructed CPU-owned buffers faithfully exporting the C++ tuple/metadata schema; shared raw readback, source layout and full verdict exercised. No GPU numerics claimed.',rows=rows)
 save(A/'L48_READBACK.json',result);print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
