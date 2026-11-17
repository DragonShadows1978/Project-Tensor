"""P1 numerical worker extended to registered batch/head boundary shapes.
Prior art: P1/AB-1 raw-bit captures, append parity and readback (2026).
Taken: all original fields and six real cases; ours: head slicing/reshape and
A3 recurrence readback. No numerical, quantization or selection change.
"""
from gpu_common import *
import numpy as np
from layout import layout

def bits_to_float(a):return (np.ascontiguousarray(a).astype(np.uint32)<<16).view(np.float32)

def inputs(case):
 host={}
 for k,r in case['files'].items():
  assert sha(r['path'])==r['sha256']
  a=np.load(r['path'],mmap_mode='r',allow_pickle=False)
  assert a.dtype==np.uint16 and list(a.shape)==r['shape']
  assert np.isfinite(bits_to_float(a)).all() and np.any(a!=0) and np.any(a!=a.flat[0])
  host[k]=a
 return host

def geometry(host,case,n,key):
 B,H,K=[case[x] for x in ('B','H','KV')]
 heads=H if key=='q' else K
 if key=='q':
  pos=n-1-(32768-256) if case['S']==32768 else n-1
  a=host[key][:B*heads,pos:pos+1,:]
 else:a=host[key][:B*heads,:n,:]
 return np.ascontiguousarray(a).reshape(B,heads,1 if key=='q' else n,128)

def read_attention(tc,I,r3,r4,vi,vs,n,S,arrays):
 """Shared real/constructed post-device path; tuple indices match bindings.cpp.
 Prior art: P1 tuple census, DS1 A3 L48 readback (2026); no new representation.
 """
 for name,t in [('eb3_out',r3[0]),('eb3_mask',r3[2]),('eb4_out',r4[0]),('eb4_flags',r4[1]),('eb4_used',r4[2]),('eb4_mask',r4[8]),('int8_v',vi),('v_scales',vs)]:
  assert t is not None,(name,'unexpected absent field')
  arrays[f'step_{n}_{name}']=raw(tc,I,t)
 if n==S:
  for prefix,rs,items in [('eb3',r3,{'qr':1,'bulk':3,'prefix':4}),('eb4',r4,{'scores':3,'logz':4,'cm':5,'cz':6,'blog':7,'bulk':9,'prefix':10,'qr':11})]:
   for name,j in items.items():
    assert rs[j] is not None,(prefix,name)
    arrays[f'{prefix}_{name}']=raw(tc,I,rs[j])

def check_layout(tc,I,r3,r4,case):
 rows=[]
 for f,r,foot,indices in [('eb3',r3,7,{'pq':1,'mask':2,'bulk':3,'prefix':4}),('eb4',r4,12,{'pq':11,'mask':8,'bulk':9,'prefix':10,'flags':1,'used':2,'scores':3,'logz':4,'cm':5,'cz':6,'blog':7})]:
  expect=layout(f,case,True);metadata=raw(tc,I,r[foot]).view(np.int64)
  assert int(metadata[0])==expect['total'],(f,metadata.tolist(),expect)
  for name,j in indices.items():
   address,shape,dt,nbytes,base,pooled,is_cuda,device,offset=I.info(r[j])
   slot=next(x for x in expect['views'] if x['name']==name)
   # info.offset is in elements; the data pointer includes that byte offset.
   width={0:4,1:2,2:8,3:1,4:1,5:2}[dt]
   assert int(offset)*width==slot['offset'] and int(address)%16==0
   assert int(nbytes)==expect['total'],(f,name,nbytes,expect['total'])
  rows.append(dict(function=f,diag=True,expected=expect['total'],measured=int(metadata[0]),status='PASS'))
 return rows

def finish_arrays(outdir,case,arrays,history,provenance):
 arrays['valid_rows_history']=np.asarray(history,dtype=np.uint64)
 metadata={name:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest(),nonzero=bool(np.any(a!=0)),nonconstant=bool(a.size>1 and np.any(a!=a.flat[0]))) for name,a in arrays.items()}
 guards=all(r['nonzero'] and r['nonconstant'] for r in metadata.values()) if case['kind']=='original_real_capture' else all(metadata[f'step_{case["S"]}_{name}']['nonzero'] and metadata[f'step_{case["S"]}_{name}']['nonconstant'] for name in ('eb3_out','eb4_out'))
 with (outdir/'integer_views.npz').open('xb') as f:np.savez(f,**arrays)
 save(outdir/'result.json',dict(status='PASS' if guards else 'FAIL_NONVacuous',case=case,provenance=provenance,arrays=metadata,nonvacuity_pass=guards,input_sha256={k:r['sha256'] for k,r in case['files'].items()},input_kind='real model post-norm/post-RoPE bf16 captures; registered head slice/reshape for seven small shapes'))
 assert guards,'uninformative reference: see result.json'

def numerical(args):
 tc,I,provenance=runtime(args.arm);os.environ['TC_APA_SP']='1';os.environ['APA_FXTC1_VARIANT']='2'
 case=next(c for c in load(ART/'INPUT_CENSUS.json') if c['id']==args.case)
 host=inputs(case);S=case['S'];outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
 def tensor(bits):
  bits=np.ascontiguousarray(bits);t=tc.tensor(bits_to_float(bits),dtype='bfloat16')
  assert np.array_equal(raw(tc,I,t),bits),'bf16 input upload changed bits'
  return t
 arrays={};history=[];layouts=[]
 for n in case['append_lengths']:
  disk();k=tensor(geometry(host,case,n,'k'))
  if not history:w=tc.ApaSpq1Workspace(k,S,fmt='tq4')
  else:w.append(k)
  assert w.valid_rows==n and w.capacity==S
  history.append(n);arrays[f'append_{n}_codes']=raw(tc,I,w.codes);arrays[f'append_{n}_norms']=raw(tc,I,w.scales)
  if n>=S-3:
   v=tensor(geometry(host,case,n,'v'));q=tensor(geometry(host,case,n,'q'));vi,vs=tc.apa_eb2b_quantize(v)
   scale=float(np.float32(1/np.sqrt(128)));delta=float(np.float32(8.1))
   r3=tc.apa_eb3(q,k,v,w,scale,delta,carry='row',diagnostics=True,variant=0)
   r4=tc.apa_eb4(q,k,v,vi,vs,w,scale,delta,case['fraction'],carry='row',diagnostics=True)
   read_attention(tc,I,r3,r4,vi,vs,n,S,arrays)
   if n==S:
    layouts=check_layout(tc,I,r3,r4,case)
    for name,t in [('eb3_nondiag',tc.apa_eb3(q,k,v,w,scale,delta,carry='row')),('eb4_nondiag',tc.apa_eb4(q,k,v,vi,vs,w,scale,delta,case['fraction'],carry='row'))]:arrays[name]=raw(tc,I,t)
    for f in ('eb3','eb4'):
     assert np.array_equal(arrays[f+'_nondiag'],arrays[f'step_{S}_{f}_out'])
     layouts.append(dict(function=f,diag=False,expected=layout(f,case,False)['total'],status='PASS_NO_LAYOUT_ASSERTION',measurement='Opaque non-diagnostic API returns output only; layout equality covered by source recurrence.'))
 arrays['rotation']=raw(tc,I,w.rotation);arrays['codebook']=raw(tc,I,w.codebook)
 finish_arrays(outdir,case,arrays,history,provenance)
 save(outdir/'layout.json',dict(status='PASS',rows=layouts))
