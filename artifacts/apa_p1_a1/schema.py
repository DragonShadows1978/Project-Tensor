"""Complete retained-array census from P1 worker and C++ return shapes.
Prior art: P1/AB-1 integer-view schemas (2026); no new data representation.
"""
import numpy as np
def fields(c):
 B,H,K,S=[c[k] for k in ('B','H','KV','S')];out={}
 def add(name,shape,dtype):out[name]=dict(shape=list(shape),dtype=dtype)
 for n in c['append_lengths']:
  add(f'append_{n}_codes',(B,K,S,64),'uint8');add(f'append_{n}_norms',(B,K,S),'uint32')
  if n>=S-3:
   for f in ('eb3_out','eb4_out'):add(f'step_{n}_{f}',(B,H,1,128),'uint16')
   for f in ('eb3_mask','eb4_mask'):add(f'step_{n}_{f}',(B,H,1,n),'uint8')
   for f in ('eb4_flags','eb4_used'):add(f'step_{n}_{f}',(B,K,(n+63)//64),'uint8')
   add(f'step_{n}_int8_v',(B,K,n,128),'uint8');add(f'step_{n}_v_scales',(B,K,n),'uint32')
 for prefix,names in [('eb3',('qr','bulk','prefix')),('eb4',('qr','bulk','prefix','scores','logz','cm','cz','blog'))]:
  for name in names:
   shape={'qr':(B,H,1,128),'bulk':(B,H,1,S),'prefix':(B,H,1,S),'scores':(B,K,(S+63)//64),'logz':(B,H),'cm':(B,H,(S+255)//256),'cz':(B,H,(S+255)//256),'blog':(B,H,(S+63)//64)}[name]
   add(prefix+'_'+name,shape,'uint32')
 for f in ('eb3_nondiag','eb4_nondiag'):add(f,(B,H,1,128),'uint16')
 add('rotation',(K,128,128),'uint32');add('codebook',(16,),'uint32');add('valid_rows_history',(len(c['append_lengths']),),'uint64')
 return out
def payload_bytes(c):return sum(int(np.prod(r['shape']))*np.dtype(r['dtype']).itemsize for r in fields(c).values())
