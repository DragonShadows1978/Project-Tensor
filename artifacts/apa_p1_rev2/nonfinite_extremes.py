"""Source-derived q Inf cases, host arithmetic only.
Prior art: TurboQuant (Zandieh et al., 2025), IEEE 754-2008, inherited dense
online-normalizer sentinel and packed max reduction (Project-Tensor, 2026).
No novel algorithm. Pure table functions are extracted from actual source.
"""
from pathlib import Path
import ast,json,math
import numpy as np
root=Path(__file__).resolve().parents[2]
ns={"np":np,"math":math,"_ROT_CACHE":{},"_CODEBOOK_CACHE":{}}
tree=ast.parse((root/"tensor_cuda/tensor_cuda/quant.py").read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ["_beta_pdf","build_codebook","_rotation"]]
exec(compile(ast.Module(body=nodes,type_ignores=[]),"quant.py pure table functions","exec"),ns)
R=ns["_rotation"](64,42);cb=ns["build_codebook"](64,4);bnd=(cb[:-1]+cb[1:])/np.float32(2)
with np.errstate(all="ignore"):
 for qsign in [1,-1]:
  q=np.zeros(64,np.float32);q[0]=np.float32(qsign*np.inf)
  k=np.full(64,-qsign,np.float32);norm=np.sqrt(np.sum(k*k,dtype=np.float32));rot=R@(k/(norm+np.float32(1e-12)));idx=np.sum(rot[:,None]>bnd[None,:],axis=1);cent=cb[idx]
  recon=(cent@R)*norm;dense_bulk=np.dot(q,recon)*np.float32(.125)
  qr=R@q;packed_bulk=np.dot(qr,cent)*norm*np.float32(.125)
  dense_m=np.float32(-1e30);dense_l=np.float32(0);dense_acc=np.float32(0)
  for _ in range(3):
   next_m=np.fmax(dense_m,dense_bulk);corr=np.exp(np.float32(dense_m-next_m));w=np.exp(np.float32(dense_bulk-next_m));dense_l=np.float32(dense_l*corr+w);dense_acc=np.float32(dense_acc*corr+w);dense_m=next_m
  dense_out=dense_acc*(np.float32(1/dense_l) if dense_l>0 else np.float32(0))
  pmax=np.fmax.reduce(np.array([packed_bulk]*3,np.float32),initial=np.float32(-np.inf));packed_l=np.exp(np.float32(packed_bulk-pmax));packed_out=np.float32(packed_l/packed_l) if packed_l!=0 else np.float32(0)
  print(json.dumps(dict(case="q+Inf negative keys" if qsign==1 else "q-Inf positive keys",D=64,dense_reconstruction_first=float(recon[0]),dense_bulk=str(dense_bulk),packed_bulk=str(packed_bulk),dense_out=str(dense_out),packed_out=str(packed_out),proof_scope="source-derived host arithmetic; no GPU run")))
