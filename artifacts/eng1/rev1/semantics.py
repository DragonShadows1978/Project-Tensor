# Prior art: PyTorch installed documentation/source: ATen InferSize.h, Pool.h; NumPy reshape.
# Taken: documented semantics, not inferred acceptance from candidate behavior.
import sys,json,subprocess,itertools,hashlib,ast
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'
sys.path.insert(0,str(root/'artifacts/eng1/tool/arms/candidate/tensor_cuda'))
import numpy as np, tensor_cuda as tc
import torch
torch.set_num_threads(1)
r={'torch_version':torch.__version__,'doc_view':torch.Tensor.view.__doc__,
   'doc_reshape':torch.reshape.__doc__,'doc_topk':torch.topk.__doc__,
   'doc_maxpool':torch.nn.MaxPool2d.__doc__}
# Mathematical empty inference/refusal census. Each source has exactly zero elements.
rows=[]
targets=list(itertools.product((-3,-2,-1,0,1,2,4),repeat=1))+list(itertools.product((-3,-2,-1,0,1,2,4),repeat=2))+list(itertools.product((-1,0,1,2),repeat=3))
for target in targets:
    def evaluate(fn):
        try:return dict(ok=True,shape=list(fn().shape))
        except Exception as e:return dict(ok=False,error=type(e).__name__,message=str(e))
    t=evaluate(lambda:torch.empty(0).reshape(target))
    c=evaluate(lambda:tc.from_numpy(np.zeros((0,),np.float32),device='cpu').reshape(list(target)))
    rows.append(dict(target=target,torch=t,candidate=c))
r['empty_cases']=rows
r['disagreements']=[x for x in rows if (x['torch']['ok'],x['torch'].get('shape')) != (x['candidate']['ok'],x['candidate'].get('shape'))]
r['large_empty']=[]
for shape in ((0,2**62,4),(0,2**63-1),(-1,2**62),(-1,2**62,4)):
    t=evaluate(lambda:torch.empty(0).reshape(shape)); c=evaluate(lambda:tc.from_numpy(np.zeros((0,),np.float32),device='cpu').reshape(list(shape)))
    r['large_empty'].append(dict(target=shape,torch=t,candidate=c))
# Exhaustively enumerate one-axis geometries in a fixed finite box; all 5 make_dims callers use this same quotient.
pool=[]; old_accept_new_reject=0; common_changed=0; checked=0
for n,k,s,p in itertools.product(range(0,21),range(1,25),range(1,7),range(0,5)):
    a=n+2*p-k; old=int(abs(a)//s)*(1 if a>=0 else -1)+1; new=a//s+1
    checked+=1
    if old>0 and new<=0: old_accept_new_reject+=1
    if old>0 and new>0 and old!=new: common_changed+=1
r['geometry']=dict(checked=checked,old_accept_new_reject=old_accept_new_reject,common_accepted_output_changes=common_changed,
                   example=dict(H=3,kernel=4,stride=2,padding=0,baseline_extent=1,candidate_extent=0))
# E6 AST canonical equality after removing tmp_path args and changing only the 8 writer path expressions back.
srcpath='tensor_cuda/tests/test_terrain_render.py'
old=subprocess.run(['git','show','1680764d:'+srcpath],capture_output=True,check=True,text=True).stdout
new=(root/srcpath).read_text()
norm=new.replace('(tmp_path):','():').replace('tmp_path /','repo_root / "artifacts" /')
r['terrain_ast_equal_after_path_normalization']=ast.dump(ast.parse(old))==ast.dump(ast.parse(norm))
r['terrain_write_calls']=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_write_ppm' for n in ast.walk(ast.parse(new)))
# Original scope hashes, and live bytes must match frozen candidate.
before=json.loads((root/'artifacts/eng1/tracked_before.json').read_text())
after=json.loads((root/'artifacts/eng1/tracked_after.json').read_text())
r['tracked_map_structure']={'before_type':type(before).__name__,'before_keys':list(before)[:6]}
(rev/'semantics.json').write_text(json.dumps(r,indent=2)+'\n')
print('regular empty shapes',len(rows),'disagreements',r['disagreements'])
print('huge shapes',r['large_empty'])
print('geometry',r['geometry'],'terrain AST equal',r['terrain_ast_equal_after_path_normalization'])
