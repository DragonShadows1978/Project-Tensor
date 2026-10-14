"""LEAD GPU ONLY: core protocol/bounds cases and ordinary-thread export release.
Prior art: DLPack v1.3 ownership/flags/strides; CUDA stream contract; array API.
Run under timeout 60. CPU empty cases never touch GPU; other cases need the reviewed core build.
Expected after fixes: invalid shapes refuse, protocol edges pass. No source edits.
"""
import gc, json, numpy as np, torch, tensor_cuda as tc
out={}
for label,shape in [('byte_count_wrap',(1<<62,)),('element_count_wrap',(1<<32,1<<32)),('rank9',(1,)*9)]:
    try:
        t=tc.empty(shape,device='cpu')
        out[label]={'accepted':True,'shape':t.shape,'data_ptr':t.data_ptr()}
    except (ValueError,RuntimeError,OverflowError) as e:
        out[label]={'accepted':False,'error':str(e)}
# Real torch producers: scalar, empty, offset and unusual singleton stride.
for label,p in [('scalar',torch.tensor(3.,device='cuda')),('empty',torch.empty(0,device='cuda')),
                ('offset',torch.arange(10.,device='cuda')[3:7]),
                ('singleton_stride',torch.empty_strided((2,1,3),(3,77,1),device='cuda'))]:
    if label=='singleton_stride': p.fill_(4)
    t=tc.from_dlpack(p)
    assert t.data_ptr()==p.data_ptr()
    np.testing.assert_array_equal(t.numpy(),p.cpu().numpy())
    cap=t.__dlpack__(max_version=(1,3)); again=tc.from_dlpack(cap)
    try: tc.from_dlpack(cap)
    except ValueError: pass
    else: raise AssertionError('capsule consumed twice')
    out[label]='PASS'
t=tc.empty(16).fill_(7)
for stream in (None,1,2,-1): t.__dlpack__(stream=stream)
try: t.__dlpack__(stream=0)
except ValueError: out['stream0']='refused'
else: raise AssertionError('stream=0 accepted')
print(json.dumps(out,indent=2))
assert all(not out[k]['accepted'] for k in ('byte_count_wrap','element_count_wrap','rank9')), 'F03: invalid shape accepted'
