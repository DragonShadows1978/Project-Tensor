import sys, subprocess, os
root = sys.argv[1]
pre = ("import sys; sys.path.insert(0, %r); import numpy as np, tensor_cuda as tc; "
       "cpu=lambda a: tc.from_numpy(np.ascontiguousarray(a), device='cpu'); ") % root
cases = {
 "import": "print('RETURN', tc.__file__)",
 "cpu numpy": "t=cpu(np.arange(6,dtype=np.float32)); print('RETURN', t.numpy())",
 "cpu reshape -1": "t=cpu(np.arange(24,dtype=np.float32)); r=t.reshape([2,-1]); print('RETURN', r.shape)",
 "cpu reshape -2,-12": "t=cpu(np.arange(24,dtype=np.float32)); r=t.reshape([-2,-12]); print('RETURN', r.shape)",
 "cpu data_ptr view": "t=cpu(np.arange(24,dtype=np.float32)); r=t.reshape([4,6]); print('RETURN', t.data_ptr()==r.data_ptr())",
 "rms_norm (2,0)": "r=tc._C.rms_norm(cpu(np.zeros((2,0),np.float32)), cpu(np.zeros((0,),np.float32)), 1e-6); print('RETURN', r.shape, r.dtype)",
 "rms_norm (0,4)": "r=tc._C.rms_norm(cpu(np.zeros((0,4),np.float32)), cpu(np.zeros((4,),np.float32)), 1e-6); print('RETURN', r.shape, r.dtype)",
 "causal (2,0,0)": "r=tc._C.causal_softmax(cpu(np.zeros((2,0,0),np.float32))); print('RETURN', r.shape, r.dtype)",
 "causal (0,4,4)": "r=tc._C.causal_softmax(cpu(np.zeros((0,4,4),np.float32))); print('RETURN', r.shape, r.dtype)",
 "apa_blend (1,1,2,0)": "r=tc._C.apa_blend_softmax(cpu(np.zeros((1,1,2,0),np.float32)), cpu(np.zeros((1,1,2,0),np.float32)), 0.0); print('RETURN', r.shape, r.dtype)",
 "apa_blend (1,0,2,3)": "r=tc._C.apa_blend_softmax(cpu(np.zeros((1,0,2,3),np.float32)), cpu(np.zeros((1,0,2,3),np.float32)), 0.0); print('RETURN', r.shape, r.dtype)",
 "topk (2,0) k0": "v,i=cpu(np.zeros((2,0),np.float32)).topk(0); print('RETURN', v.shape, v.dtype, i.shape, i.dtype)",
 "topk (0,5) k2": "v,i=cpu(np.zeros((0,5),np.float32)).topk(2); print('RETURN', v.shape, v.dtype, i.shape, i.dtype)",
 "topk (2,0) k1": "v,i=cpu(np.zeros((2,0),np.float32)).topk(1); print('RETURN', v.shape, v.dtype, i.shape, i.dtype)",
 "maxpool k5 on 2x2": "r=tc._C.max_pool2d(cpu(np.zeros((1,1,2,2),np.float32)),5,5,1,1,0,0); print('RETURN', r.shape)",
 "im2col k5 on 2x2": "r=tc._C.im2col(cpu(np.zeros((1,1,2,2),np.float32)),5,5,1,1,0,0); print('RETURN', r.shape)",
}
print(f"== CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
for label, expr in cases.items():
    p = subprocess.run([sys.executable, "-B", "-c", pre + expr], capture_output=True, text=True, timeout=120, cwd="/tmp/eng1_probe")
    out = (p.stdout.strip().splitlines() or [""])[-1]
    err = (p.stderr.strip().splitlines() or [""])[-1]
    print(f"[{label}] rc={p.returncode} out={out[:150]!r} err={err[:200]!r}")
