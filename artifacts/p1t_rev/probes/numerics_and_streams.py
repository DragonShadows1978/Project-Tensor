"""LEAD GPU ONLY: finite semantic parity + dual-handle interleave + nonlegacy ordering.
Prior art: ONNX operators and fp64 oracle; CUDA events; cuBLAS default-math determinism.
Reuses author-supplied fp64 oracles as a spot check, not independent review evidence. timeout 90, <32 MiB.
This is a spot probe, not the claimed 236-case Voice runner or 52-id core baseline comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tc_signal/tests'))
import numpy as np, torch, tc_signal as ts, tensor_cuda as tc
from tc_signal import _tc_signal as C
import tcs_lstm_oracle as O
# Both cuBLAS handles must retain requested settings after interleaved use.
ts.set_allocator(None)
rng=np.random.default_rng(20261012)
def upload(shape): return tc.tensor(rng.normal(0,.1,shape).astype(np.float32))
x=upload((1,2,16)); w=upload((3,2,3))
a=ts.conv1d(x,w)
T,N,I,H,D=5,2,3,4,2
lx=upload((T,N,I)); W=upload((D,4*H,I)); R=upload((D,4*H,H)); B=upload((D,8*H))
b=ts.lstm(lx,W,R,B,hidden_size=H,direction='bidirectional')
a2=ts.conv1d(x,w); b2=ts.lstm(lx,W,R,B,hidden_size=H,direction='bidirectional')
np.testing.assert_array_equal(a.numpy(),a2.numpy())
for first,second in zip(b,b2): np.testing.assert_array_equal(first.numpy(),second.numpy())
reference=O.lstm_onnx_f64(lx.numpy(),W.numpy(),R.numpy(),B.numpy(),direction='bidirectional',hidden_size=H)
for got,ref in zip(b,reference): assert O.normwise(got.numpy(),ref)<=2e-5
for cfg in (C.cublas_config(),C.lstm_cublas_config()):
    assert cfg['default_math'] and cfg['atomics_not_allowed'] and cfg['legacy_stream'],cfg
# L=1 with positive eps is zero before affine; eps=0 has IEEE 0/0 behavior.
x1=tc.tensor(np.array([[[2.]]],np.float32))
np.testing.assert_array_equal(ts.instance_norm1d(x1).numpy(),np.zeros((1,1,1),np.float32))
assert np.isnan(ts.instance_norm1d(x1,eps=0).numpy()).all()
values=np.array([-np.inf,np.inf,np.nan,-.5,.5,1.5,2.5,-.4,-0.],np.float32)
np.testing.assert_array_equal(ts.rint(tc.tensor(values)).numpy(),np.rint(values))
np.testing.assert_array_equal(np.signbit(ts.rint(tc.tensor(values)).numpy()),np.signbit(np.rint(values)))
# Producer is live through drain: this tests forward ordering, independently of F01 reclamation.
S=torch.cuda.Stream()
with torch.cuda.stream(S):
    p=torch.empty(1024,device='cuda');torch.cuda._sleep(20_000_000);p.fill_(1.25)
y=ts.rint(p)
np.testing.assert_array_equal(y.numpy(),np.ones(1024,np.float32))
print('PASS: finite determinism, handle settings, L=1, special rint, nonlegacy producer ordering')
