import sys
import time

import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m3/tc_signal/tests')
from tcs_lstm_oracle import lstm_onnx_f64  # noqa: E402

rng = np.random.default_rng(0)
T, N, I, H = 1193, 1, 640, 256
x = rng.standard_normal((T, N, I)).astype(np.float32)
W = rng.standard_normal((2, 4 * H, I)).astype(np.float32) * 0.08
R = rng.standard_normal((2, 4 * H, H)).astype(np.float32) * 0.1
B = rng.standard_normal((2, 8 * H)).astype(np.float32) * 0.08
if len(sys.argv) > 1 and sys.argv[1] == 'torch':
    import torch  # noqa: F401  (load torch's thread pools like the test process does)
t0 = time.time()
lstm_onnx_f64(x, W, R, B, direction='bidirectional', hidden_size=H)
print('oracle s', round(time.time() - t0, 3))
Wd = W[0].astype(np.float64)
xt = x[0].astype(np.float64)
t0 = time.time()
for _ in range(1000):
    xt @ Wd.T
print('x@W.T per call us', round((time.time() - t0) * 1e3, 1))
Rd = R[0].astype(np.float64)
h = np.zeros((1, H))
t0 = time.time()
for _ in range(1000):
    h @ Rd.T
print('h@R.T per call us', round((time.time() - t0) * 1e3, 1))
