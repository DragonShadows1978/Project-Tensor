import sys, numpy as np, torch
sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m3/tc_signal/tests')
import tc_signal as ts
from tcs_lstm_oracle import lstm_onnx_f64, lstm_torch_f64, normwise
print('ts file', ts.__file__, ts.__all__)
ts.set_allocator(lambda s: torch.full(s, float('nan'), device='cuda', dtype=torch.float32))
print('cfg', ts._tc_signal.lstm_cublas_config(0))
rng = np.random.default_rng(1)
for direction, T, N, I, H, bias, init in [('forward', 5, 2, 3, 4, True, True), ('reverse', 7, 3, 1, 20, False, True),
                                          ('bidirectional', 9, 1, 16, 24, True, False), ('bidirectional', 1193, 1, 640, 256, True, False)]:
    D = 2 if direction == 'bidirectional' else 1
    x = rng.standard_normal((T, N, I)).astype(np.float32) * 0.5
    W = (rng.standard_normal((D, 4*H, I)) * 2 / np.sqrt(I)).astype(np.float32)
    R = (rng.standard_normal((D, 4*H, H)) * 1.7 / np.sqrt(H)).astype(np.float32)
    B = (rng.standard_normal((D, 8*H)) * 0.08).astype(np.float32) if bias else None
    h0 = rng.standard_normal((D, N, H)).astype(np.float32) if init else None
    c0 = rng.standard_normal((D, N, H)).astype(np.float32) if init else None
    cu = lambda a: None if a is None else torch.from_numpy(a).cuda()
    out = ts.lstm(cu(x), cu(W), cu(R), cu(B), hidden_size=H, direction=direction, initial_h=cu(h0), initial_c=cu(c0))
    out2 = ts.lstm(cu(x), cu(W), cu(R), cu(B), hidden_size=H, direction=direction, initial_h=cu(h0), initial_c=cu(c0))
    torch.cuda.synchronize()
    o = lstm_onnx_f64(x, W, R, B, direction=direction, hidden_size=H, initial_h=h0, initial_c=c0)
    ot = lstm_torch_f64(x, W, R, B, direction=direction, hidden_size=H, initial_h=h0, initial_c=c0)
    Ns = [normwise(a.cpu().numpy(), b) for a, b in zip(out, o)]
    X = [normwise(a, b) for a, b in zip(ot, o)]
    det = all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(out, out2))
    fin = all(bool(torch.isfinite(a).all()) for a in out)
    print(direction, T, N, I, H, 'N', ['%.3e' % v for v in Ns], 'xcheck', ['%.1e' % v for v in X], 'det', det, 'finite', fin)
