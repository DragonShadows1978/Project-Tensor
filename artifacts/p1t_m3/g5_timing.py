"""P1T-M3 G5 (informational): tc_signal.lstm time on the real /shared/LSTM, max_1 (T 1193, bidirectional, I 640,
H 256, batch 1), median of 20 after warm-ups, plus torch cuDNN nn.LSTM fp32 on the same shape and weights.

Columns: wall = host time of the call (no sync); wall+sync = call then device sync; GPU = CUDA-event time on the
legacy default stream around the call (torch's default stream is the legacy stream, the one tc_signal queues on);
us/step = GPU / T. Two tc_signal operand paths: TensorCUDA tensors with the default allocator (tensor_cuda.empty; the
production path the adapter uses) and torch tensors with a torch allocator. torch nn.LSTM is timed with
cudnn.allow_tf32 False (fp32, comparable numerics) and True (torch's default), each with its N against my fp64 oracle.
Evidence class: micro-benchmark; it says nothing about model speed (P2). Prior art: none (a timing harness).
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WT, 'tc_signal', 'tests'))

import tc_signal as ts  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from tcs_lstm_oracle import lstm_onnx_f64, normwise, onnx_to_torch_gate_rows  # noqa: E402

FIX = '/mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures'
CASE = 'cases/T4/shared__LSTM__max_1.npz'
WARMUP = 3
REPS = 20


def load():
    m = json.load(open(os.path.join(FIX, 'manifest.json')))
    c = next(c for c in m['cases'] if c['case_file'] == CASE)
    cz, wz = np.load(os.path.join(FIX, c['case_file'])), np.load(os.path.join(FIX, c['weights_file']))
    return cz['x'], wz['W'], wz['R'], wz['B'], int(c['attrs']['hidden_size'])


def timed(fn):
    """(wall, wall+sync, gpu) lists in microseconds over REPS calls after WARMUP calls; returns the last output."""
    out = None
    for _ in range(WARMUP):
        out = fn()
    torch.cuda.synchronize()
    wall, wsync, gpu = [], [], []
    for _ in range(REPS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        e1.record()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        wall.append((t1 - t0) * 1e6)
        wsync.append((t2 - t0) * 1e6)
        gpu.append(e0.elapsed_time(e1) * 1e3)
    return out, wall, wsync, gpu


def row(name, T, wall, wsync, gpu, N=None):
    med = statistics.median
    r = {'path': name, 'wall_us': med(wall), 'wall_sync_us': med(wsync), 'gpu_us': med(gpu), 'us_per_step': med(gpu) / T,
         'gpu_us_min': min(gpu), 'gpu_us_max': max(gpu), 'N_vs_fp64_oracle': N}
    print(json.dumps(r))
    return r


def main():
    x, W, R, B, H = load()
    T, N, I = x.shape
    assert torch.cuda.current_stream().cuda_stream == 0, 'torch default stream is not the legacy stream'
    Y, Yh, Yc = lstm_onnx_f64(x, W, R, B, direction='bidirectional', hidden_size=H)
    res = {'case': CASE, 'shape': {'T': T, 'N': N, 'I': I, 'H': H, 'D': 2}, 'reps': REPS, 'warmup': WARMUP,
           'gpu': torch.cuda.get_device_name(0), 'loadavg': os.getloadavg(), 'rows': []}

    # tc_signal, TensorCUDA producers, default allocator (tensor_cuda.empty).
    ts.set_allocator(None)
    tx, tW, tR, tB = (tc.tensor(np.ascontiguousarray(a), dtype='float32') for a in (x, W, R, B))
    out, wall, wsync, gpu = timed(lambda: ts.lstm(tx, tW, tR, tB, hidden_size=H, direction='bidirectional'))
    n = max(normwise(o.numpy(), ref) for o, ref in zip(out, (Y, Yh, Yc)))
    res['rows'].append(row('tc_signal.lstm (TensorCUDA tensors, tensor_cuda.empty)', T, wall, wsync, gpu, n))

    # tc_signal, torch producers, torch allocator.
    ts.set_allocator(lambda s: torch.empty(s, device='cuda', dtype=torch.float32))
    gx, gW, gR, gB = (torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (x, W, R, B))
    out, wall, wsync, gpu = timed(lambda: ts.lstm(gx, gW, gR, gB, hidden_size=H, direction='bidirectional'))
    n = max(normwise(o.cpu().numpy(), ref) for o, ref in zip(out, (Y, Yh, Yc)))
    res['rows'].append(row('tc_signal.lstm (torch tensors, torch.empty)', T, wall, wsync, gpu, n))

    # torch cuDNN nn.LSTM fp32, same weights (ONNX -> torch gate order), with and without TF32.
    m = torch.nn.LSTM(I, H, num_layers=1, bias=True, batch_first=False, bidirectional=True).cuda().eval()
    with torch.no_grad():
        for d, suf in enumerate(['', '_reverse']):
            getattr(m, 'weight_ih_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(W[d])))
            getattr(m, 'weight_hh_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(R[d])))
            getattr(m, 'bias_ih_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(B[d, :4 * H])))
            getattr(m, 'bias_hh_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(B[d, 4 * H:])))
    res['cudnn_version'] = torch.backends.cudnn.version()
    for tf32 in (False, True):
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.no_grad():
            (yo, (hn, cn)), wall, wsync, gpu = timed(lambda: m(gx))
        yy = yo.view(T, N, 2, H).permute(0, 2, 1, 3).cpu().numpy()
        n = max(normwise(yy, Y), normwise(hn.cpu().numpy(), Yh), normwise(cn.cpu().numpy(), Yc))
        res['rows'].append(row(f'torch nn.LSTM cuDNN fp32 (allow_tf32={tf32})', T, wall, wsync, gpu, n))
    out_path = os.path.join(HERE, 'g5_timing.json')
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
        f.write('\n')
    print('wrote', out_path)


if __name__ == '__main__':
    main()
