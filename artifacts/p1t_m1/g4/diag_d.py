import gc, sys, time, numpy as np, torch, tensor_cuda as tc
print(tc._C.__file__)
n = 1 << 20
S = torch.cuda.Stream()
print('S handle', hex(S.cuda_stream), 'default', torch.cuda.default_stream().cuda_stream)
import ctypes
rt = ctypes.CDLL('libcudart.so.12') if False else None
for k in range(3):
    with torch.cuda.stream(S):
        p = torch.zeros(n, device='cuda')
    S.synchronize()
    u = tc.from_dlpack(p); ptr = p.data_ptr(); del p
    torch.cuda._sleep(20_000_000)
    u.fill_(float(k + 1))
    t0 = time.perf_counter()
    leg_busy_before = not torch.cuda.default_stream().query()
    del u
    t1 = time.perf_counter()
    leg_busy_after_del = not torch.cuda.default_stream().query()
    gc.collect()
    with torch.cuda.stream(S):
        q = torch.empty(n, device='cuda')
        q.fill_(-7.0)
    reused = q.data_ptr() == ptr
    S.synchronize()
    leg_busy_after_S = not torch.cuda.default_stream().query()
    torch.cuda.synchronize()
    print(f'trial {k}: reused={reused} legacy busy before del={leg_busy_before} after del={leg_busy_after_del} '
          f'after S sync={leg_busy_after_S} del took {1e3*(t1-t0):.2f} ms  q[0]={q[0].item()} all-7={bool((q==-7).all().item())}')
