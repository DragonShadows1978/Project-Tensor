import gc, time, torch, tensor_cuda as tc
n = 1 << 20
S = torch.cuda.Stream()
leg = torch.cuda.default_stream()
def busy(): return not leg.query()
for name in ('E8', 'E9', 'E10'):
    for k in range(2):
        with torch.cuda.stream(S):
            p = torch.zeros(n, device='cuda')
        S.synchronize()
        u = tc.from_dlpack(p); del p
        torch.cuda._sleep(20_000_000)
        u.fill_(1.0)
        del u
        b0 = busy()
        if name == 'E9':
            t0 = time.perf_counter(); c = gc.collect(); dt = time.perf_counter() - t0
        if name == 'E10':
            gc.set_debug(gc.DEBUG_SAVEALL); c = gc.collect(); gc.set_debug(0)
            kinds = {}
            for o in gc.garbage: kinds[type(o).__name__] = kinds.get(type(o).__name__, 0) + 1
            gc.garbage.clear(); dt = 0
        b1 = busy()
        with torch.cuda.stream(S):
            q = torch.empty(n, device='cuda'); q.fill_(-7.0)
        S.synchronize()
        extra = f'gc collected {c} in {1e3*dt:.2f} ms' if name != 'E8' else ''
        if name == 'E10': extra += f' kinds={sorted(kinds.items(), key=lambda kv: -kv[1])[:12]}'
        print(name, 'busy after del', b0, 'after gc', b1, 'after S sync', busy(), extra)
        torch.cuda.synchronize()
