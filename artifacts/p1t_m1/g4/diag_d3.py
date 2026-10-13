import gc, time, torch, tensor_cuda as tc
print(tc._C.__file__)
n = 1 << 20
S = torch.cuda.Stream()
leg = torch.cuda.default_stream()
with torch.cuda.stream(S):
    x = torch.zeros(n, device='cuda')
S.synchronize()
def busy(): return not leg.query()
def imported():
    with torch.cuda.stream(S):
        p = torch.zeros(n, device='cuda')
    S.synchronize()
    u = tc.from_dlpack(p); ptr = p.data_ptr(); del p
    return u, ptr
e = tc.empty(n); torch.cuda.synchronize()
for name in ('E4', 'E5', 'E6', 'E7', 'E8'):
    for k in range(2):
        u, ptr = imported()
        torch.cuda._sleep(20_000_000)
        if name == 'E4': e.fill_(1.0)
        if name in ('E5', 'E6', 'E8'): u.fill_(1.0)
        if name in ('E6', 'E7', 'E8'): del u
        with torch.cuda.stream(S):
            if name == 'E8':
                q = torch.empty(n, device='cuda'); q.fill_(-7.0)
            else:
                x.fill_(2.0)
        S.synchronize()
        print(name, 'legacy busy after S sync:', busy())
        torch.cuda.synchronize()
        u = None
