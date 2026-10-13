import gc, time, torch
n = 1 << 20
S = torch.cuda.Stream()
leg = torch.cuda.default_stream()
x = None
with torch.cuda.stream(S):
    x = torch.zeros(n, device='cuda')
S.synchronize()
def busy(): return not leg.query()
# E1: S work issued after a legacy sleep; does S wait for legacy?
for k in range(3):
    torch.cuda._sleep(20_000_000)
    with torch.cuda.stream(S):
        x.fill_(1.0)
    S.synchronize()
    print('E1 legacy busy after S sync:', busy()); torch.cuda.synchronize()
# E2: free a block allocated on S (pure torch), legacy sleep, reallocate on S, fill
for k in range(3):
    with torch.cuda.stream(S):
        p = torch.zeros(n, device='cuda')
    S.synchronize()
    ptr = p.data_ptr()
    torch.cuda._sleep(20_000_000)
    del p
    with torch.cuda.stream(S):
        q = torch.empty(n, device='cuda'); q.fill_(-7.0)
    S.synchronize()
    print('E2 reused', q.data_ptr() == ptr, 'legacy busy after S sync:', busy()); torch.cuda.synchronize()
# E3: same as E2 but the block passes through torch's own DLPack export/import (torch consumer), freed via deleter
for k in range(3):
    with torch.cuda.stream(S):
        p = torch.zeros(n, device='cuda')
    S.synchronize()
    ptr = p.data_ptr()
    y = torch.from_dlpack(p.__dlpack__(stream=1))
    del p
    torch.cuda._sleep(20_000_000)
    del y
    with torch.cuda.stream(S):
        q = torch.empty(n, device='cuda'); q.fill_(-7.0)
    S.synchronize()
    print('E3 reused', q.data_ptr() == ptr, 'legacy busy after S sync:', busy()); torch.cuda.synchronize()
