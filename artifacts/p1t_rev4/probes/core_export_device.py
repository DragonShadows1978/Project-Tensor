"""LEAD ONLY, >=2 GPUs. PASS iff export refuses a device mismatch explicitly OR the device-0 consumer
finishes only after the producer event and reads the written values; FAIL on any early completion/stale value.
Prior art: DLPack Python stream handoff (dmlc 2017+, Array API 2021+) and CUDA 12.6 device-scoped events.
Taken: record/wait dependency; ours: device-switch adversary. timeout 60; <1 MiB, three sequential trials.
"""
import torch
import tensor_cuda as tc
if torch.cuda.device_count() < 2:
    print('INCONCLUSIVE: two GPUs required')
    raise SystemExit(3)
# Warm both contexts and event creation before staging a pending producer write.
for device in (0, 1):
    with torch.cuda.device(device):
        warm = torch.cuda.Event()
        warm.record()
        warm.synchronize()
torch.cuda.set_device(0)
legacy = torch.cuda.default_stream(0)
consumer = torch.cuda.Stream(device=0)  # PyTorch creates a nonblocking stream.
p = torch.zeros(1024, device='cuda:0')
y = torch.empty_like(p)
u = tc.from_dlpack(p)
producer_done = torch.cuda.Event()
producer_done.record(legacy)
producer_done.synchronize()
for trial in range(3):
    torch.cuda.set_device(0)
    p.zero_()
    torch.cuda.synchronize(0)
    with torch.cuda.stream(legacy):
        torch.cuda._sleep(200_000_000)
        u.fill_(7)
        producer_done.record(legacy)
    torch.cuda.set_device(1)
    try:
        cap = u.__dlpack__(stream=consumer.cuda_stream)
    except (BufferError, RuntimeError) as e:
        assert 'device' in str(e).lower(), str(e)
        assert torch.cuda.current_device() == 1
        torch.cuda.synchronize(0)
        print('PASS: explicit mismatch refusal', str(e))
        break
    assert torch.cuda.current_device() == 1, 'export changed caller device'
    with torch.cuda.device(0), torch.cuda.stream(consumer):
        imported = torch.utils.dlpack.from_dlpack(cap)
        y.copy_(imported)
    consumer.synchronize()
    # A correct handoff cannot let this read finish ahead of the producer event.
    ordered = producer_done.query()
    torch.cuda.synchronize(0)  # retain every real allocation through the probe, even on failure
    assert ordered, 'device-0 consumer finished while device-0 producer was pending'
    assert bool(torch.all(y == 7).item()), 'consumer observed stale data'
else:
    print('PASS: three bounded export-fence trials ordered on tensor device')
