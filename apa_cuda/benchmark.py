"""Benchmark APA-Quant attention against PyTorch's scaled_dot_product_attention.

This is the apples-to-apples comparison the C++/CUDA port exists for: both sides
now run as compiled kernels with no Python interpreter in the hot path, so the
numbers reflect the *algorithms*, not framework overhead.

Run on a CUDA box after building the extension::

    python apa_cuda/benchmark.py --seqlen 1024 --heads 8 --dim 64 --dtype fp16
"""

import argparse
import time

import torch
import torch.nn.functional as F

from apa_attention import apa_scaled_dot_product_attention as apa_sdpa


def _bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--refine", type=float, default=0.15)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--backward", action="store_true",
                    help="also time the forward+backward pass")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dev = "cuda"
    shape = (args.batch, args.heads, args.seqlen, args.dim)
    q = torch.randn(shape, device=dev, dtype=dtype, requires_grad=args.backward)
    k = torch.randn(shape, device=dev, dtype=dtype, requires_grad=args.backward)
    v = torch.randn(shape, device=dev, dtype=dtype, requires_grad=args.backward)

    def torch_fwd():
        return F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)

    def apa_fwd():
        return apa_sdpa(q, k, v, is_causal=args.causal,
                        refine_percentile=args.refine, bulk_bits=args.bits)

    if args.backward:
        def torch_step():
            torch_fwd().sum().backward()
        def apa_step():
            apa_fwd().sum().backward()
        torch_ms = _bench(torch_step, args.iters)
        apa_ms = _bench(apa_step, args.iters)
        mode = "fwd+bwd"
    else:
        with torch.no_grad():
            torch_ms = _bench(torch_fwd, args.iters)
            apa_ms = _bench(apa_fwd, args.iters)
        mode = "fwd"

    print(f"shape={shape} dtype={args.dtype} causal={args.causal} "
          f"refine={args.refine} bits={args.bits} mode={mode}")
    print(f"  PyTorch SDPA : {torch_ms:8.3f} ms/iter")
    print(f"  APA-Quant    : {apa_ms:8.3f} ms/iter")
    print(f"  speedup      : {torch_ms / apa_ms:8.2f}x")


if __name__ == "__main__":
    main()
