# APAMQ-FA — INT4-Internal Causal Selective APA Kernel (+ Tensor-Core Rung)

YOUR WRITABLE TARGET is the git worktree you are launched in (a checkout
of Project-Tensor on branch `apamq-fa`) — edits to `tensor_cuda/src/`,
`tensor_cuda/tensor_cuda/`, `tensor_cuda/tests/`, and builds of the
worktree's own extension are AUTHORIZED. Run first, report after; a
registered order IS the permission.

HARD BOUNDARIES: never touch the canonical repo at
/mnt/ForgeRealm/Project-Tensor (your worktree is elsewhere; treat the
canonical tree and its installed .so as READ-ONLY reference), no git
(the lead commits/merges), no subagents, no network. Your sandbox has
NO GPU (proven twice today): build with nvcc, run CPU-side/pytest-skip
checks, and structure every GPU gate as a runnable pytest/script the
lead executes — that is the expected shape, not a failure. RED honesty;
no monitor-idling.

## Context (pre-nailed — do not re-litigate)

Read `docs/APA_MQA_FIX_PLAN.md` (in your worktree) first. You are
building workstream F-A.

The existing selective family (`apa_selective_kernel` and friends,
kernels.cu:1145+, host entries ~1847+) takes a PRE-RECONSTRUCTED
bf16 kq tensor — forcing ports to keep a resident kq ring (144 MiB at
16K on the Gemma port; the thing that makes APA memory-negative). The
noncausal DiT arc (kernels.cu:4594+) already proved the better design:
`apa_int4_pack_kernel` packs K per-key-vector symmetric INT4, and
`apa_int4_sdpa_noncausal_kernel` scores bulk from the packed form with
`apa_int4_dequant2` (fp32 dequant in-register), refines the tail from
exact K, exact softmax over all keys. Your job: bring that design to
the CAUSAL/SELECTIVE side.

## F-A1 (required): `apa_selective_attention_int4`

New host entry + kernels:

```
tc.apa_selective_attention_int4(q, k, v, scale, zthr, is_causal) -> out
```

- Signature = existing `apa_selective_attention` MINUS kq. Same
  (B,H,L,D)/(B,KVH,S,D) GQA-aware layout, same z-score threshold
  semantics (mean+z*std of |bulk*scale|, population variance), same
  merge (non-refined keys keep bulk — nothing dropped), same online
  softmax, BOTTOM-RIGHT causal (s_max=(S-L)+i+1; explicit non-square
  S>L tests — the 121→11M bug class), D up to TC_APA_MAXD=512.
- INT4 packing: reuse/extend `apa_int4_pack_kernel` (pack once per
  call into a workspace buffer, S*D/2 bytes + per-key scale — a
  transient, NOT persistent state) OR pack inline; your call, but the
  API must not require the caller to hold any persistent quantized
  state. Bulk dot via `apa_int4_dequant2`-style fp32 dequant.
- Decode coverage: mirror the existing family's dispatch (the split-K
  stats/split/merge path at L==1 small-grid shapes, WCOOP heuristics)
  so decode does not regress to a single-block-per-row grid. A shared
  packed workspace between the stats and split stages is fine.
- Sink variant (`_sink`) NOT required this order; leave a clean seam.

Gates (write them; lead runs GPU): a new
`tensor_cuda/tests/test_apa_selective_int4.py` covering (a) composed
fp32 numpy reference parity (EXP-APA-2 conventions from
test_apa_int4_sdpa_noncausal.py), (b) cross-check vs the existing
bf16-kq `apa_selective_attention` fed the DEQUANTIZED packed keys
(scores should match to fp32-vs-bf16 rounding tolerances — state your
tolerance and why), (c) rect-causal with-cache shapes S>L, (d) MQA
kv=1 D=512 and GQA kv∈{4,8} D=128 geometries, (e) decode L=1 split-K
path equivalence to the monolithic path.

## F-A2 (attempt; honest STOP allowed): tensor-core bulk pass

The measured gap: fused selective = 6–17× slower than the cuBLAS
standard path (artifacts/apamq_e1/RESULTS.md — canonical repo,
read-only) because bulk/refine dots ride CUDA cores. Attack the bulk
pass with a tiled/tensor-core formulation (mma.sync bf16 tiles, or a
staged GEMM into the online-softmax loop — your design; keep the APA
invariant: same threshold statistics, same selection, same merge
semantics, float-reassociation-class differences only). Registered
target (G-A2): ≤3× cuBLAS-standard wall at D=512 prefill S=16K, from
16.6×. Extend `scripts/apamq_e1_sweep.py` with the int4/tensor-core
paths so the lead can measure with the same harness. If the rung
does not land inside your leash, STOP honestly with a written design
+ what you measured/learned — F-A1 ships on its own.

## Done

Final message MUST contain verbatim: files created/changed with line
counts, the exact build command + result, the pytest command for CPU
checks + output, the test file's gate list with the tolerances you
registered, the F-A2 status (landed with design notes / STOPPED with
reasons), and any deviation from this order. No GPU numbers — do not
fabricate any; the lead measures.
