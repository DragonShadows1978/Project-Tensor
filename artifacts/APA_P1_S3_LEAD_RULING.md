# APA-P1-S3 lead ruling (2026-11-03): RED-SCOPE as registered, at the order's own L82 stop; S1 is narrowed and the order re-issued as r2

**The round:**
- **The order:** `orders/APA_P1_S3_PROMOTION_ITEMS.md` (82a7fc15).
- **The seat:** Codex, gpt-6.1-sol at xhigh. Run `20261103T212710Z-2950177`, about 4 minutes.
- **The report:** `artifacts/apa_p1_s3/REPORT.md`.
- **The result:** no engine or test edits. G1 to G6 were not run; G7 passes.

## What happened

S1 asked for an explicit launch stream through every APA kernel in the chain. The seat's preflight found that this
reaches files the order kept read-only:
- `packed_launcher.cuh:14-41` (rotation, preparation and packing);
- `eb2b.cuh:109,114` (V quantization);
- `launcher.cuh`;
- the forwarding in `ops.h` and `bindings.cpp`.

The order told it to stop and list them, and it did (`SCOPE_STOP.json`).

## Ruling: S1 asked for more than production needs

The engine is single-stream by design:
- its kernels launch on the CUDA legacy default stream;
- `cudaStream_t` appears only in the DLPack bindings (`bindings.cpp`);
- DLPack hands data over on stream 1, the legacy default stream (`__init__.py:82-93`);
- the ticket code itself states the contract: "Single-stream, single-device use (the engine's legacy default
  stream)" (`eb3.cuh:683-687`).

**So the defects that matter for production are these:**
- the process-static ticket buffers (`eb3::counters`, `eb4::counters`) are shared across devices;
- `l2_bytes()` caches the first device's L2 size;
- nothing guards concurrent host threads.

Threading explicit streams through the whole chain would build multi-stream support the engine does not have.

**r2 narrows S1:**
- ownership by CUDA device, with separate merge and choose buffers and the L2 size cached per device;
- a host lock around lookup and growth;
- allocation and launch on the operands' device;
- the single-stream contract stated at the code site.

All edits stay inside `eb3.cuh`, `eb4.cuh` and `eb4_tuning.cuh`. S2 to S6 are unchanged.

## Receipts

The seat's preflight census (`TICKET_BOUNDARY.json`, `SCOPE_STOP.json`, `SOURCE_RECEIPTS.txt`) is accepted for use in
r2.

## r2 withdrawn (16:33)

The lead's edit building r2 (`ed324b49`) dropped the order's Rules heading and its first three bullets: the reading
list, no git writes, and no deletes. The lead's script replaced text up to the next "- **" marker. r2's seat (run
`20261103T213222Z-2958055`) was stopped by the lead about 40 s after launch, before writing anything; the worktree is
unchanged. r2b restores the Rules section and changes nothing else.
