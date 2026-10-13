# P1T-REV registration (immutable)
Scope: review 9df5bef..00590b7 in tensor_cuda/ and tc_signal/ against the supplied P1T-REV order and alternative plan. No source edits, GPU execution, subagents, git writes or deletes.
Evidence: source reasoning; CPU-only static/host checks with timeout. All unresolved device claims receive lead-run probes. No numerical or performance promotion claims from CPU checks.
Rail: APPROVE only if no blocking or major unresolved correctness/memory safety defect; unresolved GPU validation explicitly stays pending. Author reports read only after independent source inspection.
Census: zero/0-d, offsets, singleton strides, read-only/versioned/used capsules, nonlegacy streams, finalization/exception cleanup, int32/64 boundaries, huge T, L=1, NaN/Inf, groups and aliasing.
Prior art: DLPack/array API ownership and streams; CUDA stream ordering; cuBLAS size/handle semantics; ONNX operators; shifted two-pass variance. Probe comments, ledger and report carry attribution.
