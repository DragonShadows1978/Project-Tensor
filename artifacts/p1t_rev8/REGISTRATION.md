# P1T-REV8 registration (immutable)
Scope: independent review of 9df5bef..6991939, tensor_cuda/ and tc_signal/, emphasizing FIX7 429446d..6991939. User order is the immutable plan.
No source/test/build edits, GPU execution, subagents, deletes, or git writes.
Gates: CPU-only source/ABI/build/probe checks under foreground timeout, copies built under /tmp/p1t_rev8_*. A check passes only when its stated assertions execute and pass; timeout/error is RED or INCONCLUSIVE, never pass.
Verdict: REJECT on any open BLOCKING or MAJOR finding; APPROVE-WITH-NITS only without either; do not waive real defects. Unknown GPU claims require source-linked lead probes. Closure needs source evidence; receipts alone do not prove closure.
Prior art: DLPack spec, CPython 3.12 C API/lifecycle, CUDA runtime stream/event semantics, C++17 object lifetime/concurrency, ONNX operator specifications. Probes exercise product sources; no copied product implementation.
