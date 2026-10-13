| Question | Verdict | Evidence / scope |
|---|---|---|
| R1 | SAFE | Ordinary core semantics preserved by source review; checked helpers remove overflow UB. Lead baseline-ID comparison independently matches 52/52; this is not a new GPU suite run. |
| R2 | UNSOUND | Lifetime/device/argument fixes hold; native exit error reporting fails under OOM (V8-02). |
| R3 | UNSOUND | Deferred-release/validation fixes hold; new diagnostic errors vanish and final exit reports can vanish (V8-01/V8-02). |
| R4 | CORRECT | Source arithmetic/indexing and geometry checked; host boundary/UBSan checks and CUDA compilation pass. GPU numerics not executed here. |
| R5 | PASS | Fresh full add-on build and core bindings/kernels compilation pass; ELF needs only shared CUDA/cuBLAS and standard libraries, no core ABI. |
| R6 | REJECT | Two real MINOR defects remain open. No BLOCKING/MAJOR found, but no-waiver rule prevents calling these harmless nits or accepted residuals. |
