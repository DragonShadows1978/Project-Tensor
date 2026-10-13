| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | tc_signal/src/dlpack_operand.cpp:787-875; bindings.cpp:284-286; lstm_bindings.cpp:239-242; m4_bindings.cpp:239-241,294-295,315-316,334-335 — Event-retained managers cover success/partial launch; checked fallback leaks without completion proof. V5-01 is a separate shutdown gap. |
| F02 | CLOSED | tc_signal/src/dlpack_operand.cpp:197-201,315-324; m4_bindings.cpp:65-104; bindings.cpp:240-260; lstm_bindings.cpp:178-210 — Required None refusal; absent metadata raises. Fresh CPU refusal checks pass. |
| F03 | CLOSED | tensor_cuda/src/bindings.cpp:312-336,937-988,1017-1026; tc_signal/src/checked_math.h:31-89; dlpack_operand.cpp:127-194 — Original rank/count/byte/address/geometry overflow paths checked. V5-04 separately finds fill grid narrowing. |
| F04 | CLOSED | tensor_cuda/src/bindings.cpp:710-775,794-847; tensor_cuda/tensor_cuda/__init__.py:201-236 — Core admission closes before finalization; late releases leak without GIL. Add-on callback gap is V5-01. |
| F05 | CLOSED | tensor_cuda/src/kernels.cu:232-248,338-364 — External Storage drains its own device; failed drain leaks; caller restore checked/counted. |
| F06 | CLOSED | tensor_cuda/src/bindings.cpp:531-570 — unique_ptr retains export context until capsule transfer; metadata exceptions unwind ownership. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-88; docs/P1T_FIX1_M1_ADDENDUM.md:20-33 — Coordinated native rebuild documented; closure is rollout prerequisite, not old ABI compatibility. |
| N1 | CLOSED | tc_signal/src/dlpack_operand.cpp:695-734,924-988; tc_signal/tc_signal/__init__.py:139-155 — Original query/sync failures raise; CPU no-device check passes. V5-02 is a distinct query-error accounting gap. |
| N2 | CLOSED | tensor_cuda/src/bindings.cpp:312-336,515-517; kernels.cu:80-98; tc_signal/src/checked_math.h:63-89 — Zero-neighbor extent/byte products checked; fresh 18-case UBSan census passes. |
| N3 | CLOSED | tc_signal/src/dlpack_operand.cpp:480-484,593-607,695-734,737-760,787-875 — Pre-reserved owners/recycle slots; nonallocating drain; pending push caught. CPU OOM propagation passes. |
| N4 | CLOSED | tc_signal/src/dlpack_operand.cpp:238-260; tensor_cuda/src/bindings.cpp:757-763 — Ordinary C++ deleter exceptions contained once; native CPU throwing-owner check passes. |
| N5 | CLOSED | tensor_cuda/src/bindings.cpp:734-775,794-847; tensor_cuda/tensor_cuda/__init__.py:201-236 — Original core admission/GIL race coordinated. Add-on has its own uncoordinated releases (V5-01). |
| V3-01 | CLOSED | tensor_cuda/src/bindings.cpp:794-847; tensor_cuda/tensor_cuda/__init__.py:201-236 — No bounded early return; fresh core 5.5s lifecycle slice exits0, in-flight0, diagnostic1. |
| V3-02 | CLOSED | tc_signal/src/dlpack_operand.cpp:630-663,774-795,827,871-875 — Checked aggregate before launch and sticky saturated queue total; 17 fresh UBSan accounting checks pass. |
| V4-01 | CLOSED | tensor_cuda/src/bindings.cpp:414-477,525-529; kernels.cu:322-335 — Fence selects tensor device; label proof; checked destroy/restore. Real two-device witness still pending. |
| V4-02 | CLOSED | tensor_cuda/tensor_cuda/__init__.py:107-116 — Import docstring describes unbounded wait and diagnostic-delay knob. |
| V4-03 | CLOSED | tensor_cuda/src/bindings.cpp:916-928,993-999 — Device query checked before Storage construction/capsule rename. Historical full-core refusal receipt audited. |
| V4-04 | CLOSED | tc_signal/src/dlpack_operand.cpp:924-988 — Prior error taken before drain, kept through message allocation and raised. Source and historical one-GPU receipt agree. |
