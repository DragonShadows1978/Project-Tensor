| Change | Verdict | Evidence |
|---|---|---|
| U1 | ACCEPTABLE | Both raw entries read any diagnostic value without Python methods; malformed shape still refuses. C:1495-1562; PA:222-279; fresh native-entry census. |
| U2 | ACCEPTABLE | Knob passed unchanged; unsupported values default. CI:248-255; SI:224-230; 22-class CPU exit census. |
| U3 | ACCEPTABLE | Native atexit runners cover pre-frame exceptions; unregister name documented. CI:294; SI:266; fresh interrupted-hook/runner CPU checks. |
| U4 | ACCEPTABLE | BaseException cannot skip the finally close; runner preserves original exception. SI:218-231; C:1640-1675; exception CPU census. |
| U5 | DEFECT (MINOR, V8-02) | Close/wait backstop works, but the promised last CUDA-failure report vanishes if rich-message allocation fails. O:1759-1770; C:1660-1672; both CPU OOM/control pairs. |
| U6 | ACCEPTABLE | One post-close retry reports a transient synchronize-message OOM. SI:233-238; CPU MemoryError staging. Persistent final-logger OOM is separately V8-02. |
| U7 | ACCEPTABLE | Existing changed messages name failed calls/purpose/device correctly; 22 fresh attribution assertions pass. O:821-939; cuda_guard.h:72-112; K:289-320,639-644. |
| U8 | ACCEPTABLE | Stats/attributes additive, counters are exposed and observed. O:1810 onward; C:2538-2582; raw-entry/lock census. |
| U9 | ACCEPTABLE | Both core lock attempts retry with GIL released and diagnosed delay; source C:939-1042 and fresh close/relock interposer checks pass. |
| U10 | DEFECT (MINOR, V8-01) | Additional change present in the actual report beyond the order U1-U9: marker-call failures erase errors and bypass the polled diagnostic wait. O:1049-1105; four-case CPU census RED. |
