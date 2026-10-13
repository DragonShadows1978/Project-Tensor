| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | O:1500-1603; A:262-288; L:209-244; M:229-242,287-336. Prelaunch reservation, completion retention and checked fallback intact; host release checks pass. |
| F02 | CLOSED | O:511-516,645-649; A:258-261; L:181-184,202-208; M:66-105. Required-None and absent-metadata refusals intact; argument census passes. |
| F03 | CLOSED | C:324-348,1162-1212,1241-1250; O:455-507; CM:31-89; A:101-124,175-188. Checked shape/byte/address/geometry bounds intact; UBSan and descriptor checks pass. |
| F04 | CLOSED | C:850-900; O:194-234. Finalizing-entry refusal and admission handshake retained; callback quiescence separately covered by N5/V5-01. |
| F05 | CLOSED | K:289-320,463-488. Imported Storage drains its own device and leaks without completion proof; restore and error records checked. Host multi-device trace passes; physical test lead-only. |
| F06 | CLOSED | C:637-676. unique_ptr owns export context until PyCapsule_New succeeds; all metadata allocation failures release it once. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-92; docs/P1T_FIX1_M1_ADDENDUM.md:17-31. Native ABI rebuild requirement explicit; fresh add-on includes/links no core ABI. |
| N1 | PARTLY closed | O:1287-1336,1773-1807 retain ordinary query/sync/prior errors; O:1049-1094 newly erases diagnostic errors (V8-01); O:1759-1770 drops exit reports on OOM (V8-02). Original sites work, global no-silent-error rule reopened. |
| N2 | CLOSED | C:324-348,621-624; K:80-97; CM:63-89; O:469-498. Zero-neighbor extent bound and bytes checked; zero/huge descriptor checks and UBSan pass. |
| N3 | CLOSED | O:1185-1199,1339-1370,1500-1603,1783-1807. Reserved release bookkeeping, caught push failure and persistent records survive OOM without terminate; CPU OOM checks pass. Last-reader logging is V8-02. |
| N4 | CLOSED | O:559-591; C:861-900. Manager pointer cleared before contained ordinary C++ deleter exception; fresh throwing-deleter CPU harness passes. |
| N5 | CLOSED | C:861-900,939-1068,1495-1562,1640-1675; CI:246-294. Close/wait cannot be bypassed by unreadable diagnostic values or pre-hook exceptions; overflow-knob and both lock-retry CPU stagings pass. |
| V3-01 | CLOSED | C:939-1042; CI:248-294. No timeout return; own-thread exclusion retained, runner covers interrupted hook; finite core callback completes before finalization. |
| V3-02 | CLOSED | O:1222-1255,1487-1509. Checked aggregate and sticky saturated queue bytes retained; 17-case inherited accounting UBSan check passes. |
| V4-01 | CLOSED | C:428-488,626-635; K:443-460. Label proof and tensor-device fence/destroy/restore intact; host multi-device trace passes; physical two-device fence remains lead-only. |
| V4-02 | CLOSED | CI:97-117,207-244. Documentation states unbounded wait and diagnostic delay; no reintroduced bounded-exit claim. |
| V4-03 | CLOSED | C:1141-1154,1217-1223. Checked cudaGetDevice before Storage ownership/rename; failed import keeps capsule ownership. |
| V4-04 | CLOSED | O:1776-1807,1708-1713. Prior slot persisted before drain and cleared only after captured RuntimeError; prior/OOM trace passes. |
| V5-01 | CLOSED | O:218-249,331-345,351-380,559-591,1422-1454; SI:214-231,266. Every deleter/capsule-drop/producer callback admitted; hook finally and runner always close. 22 knob classes and exception/interrupt CPU stagings pass. |
| V5-02 | CLOSED | O:1606-1643. Original next-call query failure recorded before clearing, raised once; query_once CPU check passes. New diagnostic-event site is V8-01, not a regression at this site. |
| V5-03 | CLOSED | tc_signal/src/conv.cu:90-135; tc_signal/src/lstm.cu:107-154. New-handle RAII survives map insertion OOM; FIX7 only changes status text/check helpers, not ownership. |
| V5-04 | CLOSED | K:596-647. Wide ceil/grid refusal before launch/revision intact; exact-source fill_blocks huge-grid CPU refusal passes; kernels.cu compiles. |
| V6-01 | CLOSED | O:331-345,1422-1427,1773-1782. Only exit close waits for other threads; ordinary synchronize still has no such wait. Fresh lock-inversion and late-release CPU checks pass. |
| V6-02 | CLOSED | C:544-599,1357-1454,1708-1768; PA:78-195. C-level value reads and raw shutdown entry intact. 1067 argument checks show zero conversion calls; both exit-entry censuses pass. |
| V6-03 | CLOSED | O:907-941,1000-1008,1783-1807. Tagged drain reports persist before message and clear after captured exception. All three inherited drain/OOM cases and 22 FIX7 attribution assertions pass; exit final-reader gap is V8-02. |
| V7-01 | CLOSED | C:1495-1562,1640-1675; PA:222-279; A:356-453; CI:246-294; SI:214-266. No Python knob conversion, C-level default, finally close, native runner; CPU native-entry, 22-class add-on and core overflow/lock controls pass. |
| V7-02 | CLOSED | O:780-941,1303-1325,1703-1751. FailedCall/Purpose/device/error travel together; query after successful sync names cudaEventQuery. Fresh query probe and plain/OOM attribution assertions pass. |
