| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | O:1160-1257; A:280-287; L:232-244; M:236-244,289-355. Prelaunch reserves, event retention, checked fallback and leak without completion proof intact. |
| F02 | CLOSED | O:484-488,618-627; A:241-264; L:181-211; M:66-105. Required None refusals and absent-metadata guard intact; CPU census/argument refusals pass. |
| F03 | CLOSED | C:322-346,1060-1111,1138-1188; O:414-480; CM:31-89; A:100-123,174-187. Checked rank/count/byte/address and geometry paths intact; UBSan census passes. |
| F04 | CLOSED | C:829-879; O:193-227. Original finalizing-entry GIL refusal/leak remains. Admitted-before-finalizing coordination is separately PARTLY under N5/V5-01. |
| F05 | CLOSED | K:286-308,447-473. Imported Storage drains its own device, checks restore, leaks on failed completion proof, records failures. FIX6 adds reporting without altering release order. |
| F06 | CLOSED | C:635-674. Export context stays under unique_ptr through metadata/capsule allocation, then transfers ownership once. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-92; docs/P1T_FIX1_M1_ADDENDUM.md:17-29. Coordinated native rebuild remains required; add-on includes/links no core ABI. |
| N1 | CLOSED | O:1026-1069,1354-1433. Query/sync/prior/deferred failures raise, leak only without proof; 8 fresh host trace cases pass. V7-02 is distinct incorrect attribution. |
| N2 | CLOSED | C:322-346,619-621; K:80-97; CM:63-89; O:440-480. Zero-neighbor extent/byte bounds checked; 18-case UBSan harness and operand census pass. |
| N3 | CLOSED | O:924-938,1072-1095,1160-1257,1354-1433. Pre-reserved owners/event slots, caught push failure, nonallocating drains and persistent OOM report records intact; CPU OOM propagation passes. |
| N4 | CLOSED | O:533-565; C:861-868. Ordinary C++ deleter exceptions contained, owner cleared before callback, never retried; fresh native throwing-deleter test passes. |
| N5 | PARTLY closed | C:840-970; CI:217-240. Native admission/unbounded wait correct, default control passes; diagnostic conversion can skip the hook entirely (V7-01). |
| V3-01 | PARTLY closed | C:899-954; CI:217-222. Native wait has no bound and handles own-thread exclusion; configured diagnostic OverflowError bypasses that wait (V7-01). |
| V3-02 | CLOSED | O:961-995,1147-1169,1253-1257. Checked aggregate and sticky saturated total preserve bounds; 17-case accounting UBSan harness passes. |
| V4-01 | CLOSED | C:426-485,624-633; K:431-444. Allocation-label proof and tensor-device fence/destroy/restore preserved. Physical two-device ordering remains lead-only. |
| V4-02 | CLOSED | CI:98-116,201-213. Import documentation correctly distinguishes unbounded wait from diagnostic-delay knob. V7-01 concerns exception handling, not the old bounded-wait text. |
| V4-03 | CLOSED | C:1039-1051,1116-1129. Device query checked before Storage ownership/rename; no FIX6 semantic regression at these sites. |
| V4-04 | CLOSED | O:1357-1367,1378-1383,1421-1433. Prior slot captured before drain and retained until exception exists; fresh prior_oom passes. |
| V5-01 | PARTLY closed | O:217-248,324-354,533-547,1119-1137; SI:186-209. All native callback kinds admitted and default hooks wait; FIX6 exposes diagnostic-conversion bypass (full add-on rc134; V7-01). |
| V5-02 | CLOSED | O:1260-1298,1409-1433; tc_signal/tests/test_fix5_findings.py (query regression); fresh query_once PASS. Non-NotReady drain failure recorded before clearing and raised once. |
| V5-03 | CLOSED | tc_signal/src/conv.cu:84-128; tc_signal/src/lstm.cu:107-146; tc_signal/tests/test_fix5_findings.py:894-919. Scoped handle ownership survives cache-insertion OOM; regenerated CPU conv/LSTM creates=destroys=1. |
| V5-04 | CLOSED | K:580-617; tensor_cuda/tests/test_dlpack.py:1935-1969. Wide grid calculation/refusal before launch/revision; regenerated CPU huge-grid witness passes. |
| V6-01 | CLOSED | O:304-319,1119-1123,1354-1367; tc_signal/tests/test_fix6_findings.py:154-164,362-373. Ordinary sync no other-thread wait; fresh full-addon lock witness and trace late-release pass. |
| V6-02 | CLOSED | C:542-598,1256-1354,1515-1521; tensor_cuda/tests/test_dlpack.py:2184-2209,2296-2303. C-level reads/refusal replace user conversion; actual-dispatch host census passes, no methods called. |
| V6-03 | CLOSED | O:734-765,830-831,1363-1367,1415-1433; tc_signal/tests/test_fix6_findings.py:404-487. Drain failures persisted allocation-free before message, cleared after captured exception; all3 fresh diagnostic_oom probes pass. |
