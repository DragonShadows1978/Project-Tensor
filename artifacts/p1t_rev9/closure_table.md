| Item | Status | Lines read / evidence |
|---|---|---|
| F01 | CLOSED | O:1756-1861; A:256-288; L:209-244; M:226-242,284-336; Prelaunch reservations, deferred ownership and checked fallback intact; release CPU gates pass. |
| F02 | CLOSED | O:511-516,645-649; A:258-261; L:202-208; M:66-105; Required-None and absent metadata refused before launch; descriptor/argument checks pass. |
| F03 | CLOSED | C:328-352,1161-1212,1234-1250; O:455-507; CM:31-89; A:101-124,175-188; Checked extents, bytes, addresses and geometry intact; UBSan/descriptor/build gates pass. |
| F04 | CLOSED | C:852-900; O:194-234; Finalizing refusal and admission handshake unchanged; finite exit callbacks finish before finalization. |
| F05 | CLOSED | K:289-320,493-519; Imported storage drains its own device, restores and leaks without completion proof; host multi-device trace passes. |
| F06 | CLOSED | C:357-390,639-678; Export context stays in unique_ptr until capsule creation succeeds; allocation failure unwinds ownership. |
| F07 | CLOSED | H:80-92; FIX1 addendum:17-31; add-on CMake:50-63; Native ABI rebuild requirement remains explicit; add-on includes/links no core ABI. |
| N1 | PARTLY closed | O:1531-1580,2104-2185; C:465-489,1632-1653; Original and FIX8 error paths work; V9-01 still loses an error and V9-02 loses its known attribution. |
| N2 | CLOSED | C:328-352,624-625; K:80-98; CM:63-89; O:469-498; Zero-neighbor extent bounds preserved; empty/scalar/huge metadata and UBSan checks pass. |
| N3 | CLOSED | O:1583-1626,1756-1861,2104-2185; Reserved release bookkeeping and caught push OOM intact; reporting records persist through allocation failure. |
| N4 | CLOSED | O:559-591; C:862-903; Ordinary throwing deleters contained and counted once; native throw-deleter gate passes. |
| N5 | CLOSED | C:862-903,986-1069,1544-1559,1732-1758; CI:253-308; Unconditional close/wait and native runner retained; knob/interrupt/lock CPU stagings pass. |
| V3-01 | CLOSED | C:940-1044; CI:207-308; Unbounded diagnosed wait and own-thread exclusion retained; finite callback exit gate passes. |
| V3-02 | CLOSED | O:1446-1499,1743-1760; Checked aggregate and sticky saturated queue bytes intact; accounting UBSan gate passes. |
| V4-01 | CLOSED | C:432-489,633-637; K:473-489; Own-device fence, destroy and restore intact; label proof retained. Physical ordering probe remains lead-only. |
| V4-02 | CLOSED | CI:107-117,207-249; SI:171-187; Public docs still state an unbounded wait and a diagnostic delay. |
| V4-03 | CLOSED | C:1142-1152,1217-1223; Checked device query precedes ownership/rename; failed import preserves capsule ownership. |
| V4-04 | CLOSED | O:2153-2185; Prior slot error recorded before consumption and drain; prior/OOM host gate passes. |
| V5-01 | CLOSED | O:218-249,331-380,559-591; SI:226-239; A:458-503; Deleter/capsule/producer admission retained; ordinary hooks and interrupted runners close/wait. |
| V5-02 | CLOSED | O:1864-1894; Next-call query failure recorded before consumption; one-shot host gate passes. |
| V5-03 | CLOSED | conv.cu:91-136; lstm.cu:93-138; New handle ownership survives cache-insertion OOM; unchanged by FIX8. |
| V5-04 | CLOSED | K:628-677; Wide ceil/grid refusal precedes fill launch and revision bump; huge-grid host refusal and nvcc compile pass. |
| V6-01 | CLOSED | O:331-345,2150-2160; Ordinary synchronize never waits for another thread; lock inversion and late-release gates pass. |
| V6-02 | CLOSED | C:546-600,1357-1452,1493-1559; PA:65-193,222-276; C-level argument reads/raw exit entries intact; 1067 argument checks and exit-entry census pass. |
| V6-03 | CLOSED | O:962-996,2153-2185; Drain failures persist before allocation and clear after captured RuntimeError; three drain/OOM gates pass. |
| V7-01 | CLOSED | C:1493-1559,1732-1758; A:356-407,458-503; CI:253-308; SI:226-278; Unreadable/deleted knobs and hook exceptions cannot skip quiescence; 36 exit children and lock checks pass. |
| V7-02 | CLOSED | O:780-951,1531-1580,1961-2009; G:89-117; Existing failed-call/purpose/device attribution retained; 22 host assertions pass. New core summary race is V9-02. |
| V8-01 | CLOSED | O:1169-1222,1248-1350; Marker failures recorded; fallback diagnosed; completion has proof. Four marker cases, 15 FIX8 host tests and 5 watchdog stagings pass. |
| V8-02 | CLOSED | O:2029-2185; A:428-503; C:1674-1758; K:426-439; Original final-report OOM silence fixed; both native OOM/control gates pass. New inconsistent-snapshot defect is V9-02. |
