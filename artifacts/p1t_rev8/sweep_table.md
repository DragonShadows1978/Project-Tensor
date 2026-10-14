| Row | Source sites | Verdict | Check |
|---|---|---|---|
| i1 | A:417-453; SI:214 | CONFIRMED | Fresh first-bytecode interrupt/deleted-_C children close and wait through runner. |
| i2 | A:417-453; SI:214 | CONFIRMED | Fresh first-bytecode interrupt/deleted-_C children close and wait through runner. |
| i3 | SI:215-217 | CONFIRMED | Constant local stores run no callback or conversion. |
| i4 | SI:218-231 | CONFIRMED | Exception/BaseException from synchronize cannot skip finally close; five exception-kind children pass. |
| i5 | SI:218-231 | CONFIRMED | Exception/BaseException from synchronize cannot skip finally close; five exception-kind children pass. |
| i6 | SI:224-230; PA:222-279 | CONFIRMED | Deleted knob or unreadable value selects default; 22 knob-class CPU children pass, zero method calls. |
| i7 | SI:224-230; PA:222-279 | CONFIRMED | Deleted knob or unreadable value selects default; 22 knob-class CPU children pass, zero method calls. |
| i8 | PA:239-279; A:356-371 | CONFIRMED | Malformed call refuses before close; every value uses C read/default. Native-entry CPU census passes. |
| i9 | PA:239-279; A:356-371 | CONFIRMED | Malformed call refuses before close; every value uses C read/default. Native-entry CPU census passes. |
| i10 | O:1422-1454 | CONFIRMED | Atomic close precedes unbounded wait; result allocation follows. close_raises child still reaches quiescence. |
| i11 | O:1422-1454 | CONFIRMED | Atomic close precedes unbounded wait; result allocation follows. close_raises child still reaches quiescence. |
| i12 | SI:232-260 | CONFIRMED | stderr work follows close; three broken-stderr children complete. |
| i13 | A:417-453 | CONFIRMED FOR QUIESCENCE | Native runner always reaches close after hook raises; final-report allocation gap separately i24. |
| i14 | C:1640-1675; CI:246 | CONFIRMED | Raw runner closes after pre-hook exception; source and raw-runner CPU control checked. |
| i15 | C:1640-1675; CI:246 | CONFIRMED | Raw runner closes after pre-hook exception; source and raw-runner CPU control checked. |
| i16 | CI:248-255; C:1495-1562 | CONFIRMED | No diagnostic conversion callbacks; huge-int real hook plus both raw entry call styles pass. |
| i17 | CI:248-255; C:1495-1562 | CONFIRMED | No diagnostic conversion callbacks; huge-int real hook plus both raw entry call styles pass. |
| i18 | CI:248-255; C:1495-1562 | CONFIRMED | No diagnostic conversion callbacks; huge-int real hook plus both raw entry call styles pass. |
| i19 | C:939-1042 | CONFIRMED | Initial/re-lock failures retried with diagnostics; 15-failure CPU close and relock witnesses pass. |
| i20 | C:1015-1042 | CONFIRMED | Current libstdc++ timed wait path retains explicit GIL restore; no early timeout return; scoped implementation reasoning. |
| i21 | CI:257-290; C:1608-1627 | CONFIRMED AT LISTED SITES | Post-close result failure cannot skip raising records; runtime record consumption waits for captured RuntimeError. Rich final-logger failure is i25. |
| i22 | CI:257-290; C:1608-1627 | CONFIRMED AT LISTED SITES | Post-close result failure cannot skip raising records; runtime record consumption waits for captured RuntimeError. Rich final-logger failure is i25. |
| i23 | A:424-435; O:1287-1336 | CONFIRMED | Interrupted runner drains before closing when hook did not close; existing source-linked release/interrupt probes pass. |
| i24 | SI:233-238; A:424-451; O:1759-1770 | PARTLY | Transient message OOM retry works, but last native logger swallows bad_alloc: V8-02. Fresh OOM/control pair disproves unconditional no-loss claim. |
| i25 | CI:275-283; C:1660-1672 | PARTLY | Normal interrupted-hook report exists; core runner swallows reporting bad_alloc with no output: V8-02, independently reproduced. |
| i26 | A:444-449; C:1646-1653 | CONFIRMED | Successful rich result reports runner wait; OOM of final failure logger separately covered by i24/i25. |
| j1 | O:1292-1299,825-829 | CONFIRMED | Current-device query names cudaGetDevice and does not invent a current device; plain/OOM trace assertions pass. |
| j2 | O:1098-1137,1305-1308,830-843 | CONFIRMED | Guard tags and spill device retained; guard/set/get and 70-device spill trace assertions pass. |
| j3 | O:1098-1137,1305-1308,830-843 | CONFIRMED | Guard tags and spill device retained; guard/set/get and 70-device spill trace assertions pass. |
| j4 | O:915-939,1324,1720-1732 | CONFIRMED | Query tag preserved through merge/OOM; capped record overflow counts unnamed failures; fresh trace assertions pass. |
| j5 | O:915-939,1324,1720-1732 | CONFIRMED | Query tag preserved through merge/OOM; capped record overflow counts unnamed failures; fresh trace assertions pass. |
| j6 | O:915-939,1324,1720-1732 | CONFIRMED | Query tag preserved through merge/OOM; capped record overflow counts unnamed failures; fresh trace assertions pass. |
| j7 | O:936-939,1548-1602 | CONFIRMED | First() pairs one call/device/error; record guard and fallback differentiated; bound/fallback trace assertions pass. |
| j8 | O:936-939,1548-1602 | CONFIRMED | First() pairs one call/device/error; record guard and fallback differentiated; bound/fallback trace assertions pass. |
| j9 | O:936-939,1548-1602 | CONFIRMED | First() pairs one call/device/error; record guard and fallback differentiated; bound/fallback trace assertions pass. |
| j10 | O:1170-1182,1481-1484,1606-1643 | CONFIRMED | Destroy guard, next-call query and caller restore name actual API/target. Fresh destroy/query/restore host assertions pass. |
| j11 | O:1170-1182,1481-1484,1606-1643 | CONFIRMED | Destroy guard, next-call query and caller restore name actual API/target. Fresh destroy/query/restore host assertions pass. |
| j12 | O:1170-1182,1481-1484,1606-1643 | CONFIRMED | Destroy guard, next-call query and caller restore name actual API/target. Fresh destroy/query/restore host assertions pass. |
| j13 | O:1524-1534,1708-1713 | CONFIRMED | Pre-existing per-thread slot explicitly has unknown producing call/device; prior and prelaunch trace assertions pass. |
| j14 | O:1524-1534,1708-1713 | CONFIRMED | Pre-existing per-thread slot explicitly has unknown producing call/device; prior and prelaunch trace assertions pass. |
| j15 | O:1339-1369 | CONFIRMED | Creation vs failed guard call distinguished; creation-guard CPU assertion passes. |
| j16 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j17 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j18 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j19 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j20 | tc_signal/src/conv.cu:283-321; cuda_guard.h:120-189 | CONFIRMED | No-device property query and scoped guard target/restore messages match call purpose. |
| j21 | tc_signal/src/conv.cu:283-321; cuda_guard.h:120-189 | CONFIRMED | No-device property query and scoped guard target/restore messages match call purpose. |
| j22 | SI:232-260; O:276-326 | CONFIRMED | Hook exception/result text follows close; timed/polling wait text describes actual wait; CPU staging passes. |
| j23 | SI:232-260; O:276-326 | CONFIRMED | Hook exception/result text follows close; timed/polling wait text describes actual wait; CPU staging passes. |
| j24 | A:428-451; O:1759-1770 | PARTLY | Existing emitted text is correct, but OOM silently emits no recorded CUDA failure: V8-02. Diagnostic event calls added by L1 are missing from j sweep: V8-01. |
| k1 | K:289-320,339-371 | CONFIRMED BY SOURCE/BUILD | Each imported-drain branch names one API; restore names its target; fill scoped guard remains checked. kernels.cu compiles. |
| k2 | K:289-320,339-371 | CONFIRMED BY SOURCE/BUILD | Each imported-drain branch names one API; restore names its target; fill scoped guard remains checked. kernels.cu compiles. |
| k3 | K:289-320,339-371 | CONFIRMED BY SOURCE/BUILD | Each imported-drain branch names one API; restore names its target; fill scoped guard remains checked. kernels.cu compiles. |
| k4 | K:449-451,622-644 | CONFIRMED BY SOURCE/BUILD | Pending-slot source unknown; launch-check ambiguity stated; failed pointer query says allocation device unknown. kernels.cu compiles; real driver checks lead-only. |
| k5 | K:449-451,622-644 | CONFIRMED BY SOURCE/BUILD | Pending-slot source unknown; launch-check ambiguity stated; failed pointer query says allocation device unknown. kernels.cu compiles; real driver checks lead-only. |
| k6 | K:449-451,622-644 | CONFIRMED BY SOURCE/BUILD | Pending-slot source unknown; launch-check ambiguity stated; failed pointer query says allocation device unknown. kernels.cu compiles; real driver checks lead-only. |
| k7 | C:428-488,1141-1154,1261-1286 | CONFIRMED | Fence errors name failing step/tensor device and restore target; import/empty current query has no invented current device. |
| k8 | C:428-488,1141-1154,1261-1286 | CONFIRMED | Fence errors name failing step/tensor device and restore target; import/empty current query has no invented current device. |
| k9 | C:1596-1627,1646-1675; CI:257-290 | CONFIRMED FOR EMITTED TEXT | Reporter paths and first call/device correctly described. Missing final text on OOM is independently accounted in i25/V8-02. |
| k10 | C:1596-1627,1646-1675; CI:257-290 | CONFIRMED FOR EMITTED TEXT | Reporter paths and first call/device correctly described. Missing final text on OOM is independently accounted in i25/V8-02. |
| k11 | C:1596-1627,1646-1675; CI:257-290 | CONFIRMED FOR EMITTED TEXT | Reporter paths and first call/device correctly described. Missing final text on OOM is independently accounted in i25/V8-02. |
| a1 | O:276-345 | CONFIRMED | Condition wait and polling fallback diagnose; fresh interposed mutex failure writes polling lines and callback finishes before finalization. |
| a2 | O:276-345 | CONFIRMED | Condition wait and polling fallback diagnose; fresh interposed mutex failure writes polling lines and callback finishes before finalization. |
| a3 | O:1049-1105; A:399-418 | PARTLY | Healthy marker polling diagnoses. Failed create/record returns into silent cudaStreamSynchronize; nonsticky event errors cleared without record (V8-01). |
| a4 | O:1260-1265,559-574 | CONFIRMED | Executing producer code on exiting thread is not a new P1-T wait; a nonreturning deleter violates producer contract. |
| a5 | C:939-1042 | CONFIRMED | Both lock retry sites diagnosed; fresh 15-failure CPU stagings each count diagnostic lines and quiesce. |
| a6 | C:939-1042 | CONFIRMED | Both lock retry sites diagnosed; fresh 15-failure CPU stagings each count diagnostic lines and quiesce. |
| a7 | C:1017-1057,1640-1658 | CONFIRMED | Timed wait and runner share unbounded diagnosed close; source and finite callback/lock witnesses pass. |
| a8 | C:1017-1057,1640-1658 | CONFIRMED | Timed wait and runner share unbounded diagnosed close; source and finite callback/lock witnesses pass. |
| a9 | K:391-428 | CONFIRMED WITH REASON CORRECTION | No producer/Python callback under registry mutex; snapshot does allocate a std::string, so the literal no-call rationale is too broad. Reporting OOM handled as V8-02; no new unbounded application-callback wait found. |
| a10 | K:463-488 | CONFIRMED | Storage drain is unchanged per-drop lifetime fence, not a wait introduced/called by exit hook. Physical GPU execution excluded. |
| a11 | O:206-215,289-296,309-324; C:1021-1033 | CONFIRMED IN STATED SCOPE | Lifecycle mutex released before diagnostics; leave uses short critical section; stderr channel backpressure itself is outside the promised diagnostic delivery guarantee. |
| a12 | O:206-215,289-296,309-324; C:1021-1033 | CONFIRMED IN STATED SCOPE | Lifecycle mutex released before diagnostics; leave uses short critical section; stderr channel backpressure itself is outside the promised diagnostic delivery guarantee. |
| missing-j-L1-event-calls | O:1052-1059,1066-1069,1094 | MISSING / DEFECT (MINOR) | Diagnostic marker create/record/query/destroy have no failure tags/records; four-case CPU census RED (V8-01). Query-error branch may also emit an unproved finished line. |
| missing-i-final-logger-OOM | O:1768-1770; C:1671-1672 | MISSING FAILURE CASE / DEFECT (MINOR) | Final logger message/snapshot allocation can fail after rich synchronize failure; empty catches keep a record with no later reader (V8-02). |
