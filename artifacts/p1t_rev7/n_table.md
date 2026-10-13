| Item | Verdict | Reason / evidence |
|---|---|---|
| N1 | CORRECTLY NOT A DEFECT | Producer-initiated pthread_exit/cancellation violates returning C deleter contract; independent finite callback is V7-01, not cancellation. O:533-547; C:861-868; vendored dlpack.h:397. |
| N2 | CORRECTLY NOT A DEFECT | A nonreturning producer prolongs documented exit-only quiescence. Unbounded waiting is necessary; V7-01 is a different accidental skipped wait. |
| N3 | CORRECTLY NOT A DEFECT | Ordinary count wait removed. Native exit closes admission before counting, so later releases back out. O:217-227,1119-1123. |
| N4 | CORRECTLY NOT A DEFECT | Explicit unregister deliberately disables the documented hook; finalizing-entry checks remain. Repro V7-01 leaves hooks registered. |
| N5 | CORRECTLY NOT A DEFECT | No P1-T host function enqueued. App callback/app lock order remains app responsibility; lead CUDA host-function citation checked; stream waits are required for memory safety. |
| N6 | CORRECTLY NOT A DEFECT | Accept lead correction: global FIFO lets another calling thread run a deleter; finalizers may run under arbitrary app locks. O:837-839,997-1004; README:157-160; Python finalizer docs. |
| N7 | CORRECTLY NOT A DEFECT | Environment/performance limits are evidence boundaries, not implementation defects. Lead historical mutant/249-case/baseline claims independently audited; no new GPU run claimed. |
| N8 | CORRECTLY NOT A DEFECT | With lead correction: undeclared free-thread support reenables GIL; own-GIL interpreters reject single-phase modules; legacy shared-GIL mixing with PyGILState is already unsupported in baseline core. No free-thread/subinterpreter certification. |
| N9 | CORRECTLY NOT A DEFECT | Former R7 was a defect and is fixed by deferred raise/ack; K:255-275,391-417; C:1466-1493; CI:1371-1383. Not a waiver. |
