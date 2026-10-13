| Item | Verdict | Reason / evidence |
|---|---|---|
| U1 | ACCEPTABLE | Ordinary sync no longer waits on another thread; release stays on that thread. Correctly removes caller-lock inversion. O:1354-1367; fresh lock/late probes. |
| U2 | ACCEPTABLE | DLPack typed protocol permits refusing conversion-only objects/numpy bool/int scalars; exact values/subclasses read safely. C:542-598; CPU census. |
| U3 | ACCEPTABLE | All new native P1-T parameters use C-level matching/read/refusal. Public wrappers convert in Python. No existing baseline binding rebinding; 1067 CPU checks. |
| U4 | ACCEPTABLE | Retain drain error through reporting OOM and raise on retry; O:1363-1433; three trace probes pass. |
| U5 | ACCEPTABLE | Core deferred errors now have a public reporting path and exit logging; C:1466-1493; K:391-417; CI:1371-1383. Semantics intentional; V7-01 is separate hook reachability. |
| U6 | ACCEPTABLE | Close before waiting admits no new callbacks; later releases deliberately leak at process exit. Native mechanism correct; open V7-01 must be fixed for Python hook to reach it. |
| U7 | ACCEPTABLE | Consumed-capsule destructor remains producer code; admit last reference or keep after close. O:324-354. No manager double-release. |
| U8 | ACCEPTABLE | Raw CPython wait functions retain no dispatcher-owned Python objects across GIL release. C:1383-1457; A:352-379; py_args.h:203-248. |
| U9 | ACCEPTABLE | Two bounded one-shot test hooks permit real-source regression testing; defaults unarmed, allocation-free flag consumption. O:1031-1033,1048-1049,1263-1264. |
