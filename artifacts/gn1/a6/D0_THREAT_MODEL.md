# D0 threat model

Authority: the unchanged orders/GN1_A6_REV5_FIXES.md at order commit 1de3ed5b63addceb1a012deccb94fd8f2721b763, hashed in the seal. This explanatory copy does not amend the plan.

In scope: receipt/log edits, truncation, forgery and malformation; any staged argv change; substitutes/wrappers for the runner, timeout and interpreter, PATH-resolved programs, and differing PATH or FORGE_GPU_* values; interruptions produced by the installed runner and timeout.

Out of scope: coordinated replacement of every receipt and log together with the seal; root tampering with the installed runner between seal and run beyond the SHA-256 check; kernel process hiding; GPU/driver defects. Reviews classify matters outside D0 as observations.

The finite controls and exact consistency checks are the evidence class here. No formal authentication proof or GPU execution is claimed. Program identities are resolved and hashed from the actual child argv/PATH immediately before launch. The parent interpreter also has /proc/self/exe evidence. D0's installed-file race exclusion remains applicable.

**Known in-scope failure:** D0_MODE_PROMOTION.json accepts a CPU control root after59 receipt metadata edits, with seal/children/logs/programs/argv unchanged. D1 mode/environment provenance is not authenticated. Overall verdict RED-D0.

CPU control START and parent receipts explicitly declare control=true. The scorer recomputes all line targets and rejects root GPU acceptance while the recorded mode remains control=true. The receipt-only promotion defeats this untrusted declaration. Unaltered audit receipts also refuse acceptance; this is a recorded-mode check, not authenticated provenance. The two inherited full-root positives now check all-line sensitivity plus root rejection; this D1 adaptation was registered before replay. Exact staged argv use the installed runner, pinned timeout and pinned interpreter, with scratch ownership files and a child fixture that requires a control START. Production staging clears control mode.

D4 fallback: a failed full receipt write gets exactly one minimal error-receipt attempt, using the inherited write/fsync/link no-overwrite publication. A failed retry leaves the child start/outputs available but no parent completion receipt. The scorer's incomplete handling refuses that line and cannot credit a kill. This residual is deliberate and tested, not a receipt-publication guarantee under arbitrary filesystem failure.
