# APA-P1-S3 registration

Registered UTC: 2026-11-03T21:30:10.452290+00:00.
Immutable plan: orders/APA_P1_S3_PROMOTION_ITEMS.md.
Plan SHA-256: 902153c024de297c4fc855957419b99c293fb036ebc0460c3c26c419623d69f0.

The order's G1–G7 criteria and acceptance threshold are adopted without change.
The pre-registered L82 stop condition applies if explicit stream propagation
reaches read-only APA launch helpers. Do not edit those helpers or continue
implementation/builds/gates after that finding. Record the scope evidence and
report instead. No amendment to the order is made here.

Before this registration, only instruction, memory, mechanism, order, source
and existing-receipt reads ran. No build, test, instrument gate, GPU staging,
GPU query, source edit, git command, subagent, deletion or move ran.
The read-only preflight identified the L82 stop condition. This registration
does not claim to precede that discovery; it precedes every gate, none of
which will run in this stopped round.

Execution: synchronous foreground commands under explicit timeouts;
CUDA_VISIBLE_DEVICES is empty. All new files remain in artifacts/apa_p1_s3/.
Build/pytest/cuobjdump/forge-gpu-run instrumentation checks are NOT RUN because
of the stop. No permission to extend the writable set is inferred.
