# ENG1-A2 execution amendments

1. Before a completed trace exists: initial group A exited 1 during pytest
   plugin initialization, before collecting tests. The unchanged write fence
   allowed the relocated trace scratch directory, while inherited TMPDIR was
   its sibling. Verbatim final error: `FileNotFoundError: [Errno 2] No usable
   temporary directory found in [...]`. Full traceback is preserved in
   `q2_readers_A.stderr`; parent exit is in `q2_readers_A.exit.json`.
   Route TMPDIR under the same permitted trace scratch root in the A2 adapter.
   No change to copied REV2 code, fence semantics, pytest plugins, selectors,
   thresholds or acceptance. Retry with a new supervisor label; no completed
   trace output was overwritten. REGISTRATION.md remains unchanged.

2. Audit startup refused before child launch: `TMPDIR must resolve to /tmp/eng1_*`. The original scratch guard requires the direct parent /tmp, so /tmp/eng1_a2_work/tmp is rejected. Preserved the entire refused run root as audit_preflight_refused/ and its supervisor files under audit_preflight_refused.* (moves only, no deletes). Retry at fresh audit/ with environment TMPDIR=/tmp/eng1_a2_audit. No code, seal, pin, evaluator or threshold changes; the initial seal stays byte-identical.
