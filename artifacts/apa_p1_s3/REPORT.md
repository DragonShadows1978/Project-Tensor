RED: SCOPE MISMATCH — stopped at the order's L82 rail before implementation.

S1 requires explicit streams through packing and quantization. Those host
launches are in read-only files:

- `tensor_cuda/src/apa_spo/packed_launcher.cuh:14,19,24,29,33,39,41`:
  rotation, key preparation and TQ4/INT4 packing have no stream parameter and
  launch on implicit stream 0.
- `tensor_cuda/src/apa_spo/eb2b.cuh:109,114`: V quantization likewise has no
  stream parameter and launches on implicit stream 0.

The packed predecessor route also reaches `launcher.cuh:5,98–148` if included
in S1. `parallel.cuh` contains kernel definitions; this preflight found no host
launch site there and establishes no need to edit it. Existing declarations in
`tensor_cuda/include/tc/ops.h:316–342` and forwarding in
`tensor_cuda/src/bindings.cpp:2757–2810` carry no stream. Binding edits are
currently authorized only for S2. Continuation needs an explicit scope
amendment covering the necessary host helpers and internal forwarding, or a
narrowed S1 requirement. No public API change is proposed here.

Evidence class: static source reading. `SOURCE_RECEIPTS.txt` records commands
and output; `SCOPE_STOP.json` records the boundary; `TICKET_BOUNDARY.json` is a
partial preflight census, not completed S1 evidence.

- G1 FAIL — ownership change, host unit test and concurrency staging not run.
- G2 FAIL — deterministic hook, CPU structural check and mutant staging not run.
- G3 FAIL — exact-tie suite integration, CPU fixture check and mutant staging not run.
- G4 FAIL — default-switch design and cross-repository caller census not run.
- G5 FAIL — no GPU lines staged; no dry runs; no exit-78 result claimed.
- G6 FAIL — no CPU build, before/after outcomes or SASS comparison run.
- G7 PASS — only listed artifact files written; no GPU, git, subagents,
  source edits, moves or deletions. Every shell command hid CUDA and used an
  explicit timeout. No concurrent commands or background waits.

Diff summary: ten new reporting/evidence files under `artifacts/apa_p1_s3/`;
zero engine or test edits. `ENGINE_CHANGE.diff` is empty. Changed-kernel list:
none. Source hashes of the inspected files remained equal during report
writing; this is not a binary/SASS or regression result.

Tests and CPU results: no tests written or executed. Build and instrument
checks were not run after the stop. Existing A1 receipts were read only and
are not substituted for this round's missing measurements.

Design summary and caller census: not produced after L82 stopped the round.
The local launch-boundary findings above are not the requested cross-repo
caller census. No other project's repository was entered.

Staged lines and dry runs: none. `STAGED_GPU_LINES.sh`, `EXPECTED.json` and
`DRYRUN.json` were not fabricated for a build that does not exist. The build
receipt, `SASS_DIFF.json`, `OUTCOMES_before.json`, `OUTCOMES_after.json` and
`DEFAULT_SWITCH_DESIGN.md` are likewise absent, explicitly uncompleted.

## Prior art

No algorithm implemented or newly proposed. The survey's inherited references
are NVIDIA CUDA Samples threadFenceReduction and the stream-ordered allocator
(2021); DeMillo, Lipton and Sayward (1978) mutation testing; and Batcher (1968)
stable sorting through the existing EB4 fixture (2026). Literature metadata
unverified — lead to check those names and years. No code site changed.

Every file written, relative to `artifacts/apa_p1_s3/`:

- `REGISTRATION.md`
- `LEDGER.md`
- `SOURCE_RECEIPTS.txt`
- `SCOPE_STOP.json`
- `TICKET_BOUNDARY.json`
- `SOURCE_HASHES.json`
- `ENGINE_CHANGE.diff`
- `FILES_WRITTEN.md`
- `VERIFICATION.json`
- `REPORT.md`

No scratch, builds or files needing removal were created. No fix is claimed.
