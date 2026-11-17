# APA-P1-D: packed storage becomes the SP default (implementing round), with packed margins measured and frozen

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d`
(branch `apa-p1d`, at the commit carrying this order). Edits, CPU builds, CPU runs and GPU runs through
`forge-gpu-run` are AUTHORIZED in:
- `tensor_cuda/tensor_cuda/apa_sp2.py`; a new `tensor_cuda/tensor_cuda/apa_packed.py`; and, in
  `tensor_cuda/tensor_cuda/__init__.py`, only the APA packed-binding region (`APA_SPQ1_BEGIN` to `APA_SPQ4_END`) and
  the `apa_sp2` import/export lines;
- `tensor_cuda/src/bindings.cpp`, `tensor_cuda/src/apa_spo/packed_launcher.cuh` and `tensor_cuda/src/apa_spo/tensor.cuh`,
  for **host-side** validation, device guards and typed adapters only. No device code changes anywhere;
- `tensor_cuda/CMakeLists.txt`, for the `TC_APA_PACKED` default only;
- `tensor_cuda/tests/test_apa_sp2.py` (add cases; every existing case stays, unchanged), and new
  `tensor_cuda/tests/test_apa_p1d_*.py`;
- `scripts/apa_sp2_gpu.py` and `scripts/apa_pidv.py`, only to pass `storage="dense"` explicitly at their quantizer
  calls; new `scripts/apa_p1d_*.py`;
- `artifacts/apa_p1d/` (new): builds, receipts, registrations, tables, mutants and the report;
- scratch under `/tmp/apap1d_*`. Point `TMPDIR` there for every command.

**Read-only:**
- every other file in this worktree, including all of `tensor_cuda/src/apa_spo/` except the two files named above, and
  `kernels.cu`;
- `tensor_fork_apa_wt_apa2pprod` (a blind red team of the P1 port runs there now; read its S3 design, builds and
  receipts only);
- the production repository `/mnt/ForgeRealm/Project-Tensor` (read its `artifacts/apa_sp2/e_q_table.json`, sha256
  `5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d`, the dense margin table);
- every other worktree, and the other projects' repositories (GRAPA, GraftRepository, Project-Voice), which the caller
  census only reads.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside named repo roots only.

## Why

The owner's 11-02 direction: APA memory back to the two-pass (2P) level. P1 ported the lab's packed SP path onto
production as an opt-in (`d41cf833`, A1 `f7e6f49b`, S3 r2b `58ef3ed2`; GPU lines all as registered: `cdb7d21a`,
`daef0291`). Production SP stores a dense fp32 reconstruction of the 4-bit keys; packed storage keeps the codes. On the
bf16 raw layout that is 768 → 580 bytes per key per KV head (2P-INT4 parity); on the public fp32 route it is 1,536 →
1,092. Orchestrator rule 1: the owner's requirement ships as the default, not as an opt-in.

The design is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_s3_r2/DEFAULT_SWITCH_DESIGN.md`
(§(a)–(d); the raw census tables after line 135). This round implements §(c) with the rulings below. Where this order
and the design differ, this order binds.

This branch is P1 (`daef0291`) merged with production main `2478a4d` (ENG1, ENG2) in `e2fb85a1`; the merge had no
conflicts.

## Lead rulings on the design's open points (binding)

- **R-A, the policy.** `quantize_sp2_keys(k, bulk_bits, *, storage=None)`. Precedence: the explicit argument, then
  `TC_APA_SP_STORAGE`, then the shipped default `"packed"`. Accepted values are exactly `"packed"` and `"dense"`; any
  other value raises `ValueError` before any device work.
- **R-B, ineligible inputs.** Packed storage covers 4-bit bulk at D64/D128 on the public fp32 route. Under the default
  (no explicit request), an ineligible input (2-bit bulk) takes the dense representation and records the fallback
  reason. An explicit `storage="packed"` that cannot be honoured raises. Nothing changes 2 bits to 4.
- **R-C, the packed object.** `PackedSpKeys`, an owned object (not a tuple, not a Tensor subclass), per §(c): codes,
  norms, rotation, codebook, B/KV/D, capacity and visible rows, dtype, device ordinal, bulk bits, quantizer and table
  provenance, and the exact-K identity (object identity plus an append generation). It may be built on
  `ApaSpq1Workspace` (`__init__.py`, `APA_SPQ1_BEGIN`). Append is explicit and validated. It owns no K-shaped dense
  tensor.
- **R-D, the public wrapper.** `apa_selective_attention_sp(q, k, kq, v, ...)` keeps its signature. A `PackedSpKeys` in
  `kq` takes the packed route:
  - bulk bits match;
  - `k` is the object's exact-K source, with visible rows equal to `k.shape[2]`;
  - `margins` is the packed table. A dense table with a packed object raises, and so does a packed table with dense Kq;
  - delta comes from the same upward-rounded rule as the native epsilon entry (`apa_sp2.cuh:3-22`). One shared helper;
    its results equal the native derivation on registered boundary fixtures.

  A dense Tensor in `kq` keeps today's behaviour exactly. The fp32 public route uses the ordered packed-r2 path
  (`gqa_path="r2"`), never a cast into EB3.
- **R-E, prefill (L > 1) with packed keys.** If the transient dense reconstruction of a `PackedSpKeys` is **bitwise
  equal** to `quantize_sp2_keys(k, 4, storage="dense")` on the same keys, a prefill call reconstructs transiently
  (released after the call), uses the dense table, and reports the route. If it is not bitwise equal, a prefill call on
  packed keys raises with a message naming `storage="dense"`. G6 measures which branch holds; you implement that
  branch. No persistent second representation either way.
- **R-F, packed margins.** A packed table with its own quantizer identity, entries `4:64` and `4:128`, measured by
  SP2's G2 procedure (`scripts/apa_sp2_gpu.py`: the same `data()` generator, seeds, shapes, classes and statistic, for
  the 4-bit targets), with the bulk scores taken from the packed-r2 arithmetic. Import the procedure's pieces from the
  script and do not copy-edit them. Your measurement script runs under `forge-gpu-run`. It never calls the script's
  `require_lease()` or `scripts/apa_sp2_lead_gpu.sh`, and never sets a driver-library override. Register the
  procedure (`artifacts/apa_p1d/MARGIN_REGISTRATION.json`, sha256 in the report) before the first measurement. Freeze
  it the way SP2's `freeze()` does: max, p99.9, mean and count per entry, then an upward-rounded `e_q`. Pin the sha in
  code. Load it through a packed loader that applies `RegisteredMargins.load`'s checks to its own registration,
  quantizer identity and two entries. The dense loader and table stay unchanged. The table is a measurement, not a
  choice: nothing is re-selected after any quality result.
- **R-G, build.** `TC_APA_PACKED` defaults to `ON`. The S3 test hooks stay off in the default build.
- **R-H, observability.** The route taken (storage, backend, fallback reason) is readable without a new attention
  output schema, through the existing `diagnostics` path or an internal receipt. Tests read it.
- **R-I, callers.** Re-run S3's census commands (`artifacts/apa_p1_s3_r2/CALLER_CENSUS_FINAL_COMMANDS.md` in
  `tensor_fork_apa_wt_apa2pprod`) at this commit. Give every current-source row (S3: 80) a disposition:
  - unchanged and correct under the new default;
  - explicit `storage="dense"` (a control that needs the dense Tensor). This applies to the two quantizer callers;
  - or historical/frozen (artifacts and references: listed, never edited).

  No current caller may change representation silently. A packed object passed into a raw dense `_C` entry raises.

## Sections (binding)

- **S1, implement R-A to R-I.** Annotate prior art at each code site (Rules).
- **S2, two builds,** one Release/sm89 build each, using the A1 build script's path adapter
  (`tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_a1/BUILD.py`, `BUILD_PATH_ADAPTER.diff`):
  - **Build A (before):** commit `e2fb85a1` with `-DTC_APA_PACKED=ON`, hooks off.
  - **Build B (after):** this round's tree in its default configuration, with no `-D` flags.
- **S3, SASS.** Build B against Build A: every kernel is identical, because this round changes no device code. Also,
  closing S3's open item: Build A against A1's build (`artifacts/apa_p1_a1/`). List any differing kernel, each with
  its cause.
- **S4, CPU tests.** The engine cannot build a tensor without a visible GPU (open item HOST-TENSOR-NEEDS-DEVICE), so
  CPU tests use a stub for `_C` and tensors, and cover the policy logic:
  - precedence and refusals (R-A);
  - the fallback rule (R-B);
  - the packed object's fields and ownership (R-C);
  - table and object matching (R-D);
  - delta helper boundary fixtures against the native rule, re-derived independently in the test;
  - route reporting (R-H).

  Also run production's existing CPU suite before and after, per test ID (P1's G3 set, `artifacts/apa_p1_a1/g3_ids.tsv`).
- **S5, mutants** (§(d) negative controls). Each is a patched copy under `artifacts/apa_p1d/mutants/<name>/`, run
  against the named test, and expected to fail with a named assertion:
  - `epsilon_as_delta`;
  - `dense_table_on_packed`;
  - `two_bits_to_four`;
  - `stale_packed_keys` (append generation not checked);
  - `causal_offset_dropped`;
  - `sinks_dropped`;
  - `dense_kq_retained` (the packed object keeps a dense copy);
  - `env_ignored`;
  - `explicit_packed_falls_back`.
- **S6, GPU lines, run by you** through `forge-gpu-run` once their expectations are written (`artifacts/apa_p1d/EXPECTED.json`,
  sha256 in the report, written before the first GPU line). One line at a time:
  - **L1, default and opt-out.** On the same tensors, Build B's default equals its explicit packed call, bitwise
    (output and selected mask). Build B's dense opt-out, by argument and by environment, equals Build A's public dense
    route, bitwise. Cover D64/D128, B1 and B2, H = KV and every GQA group the packed-r2 path supports, S
    255/256/257/2048, causal on and off, and sinks on and off.
  - **L2, refusals and fallbacks** on the device: each R-A/R-B/R-D refusal raises before launch; 2-bit default takes
    dense with its reason; 2-bit explicit packed raises; prefill per R-E.
  - **L3, R-E's reconstruction check:** bitwise comparison on 4-bit keys at D64/D128, S 255/2048, B1/B2. It decides
    R-E's branch. Run it before you implement R-E's prefill path.
  - **L4, bytes.** For the packed default on Qwen3-1.7B and Qwen3-4B decode geometries (KV 8, D128) at S 2,048, 8,192
    and 32,768:
    - the object's owned buffers sum to 68 B per key per KV head plus the fixed tables;
    - it owns no K-shaped tensor;
    - own-process device memory after construction is at or below the dense representation's by the predicted
      difference, within the allocator's granularity (state it).
  - **L5, packed margins** (R-F): the registered 4-bit `eq_` targets of SP2's G2 procedure, then the freeze. SP2's
    sweep targets (its G3 epsilon curve) are not part of this round. Where a class samples prefill queries, score
    each sampled query as an L = 1 decode call on the packed path. The per-pair bulk error depends only on the query
    and the key, so it is the same quantity.
  - **L6, production GPU regression:** P150's 44 files (`artifacts/apa_p1_a1/STAGED_LINES.json`) on Build A and Build B.
    Outcomes per ID are identical, except a listed set of intended changes, each with its reason.

  Each line: `forge-gpu-run --min 30 --max <≤1800> --label p1d-<line> -- timeout <≤1750>s ...`. Exit 76 (lock not
  acquired) means nothing ran: retry unchanged after at least 5 minutes, at most 6 times, each recorded. Never set an
  NVML or driver library override.

## Gates

- **G1, policy:** S4's precedence, refusal and fallback tests pass. L1 and L2 hold.
- **G2, the object:** R-C's fields exist. L4 shows no K-shaped owned tensor, and the byte sum.
- **G3, the wrapper:** R-D's matching rules pass on the CPU and the device. The delta helper equals the native rule on
  every boundary fixture.
- **G4, callers:** R-I's census is re-run at this commit with commands and output pasted. Every current-source row has
  a disposition.
- **G5, margins:** the procedure is registered before measurement. Both entries are measured and frozen, with the sha
  pinned in code. The per-entry max, p99.9, mean and count are reported beside the dense table's.
  - **Registered prediction (G5b):** each packed `e_q` is within 1% (relative) of the dense table's entry, because
    quantization error dominates both. A miss is reported as NOT HELD. It changes no code, and the lead rules before
    the next round.
- **G6, prefill:** L3 is run, and R-E's matching branch is implemented and tested.
- **G7, SASS:** Build B equals Build A on every kernel. Build A against A1 is listed.
- **G8, mutants:** all nine are killed, each by its named assertion.
- **G9, no regression:** per-ID CPU outcomes, and L6's GPU outcomes, are identical except the listed intended changes.
- **G10, scope:** writes only in the writable set. GPU only through `forge-gpu-run`. No git writes, nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G10 pass (G5b is a prediction, reported either way). Then the lead
verifies on disk, then the model-quality round (registered by the lead from AB-1's metrics, against the frozen packed
table), then a blind Codex red team over this round's diff, then the promotion ticket.

## Mechanism & scope

- **Measures:** whether production's SP entry can default to packed key storage with the old dense behaviour intact
  behind an explicit opt-out; whether the packed object owns only codes, norms and tables; whether the packed route is
  governed by its own measured margins; and whether anything else in the engine changes.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_s3_r2/DEFAULT_SWITCH_DESIGN.md`.
- **Exercises:** the Python policy, the packed object and wrapper, host-side guards, two builds, CPU tests, mutants,
  a caller census, and six GPU lines (equivalence, refusals, reconstruction, bytes, margins, regression).
- **Does not cover:**
  - model quality (next round);
  - speed;
  - EB3/EB4 tickets and kernels (S3's, unchanged here);
  - the bf16 raw EB3 route's own calibration;
  - DS1's pitched cache (its port follows this one);
  - multi-stream use;
  - the red team;
  - promotion.
- **Required evidence:**
  - the diff;
  - both build receipts and module sha256s;
  - `SASS_COMPARISON.json`;
  - CPU test results and the per-ID before/after outcomes;
  - mutant results;
  - `CALLER_DISPOSITIONS.json` with the census commands;
  - `MARGIN_REGISTRATION.json`, the frozen packed table and its sha;
  - `EXPECTED.json`, then each GPU line's `RESULT.json` and log;
  - `REPORT.md`.
- **Source records (L81):** R-A to R-I come from the design §(c) with these rulings. The margin procedure comes from
  `scripts/apa_sp2_gpu.py`, and the dense table from production's `artifacts/apa_sp2/`. The 580 / 1,092 byte figures
  come from the design's storage table.
- **Operand census:** every current-source caller (R-I), every public and raw entry the wrapper reaches, and every
  buffer the packed object owns.
- **Baseline isolation:** Build A and the A1 build are references, never edited. Build B goes under
  `artifacts/apa_p1d/`.
- **Forbidden configuration:**
  - device code changes;
  - changing any existing test case;
  - a default other than packed;
  - silent fallback on an explicit request;
  - re-selecting margins after a quality result;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, with at most 3
  build jobs. Start one only with `MemAvailable` at or above 12 GB. The machine is shared and loaded, so expect slow
  builds.
- **Time:** every command under 30 minutes, except builds, which stay under 60. The round's ceiling is 8 hours of wall
  time, with GPU use at most 3 hours in total. If it cannot finish, stop and report `RED-BUDGET` with what was done.
- **GPU:** only through `forge-gpu-run`, one line at a time, as in S6.

## Registration checklist (order-lint)

- **L01:** the instruments exist, each checked by running it first: the A1 build script and path adapter,
  `cuobjdump`, pytest, `forge-gpu-run --status --min 30 --max 600`, and `scripts/apa_sp2_gpu.py list`.
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** R-B's fallback is gated (G1): each fallback records its reason, and the explicit request never falls back
  (mutant `explicit_packed_falls_back`).
- **L07:** G5's prediction is registered before L5 runs. Its consequence (no code change; the lead rules) is fixed
  here.
- **L08:** builds and receipts go under `artifacts/apa_p1d/`, and scratch under `/tmp/apap1d_*`.
- **L09:** identity is compared per output array with `np.array_equal` on same-width unsigned views, after dtype and
  shape checks. Test outcomes are compared per ID, and SASS per kernel. Logs and timings are never compared.
- **L10:** n/a for G1 and G7: no numerics change on the dense route, so its outputs are bitwise against Build A. The
  packed route's numerics are P1's, unchanged, and its quality is the next round.
- **L15:** n/a. No threshold is fitted. G5's 1% is a stated prediction, not a tuned bar.
- **L16:** the cheapest passing work would flip the default flag and leave the wrapper accepting a packed object
  without checking it against its source K or table. G3's matching rules, the mutants `stale_packed_keys` and
  `dense_table_on_packed`, and L2 exclude that. A default that silently falls back to dense everywhere is excluded by
  L1 (default equals explicit packed, bitwise) and L4 (bytes).
- **L17:** n/a. No ranking or selection among arms.
- **L18:** this round changes the shipped default (packed) as the owner's requirement, with an explicit opt-out.
- **L19:** every bitwise comparison checks first that each side is nonzero and non-constant, at full width.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`.
- **L22:** the margin targets come from SP2's script unchanged. Shapes in L1 to L4 are fixed here. The P150 set and
  the G3 IDs come from A1's receipts.
- **L25:** bitwise comparisons put both sides on the same device, in the same process kind, from one build per side
  as named.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool), with the GPU through `forge-gpu-run`. The red team that
  follows is Codex, so it is cross-model.
- **L38:** existing tests are never changed. New cases go in new functions or new files.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/apap1d_*`.
- **L41:** each mutant names its expected failing assertion in `artifacts/apa_p1d/mutants/MUTANTS.json`, written
  before it runs.
- **L42:** see Machine rules.
- **L47:** the mutants' expected failures come from this order's rules, never from this round's runs.
- **L48:** every GPU line runs as written, from the stated cwd, through the real `forge-gpu-run`. Its receipt lists
  each post-device step with a terminal label. A harness exception is not a device result.
- **L55:** Build B's dense route reproduces Build A's bitwise only if the dense code path is untouched. Show that from
  the diff (the dense branch of the wrapper and every `_C` dense entry unchanged), then measure it in L1.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty. GPU commands run only inside `forge-gpu-run`.
- **L60:** for each host-side guard you add, census its boundary: device ordinal versus the operands' device, dtype,
  rank, extents, visible rows versus capacity, and append generation. Record it in `BOUNDARY.json` with the baseline
  (Build A) behaviour.
- **L64:** reads are this worktree, the read-only paths above, and the other repos for R-I only.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `artifacts/apa_p1d/`. Stop if `/mnt/ForgeRealm` has under 40 GB free.
- **L72:** the write target is the named files, `artifacts/apa_p1d/`, and scratch.
- **L74:** plain wording. This is an engine default and its tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per line before the real run.
- **L12:** the inherited gates are P150's per-ID outcome identity (G9) and SP2's G2 statistic (G5). The new state is
  the packed default.
  - SP2's statistic applies unchanged: only the arithmetic that produces the bulk score changes.
  - P150's identity is contradicted by the new default for any test that asserts the quantizer returns a dense Tensor
    under no argument. For those IDs only, the replacement predicate is the same assertion run under
    `storage="dense"`, which must hold. Each such ID is listed with its reason.
  - Every other pair applies unchanged.
- **L20:** no parameter is picked as the most aggressive value that passes. `e_q` is the maximum over all registered
  valid scores, upward-rounded, the same conservative statistic as SP2. Its adequacy for model quality is the next
  round's registered check.
- **L35:** your registrations (`MARGIN_REGISTRATION.json`, `EXPECTED.json`, `MUTANTS.json`) may fix cases, seeds,
  predictions and hashes. They may not remove, narrow or reclassify anything a gate covers. If a clause cannot hold as
  written, stop and report RED with the reason. Only the lead amends a gate.
- **L99:** the rulings rest on these lead readings, each run before sealing:
  - `git merge-tree --write-tree --name-only daef0291 origin/main` exited 0 with no conflicted paths, and the merge
    `e2fb85a1` completed with no conflicts.
  - `rg -n 'apa_selective_attention_sp|quantize_sp2_keys|_apa_spq[0-9]_entry' tensor_cuda/tensor_cuda/` found only
    `apa_sp2.py` (the definitions) and `__init__.py` (exports and the packed wrappers). There is no SP cache builder
    inside the package.
  - `git diff 0e1dea3 daef0291 -- tensor_cuda/CMakeLists.txt` shows `option(TC_APA_PACKED ... OFF)`.
  - In `scripts/apa_sp2_gpu.py`, the modes are `list`, `next`, `summary`, `freeze`, `worker` and `validate`.
    `validate` only decodes a target name. Measurement calls `require_lease()`, which needs the old lead wrapper's fd 9.
  - S3's census gives two current quantizer callers (`scripts/apa_pidv.py:269,303`, `scripts/apa_sp2_gpu.py:53`).

  Two premises are seat checks that run first:
  - whether the packed path is decode-only (L = 1);
  - R-E's bitwise reconstruction (L3).

  Either failing defers its item, with a report. It does not stop the round.
- **L82:** the writable product files are the ones listed at the top. `rg -n -e quantize_sp2_keys -e
  apa_selective_attention_sp -e ApaSpq1Workspace tensor_cuda/tensor_cuda` lists the in-package sites, and R-I's census
  lists the rest.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md`, `/mnt/ForgeRealm/Project-Tensor/AGENTS.md`, `docs/APA.md`, the design note, and
  P1's rulings (`tensor_fork_apa_wt_apa2pprod/lead/LEAD_RULING_A1_GPU.md`, `lead/LEAD_RULING_S3_GPU.md`) first. This
  order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout, and never start one while another is running.
- No deletes by you, in the leave-and-list form. Never kill a process you did not start.
- **The report:** write `artifacts/apa_p1d/REPORT.md` early and keep it current. Repeat it in your final message.
- **Prior Art Directive:** annotate each code site and the report. Known: TurboQuant (Zandieh et al., 2025) for the
  codes; BLASST (2025) for the running-max criterion; SP2's two-error margin argument (inherited); PyTorch-style
  owned storage handles for the object. Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G10, each PASS or FAIL, and G5b HELD or NOT HELD;
- the diff summary;
- both module sha256s and the SASS result;
- the CPU and per-ID results;
- the mutant table;
- the caller dispositions summary;
- the packed table (entries, sha) beside the dense table's entries;
- each GPU line's result, with its retries;
- R-E's branch;
- every file written.
