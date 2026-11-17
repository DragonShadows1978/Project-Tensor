# APA-P1-F1: fix the twelve P1-REV1 findings on the packed default (implementing round)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d`
(branch `apa-p1d`, at the commit carrying this order; P1-D's receipts are `f01201cf`). Edits, builds, CPU runs and GPU
runs through `forge-gpu-run` are AUTHORIZED in:
- `tensor_cuda/src/apa_spo/` (every file there);
- `tensor_cuda/src/bindings.cpp`, only inside the `_tensor_cuda_spo2` module (from `PYBIND11_MODULE(_tensor_cuda_spo2`
  to its closing brace) and the packed launch helpers it calls;
- `tensor_cuda/tensor_cuda/__init__.py`, only the packed regions `APA_SPO2_D1_BEGIN` to `APA_EB4_END` and the
  `_spo2` loader lines just above them;
- `tensor_cuda/tensor_cuda/apa_packed.py`;
- `tensor_cuda/tests/test_apa_p1_exact_tie.py`, **one edit only**: its `tc.apa_eb2b(...)` call (line 70) gains
  `delta=8.1, carry="row"`, the values the launch site hard-codes today. Nothing else in any existing test changes;
- new `tensor_cuda/tests/test_apa_p1f1_*.py`, new `scripts/apa_p1f1_*.py`, and `artifacts/apa_p1f1/` (new);
- scratch under `/tmp/apap1f1_*`. Point `TMPDIR` there for every command.

**Read-only:**
- every other file in this worktree, including `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/apa_sp2.cuh`,
  `tensor_cuda/CMakeLists.txt`, `tensor_cuda/tensor_cuda/apa_sp2.py`, and all of `artifacts/apa_p1d/` (Build B there is
  your before-build);
- `tensor_fork_apa_wt_apa2pprod` (REV1's findings and reproductions: `artifacts/apa_p1_rev1/findings.json`,
  `repro.py`, `cpu/`, `gpu/*/result.json`; S3's hooks build and staged lines under `artifacts/apa_p1_s3_r2/`);
- the production repository and every other worktree.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree and the named read-only paths only.

## Why

P1-D made packed SP keys the production default (`f01201cf`). The blind red team of the P1 port (P1-REV1, receipts
`08b36d09`) found 2 MAJOR, 6 MINOR and 4 NIT defects in the code that now carries that default. F01 is on the default
route itself: a NaN in q or in a key makes the packed public route (`r2_tq4`, fp32) return zeros where production's
dense SP returns NaN. Bugs are fixed, not waived. NITs are fixed once and closed with no further review round.

Read `findings.json` first. Each finding has its file and line, mechanism, reproduction and suggested fix. Where this
order and a suggested fix differ, this order binds.

## Lead rulings (binding)

- **R1 (F01, non-finite inputs).** All three merges guard only an exactly-zero denominator (`den != 0 ? acc/den : 0`):
  `merge.cuh` (`spo2::merge_kernel`), and the fused merges in `eb3.cuh` and `eb4.cuh`. A NaN therefore propagates as on
  the dense path. Every quantizer a packed entry owns (key codes, tq4 and int4, and the value quantizers that EB2b and
  EB4 use) carries a non-finite input element into that row's stored norm or scale, so its scores stay non-finite. The
  only exception is where the stored form already does this; show it from the code. On finite inputs nothing changes,
  bitwise (LF1).
- **R2 (F02, alignment).** Every launcher that reads an operand with a vector type refuses, before any launch, a base
  pointer not aligned to that width: 16 B for the uint4/float4 loads, 8 B for the uint2 loads. Census the operands per
  launcher from the code (REV1's list is a starting point, not the census). The refusal is a `ValueError` that names
  the entry, the operand and the required alignment. Routes that do not vector-load an operand keep accepting it
  unchanged, notably the dense production entries and the r2 public route. No silent aligned copy.
- **R3 (F03, carry defaults).** The public `apa_eb3`/`apa_eb4` wrappers and their bindings default to
  `carry="partition"`, as the other packed wrappers do. Row carry stays available by explicit argument.
- **R4 (F04, EB2b).** `apa_eb2b` takes `delta` and `carry` as required keyword arguments. They are validated like
  EB3's, with no hard-coded launch-site literal. `carry` accepts exactly the modes the EB2b kernel implements, stated
  from the code; an unsupported mode raises.
- **R5 (F05, tickets).** A test-hook timeout resets its merge-ticket slot to zero before it throws. The next fused merge
  on that slot behaves as in a fresh process.
- **R6 (F06).** `int4_stats` (and every packed host entry) validates rank, dtype, device and every divisor (KVH > 0,
  H % KVH == 0) before reading shapes or launching. Reuse `apa_spq1_buffer_check` where it fits.
- **R7 (F07, knobs).** In the default build (test hooks off), the A/B environment knobs are compiled out:
  `APA_FXTC1_VARIANT`, `TC_APA_SPQ1_REFERENCE`, and any other environment read that changes numerics or dispatch on a
  packed entry, censused from the code. Setting one to a non-default value then raises, naming the variable. It is never
  silently ignored. `TC_APA_SP` (the feature switch) is not a knob and stays. The hooks build keeps today's behaviour.
- **R8 (F08).** `ApaSpq1Workspace` uses the same exact-K identity check as `PackedSpKeys`: the source object, the append
  generation and the storage revision, through one shared helper. A non-append edit, or a different K, raises.
- **R9 (F09).** In a packed build, a failure to load the second initializer raises at package import, with the real
  cause. In a build without `TC_APA_PACKED`, the packed API raises a clear "built without TC_APA_PACKED" error before
  allocating anything.
- **R10 (F10).** Remove the dead first `_apa_spq1_entry` wrapper and its assignment, so the name has one definition.
- **R11 (F11).** Every packed entry validates everything, including `TC_APA_SP`, scale, shapes and G-range, before its
  first launch. Empty extents (B, H, S or KV = 0) raise as the dense entry does. EB2b's cursor check is `!=`. An
  unsupported `dots` value for the dtype raises instead of being ignored.
- **R12 (F12).** `eb3.cuh` stops compilation under `CUDA_API_PER_THREAD_DEFAULT_STREAM` with an `#error` that names the
  single-stream ticket contract.

## Sections (binding)

- **S1, implement R1 to R12.** Annotate prior art at each code site (Rules).
- **S2, builds.** Use P1-D's build script and path adapter (`scripts/apa_p1d_build.py`; import it, do not copy-edit it):
  - **Build F:** this round's tree, default configuration (hooks off).
  - **Build FH:** the same tree with the test hooks on, built the way S3's hooks build was
    (`tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_s3_r2/`; state the flag you used).
  - **Build B** is P1-D's, `artifacts/apa_p1d/build_B`, module sha256
    `8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e`. Check the sha before every use, and refuse on
    a mismatch.
- **S3, SASS.** Before building F, write the expected kernel differences into `artifacts/apa_p1f1/EXPECTED.json`: each
  kernel you will change or remove, with its finding. Then compare Build F against Build B per kernel, using P1-D's
  comparison method (`scripts/apa_p1d_sass.py`, the bijective `_cu_<hex>` suffix mapping only). Every differing or
  missing kernel must be on the expected list. An unlisted difference fails G7.
- **S4, CPU tests.** The engine cannot build a tensor without a GPU (HOST-TENSOR-NEEDS-DEVICE), so CPU tests use stubs
  where needed. Cover:
  - R3's and R4's signatures and defaults;
  - R8's identity rules;
  - R9's loader behaviour (a simulated load failure in a packed build; the clear error on a non-packed module);
  - R10;
  - R12, by preprocessing or compiling `eb3.cuh` standalone with and without the define.

  Also run the CPU G3 set (`artifacts/apa_p1_a1/g3_ids.tsv` in `tensor_fork_apa_wt_apa2pprod`) and P1-D's
  `test_apa_p1d_policy.py`, before (Build B tree, as in P1-D's arm B_v2) and after, per test ID.
- **S5, mutants** (validity of the new tests). Each is one change in a patched copy, rebuilt incrementally in a copy of
  Build F's or FH's tree, or compiled standalone from the patched source as `test_apa_p1_tickets.py` and
  `test_apa_p1_early_counter.py` do. Each runs against its named test or line and must fail it with the assertion named
  in `artifacts/apa_p1f1/mutants/MUTANTS.json`, which is written before any mutant runs:
  - `m01_spo2_merge_guard`: `merge.cuh` back to `denom>0` (killed by LF2);
  - `m02_eb3_merge_guard`: the EB3 fused merge back to `den>0.f` (LF2);
  - `m03_eb4_merge_guard`: the EB4 fused merge back to `den>0.f` (LF2);
  - `m04_int4_scale_nan_dropped`: the int4 key quantizer's non-finite carry removed (LF2);
  - `m05_alignment_check_passes`: the shared alignment check accepts everything (LF3);
  - `m06_eb3_default_row`: EB3's default carry back to `"row"` (S4);
  - `m07_eb2b_delta_ignored`: the EB2b launch site ignores the delta argument and uses 8.1 (LF4);
  - `m08_ticket_reset_dropped`: the timeout reset removed (LF5, hooks build);
  - `m09_int4_stats_unchecked`: `int4_stats`' divisor and rank checks removed (LF4, run in a child process);
  - `m10_knob_silently_ignored`: the default build ignores a non-default `APA_FXTC1_VARIANT` instead of raising (LF4);
  - `m11_workspace_identity_dropped`: `ApaSpq1Workspace` back to the row count only (S4).

  NIT fixes (R9 to R12) take no mutants.
- **S6, GPU lines, run by you** through `forge-gpu-run`, one at a time. Write their expectations into `EXPECTED.json`
  before the first line, and give its sha256 in the report. Before each line, do one dry run with
  `FORGE_GPU_DRY_RUN=1`, which returns 78.
  - **LF1, finite identity.** Build F against Build B, bitwise (output, selection masks and diagnostics), on finite
    inputs:
    - the P1-D L1 grid on the public route;
    - every raw packed entry (d1, r2 tq4/int4, tc, EB2b, EB3, EB4, SPQ4, SPQ6) on REV1's rule-line shapes
      (`gpu/rule`), with every carry and delta argument passed explicitly and equal on both sides. Where Build B's
      entry hard-wired a value, Build F passes that value.
  - **LF2, non-finite.** For each packed entry that reaches a merge, and production's dense SP entry, from one build in
    one process: a NaN in q, a NaN in k, a NaN in v, +Inf in q and +Inf in k, at REV1's positions (`gpu/nan`) and at
    one more key position each. A row is one (batch, head, query) output vector. It is non-finite if any element is.
  - **LF3, alignment.** For each launcher in R2's census and each vector-loaded operand, a DLPack import with a 2-byte
    base offset (REV1's `gpu/bounds` method):
    - The call raises R2's `ValueError` before any launch.
    - In the same process, the aligned call that ran before the refusal, run again after it, returns a bitwise-equal
      output. The context survives.
    - The dense production entry and the r2 public route, on the same misaligned operands, equal their aligned copy
      bitwise, as in REV1.
  - **LF4, refusals and knobs**, Build F:
    - REV1's `int4_stats` case (`gpu/bounds_rerun`) raises `ValueError`, and the child process exits 0;
    - empty extents raise;
    - an unset `TC_APA_SP` raises before any launch;
    - EB2b's delta argument changes its selected count on a case where 8.1 and the other value differ (state the case);
    - each R7 knob set to a non-default value raises, naming it;
    - unsupported `dots` and `carry` values raise.

    State how you show that nothing launched (P1-D's L2 method, or another stated one).
  - **LF5, tickets** (Build FH): force the test-hook timeout on one slot. Then:
    - that slot reads zero;
    - the next fused merge on it, in the same process, equals the same call in a fresh process, bitwise;
    - the hooks build's knobs behave as in S3.
  - **LF6, regression:** P1-D's L6 set (P150's 44 files), Build B against Build F. Per-ID outcomes are identical except
    a listed set of intended changes, each with its finding. R3's default change and R4's required arguments are the
    expected sources.

  Each line: `forge-gpu-run --min 30 --max <≤1800> --label p1f1-<line> -- timeout <≤1750>s ...`. Exit 76 (lock not
  acquired) means nothing ran. Retry unchanged after at least 5 minutes, at most 6 times, each recorded. Never set an
  NVML or driver library override.

## Gates

- **G1, finite identity:** LF1 is bitwise on every case, and both sides are informative.
- **G2, non-finite** (LF2):
  - For a NaN in q, k or v, each entry's set of non-finite rows equals the dense entry's set on the same tensors.
  - In every LF2 case, no row is all-zero in an entry where the dense row is non-finite.
  - **Registered prediction (G2b):** for +Inf in q or k, each entry's non-finite set equals the dense set. Reported as
    HELD or NOT HELD, with the rows that differ. A miss changes no code; the lead rules.
- **G3, alignment:** LF3 holds for every census row. The census is in `ALIGNMENT_CENSUS.json`, with each operand's
  vector width and load site (file:line).
- **G4, refusals:** LF4 holds, and S4's signature, identity and loader tests pass.
- **G5, tickets:** LF5 holds.
- **G6, NITs:** R9 to R12 are implemented, each with its test or check.
- **G7, SASS:** every kernel difference between Build F and Build B is on the expected list, and every listed kernel
  differs or is removed as stated.
- **G8, mutants:** 11 of 11 killed, each by its named assertion.
- **G9, no regression:** the CPU per-ID outcomes and LF6's outcomes are identical except the listed intended changes.
- **G10, scope:** writes only in the writable set. GPU only through `forge-gpu-run`. No git writes, nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G10 pass (G2b is reported either way). Then the lead verifies on disk, and
a blind Codex red team reviews P1-D and this round together. NITs are closed by this round.

## Mechanism & scope

- **Measures:** whether the twelve REV1 defects are closed on the code that ships as the packed SP default, without
  changing any finite-input result, and whether the new tests can detect each defect.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_rev1/findings.json`.
- **Exercises:** device and host code under `src/apa_spo/`, the spo2 bindings, the packed Python regions, two builds
  plus mutant builds, CPU tests, mutants, and six GPU lines (finite identity, non-finite, alignment, refusals, tickets,
  regression).
- **Does not cover:**
  - model quality;
  - speed;
  - the dense production SP path (read-only, the comparator);
  - multi-stream support (R12 refuses it at build time);
  - multi-GPU device mismatch (one GPU);
  - the bf16 raw EB3 route's own calibration;
  - DS1;
  - promotion.
- **Required evidence:**
  - the diff;
  - Build F, Build FH and the mutant build receipts, with module sha256s;
  - `EXPECTED.json` and its sha;
  - `SASS_COMPARISON.json`;
  - `ALIGNMENT_CENSUS.json`;
  - `KNOB_CENSUS.json` (R7);
  - CPU results and the per-ID before/after outcomes;
  - `mutants/MUTANTS.json` and `MUTANT_RESULTS.json`;
  - each GPU line's `RESULT.json`, log, start and exit files;
  - `REPORT.md`.
- **Source records (L81):** the findings and their repro methods come from REV1's `findings.json`, `repro.py` and
  `gpu/*/result.json`. The L1 grid and L6 set come from P1-D's `artifacts/apa_p1d/` receipts.
- **Operand census:** every vector-loaded operand of every packed launcher (R2), every environment read on a packed
  entry (R7), every merge site (R1), and every quantizer a packed entry owns (R1).
- **Baseline isolation:** Build B and every REV1/S3 receipt are references, never edited. New builds go under
  `artifacts/apa_p1f1/`.
- **Forbidden configuration:**
  - edits outside the writable set;
  - any change to a finite-input result;
  - an aligned copy in place of a refusal;
  - a knob honoured in the default build;
  - changing any existing test other than R4's one call;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, with at most 3
  build jobs. Start one only with `MemAvailable` at or above 12 GB. A full build compiles `kernels.cu` (P1-D: about 18
  minutes under load). A mutant rebuild in a copied tree recompiles only what changed.
- **Time:** every command under 30 minutes, except builds, which stay under 60. Long commands run detached and are
  waited on in the foreground with `timeout <n> tail --pid` (P1-D's D2). The round's ceiling is 12 hours of wall time,
  with GPU use at most 3 hours in total. If it cannot finish, stop and report `RED-BUDGET` with what was done.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** only through `forge-gpu-run`, one line at a time, as in S6.

## Registration checklist (order-lint)

- **L01:** the instruments exist, each checked by running it first:
  - P1-D's build script;
  - `cuobjdump`;
  - `nvcc --version`;
  - pytest;
  - `forge-gpu-run --status --min 30 --max 600`;
  - REV1's `repro.py --help` (or its import).
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. R2 refuses, and the forbidden list excludes an aligned copy.
- **L07:** G2b's prediction is registered here, before LF2 runs. Its consequence (no code change; the lead rules) is
  fixed here.
- **L08:** builds and receipts go under `artifacts/apa_p1f1/`, and scratch under `/tmp/apap1f1_*`.
- **L09:** identity is compared per output array with `np.array_equal` on same-width unsigned views, after dtype and
  shape checks. Test outcomes are compared per ID, and SASS per kernel. Logs and timings are never compared.
- **L10:** finite numerics must not change, so LF1 is bitwise (G1). The non-finite gate compares row sets, not values.
- **L12:** the inherited gates are P1-D's per-ID identity (L6 and the CPU G3 set) and its L1 bitwise equivalence.
  - R3 contradicts identity for any call that omits `carry` on EB3/EB4 and relied on row carry. Its replacement
    predicate is the same call with `carry="row"`, which must be bitwise equal to Build B's.
  - R4 contradicts identity for any EB2b call without `delta` and `carry`. The replacement is the same call with
    `delta=8.1, carry="row"`, bitwise equal to Build B's.
  - Every other pair applies unchanged.
- **L15:** n/a. No threshold is fitted.
- **L16:** the cheapest passing work would edit only the three merges and leave the int4 quantizer's NaN loss,
  pass LF2 on q-NaN cases, and add a refusal only where REV1 reproduced a fault. Excluded by:
  - LF2's k- and v-NaN cases, with row-set equality;
  - mutant `m04`;
  - R2's code census, which G3 checks row by row;
  - LF3 run per census row.
- **L17:** n/a. No ranking.
- **L18:** the shipped default stays packed, and the public route's carry (partition) is unchanged. R3 changes only the
  raw EB3/EB4 wrappers' default, from row to partition. REV1 F03 measured row carry refining a strict subset of the
  dense rule's keys.
- **L19:** every bitwise comparison checks first that each side is nonzero and non-constant, at full width. For LF2,
  each case shows that the dense side has at least one non-finite row and at least one finite row.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`. Your scripts must not
  contain a home-path literal; build paths from `Path.home()` or the environment.
- **L22:** the shapes and cases come from REV1's receipts and P1-D's L1/L6 receipts, named above.
- **L25:** bitwise comparisons put both sides on the same device, in the same process kind, from one build per side.
  LF2 compares entries within one build and one process.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool), with the GPU through `forge-gpu-run`. The red team that
  follows is Codex, so it is cross-model.
- **L38:** existing tests are not changed, except R4's one named call.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/apap1f1_*`.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json`, written before it runs.
- **L42:** see Machine rules.
- **L45:** each GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit, and the log's sha256. Exit 124 is a
  timeout, never a pass.
- **L47:** the mutants' expected failures come from this order's rules, never from this round's runs.
- **L48:** every GPU line runs as written, from the worktree root, through the real `forge-gpu-run`. Its receipt lists
  each post-device step with a terminal label. A harness exception is not a device result.
- **L55:** LF1's bitwise claim holds only if the merge guards differ from today's on non-finite denominators alone, and
  the quantizer carries touch only rows with a non-finite element. Show both from the code (the denominator is a sum of
  non-negative exponentials), then measure LF1.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty. GPU commands run only inside `forge-gpu-run`.
- **L60:** for each refusal you add, record in `BOUNDARY.json` the input, the refusal, and Build B's behaviour on the
  same input (fault, zero output, silent acceptance, or the same refusal).
- **L64:** reads are this worktree and the read-only paths above.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `artifacts/apa_p1f1/`. Stop if `/mnt/ForgeRealm` has under 40 GB free. Mutant tree copies
  count against that.
- **L72:** the write target is the named files and regions, `artifacts/apa_p1f1/`, and scratch.
- **L74:** plain wording. These are engine guards and their tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per line before the real run.
- **L20:** n/a. No parameter is picked or calibrated in this round. The "calibration" under Does not cover is the EB3
  route's own, which is out of scope.
- **L31:** n/a. No arm tests an alternative to an owner requirement. The packed default, the owner's requirement, is
  unchanged.
- **L37:** no knob is added. R7 removes knobs from the default build. R4's new EB2b arguments run from the Python
  wrapper (`__init__.py`, `APA_EB2B` region), through the spo2 binding (`bindings.cpp`), to the launcher (`eb2b.cuh`).
  All three are writable.
- **L103:** each build receipt (F, FH and every mutant) records the module sha256 and a sha256 map of every engine
  source file (`tensor_cuda/src/`, `tensor_cuda/include/`, `tensor_cuda/tensor_cuda/`) at build time. Each GPU line
  refuses, before the lease:
  - a module whose sha256 differs from its receipt;
  - for Build F and FH, a working tree whose source map differs from the receipt's.

  Build B is checked against `8728e67e…`. No copied file is named, except P1-D's build script, which is imported.
- **L82:** the writable product files are the ones listed at the top. `rg -n 'getenv' tensor_cuda/src/apa_spo` lists
  R7's environment reads. `rg -n 'uint4|uint2|float4' tensor_cuda/src/apa_spo` lists R2's candidate vector loads.
- **L35:** your registrations (`EXPECTED.json`, `MUTANTS.json`, the censuses) may fix cases, seeds, positions and
  hashes. They may not remove, narrow or reclassify anything a gate covers. If a clause cannot hold as written, stop
  and report RED with the reason. Only the lead amends a gate.
- **L99:** the rulings rest on these lead readings, each run before sealing:
  - `findings.json` lists F01 to F12, with the sites quoted in each ruling;
  - `rg` in this worktree found:
    - `carry="row"` at `__init__.py:1666,1684` and `bindings.cpp:2818,2834`;
    - the only existing test call to `apa_eb2b` at `test_apa_p1_exact_tie.py:70`;
    - environment reads at `spq6.cuh:240`, `launcher.cuh:93`, `eb3.cuh:104,864` and `eb4.cuh:680`;
    - the fused merges' `den>0.f` at `eb3.cuh:755-757`;
  - `pybind11_add_module` compiles `src/kernels.cu`, which includes the `apa_spo` headers. Build B's receipt shows
    about 1,070 s for the build step.

  Two premises are seat checks that run first:
  - that the tq4 key quantizer already carries a NaN into the norm (R1's exception);
  - the flag S3 used for its hooks build.

  If either fails, the item is handled as the ruling states. It does not stop the round.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md`, `/mnt/ForgeRealm/Project-Tensor/AGENTS.md`, `docs/APA.md`, REV1's
  `findings.json`, and P1-D's `artifacts/apa_p1d/REPORT.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout, and never start one while another is running.
- No deletes by you, in the leave-and-list form. Never kill a process you did not start.
- **The report:** try to write `artifacts/apa_p1f1/REPORT.md` early and keep it current. If the write is refused, say
  so; your final message is then the report, and the lead files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known:
  - IEEE 754 NaN propagation;
  - the online-softmax merge (Milakov and Gimelshein, 2018; FlashAttention, Dao et al., 2022);
  - CUDA's vector-load alignment rules (CUDA C++ Programming Guide);
  - mutation testing (DeMillo, Lipton and Sayward, 1978).

  Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G10, each PASS or FAIL, and G2b HELD or NOT HELD;
- the diff summary;
- every module sha256 and the SASS result against the expected list;
- the alignment and knob censuses (summaries);
- the CPU and per-ID results;
- the mutant table;
- each GPU line's result, with its retries;
- the deviations;
- every file written.
