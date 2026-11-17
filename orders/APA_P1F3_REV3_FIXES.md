# APA-P1-F3: fix the three P1-REV3 findings on the packed default (narrow implementing round)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2`
(branch `apa-p1f2`, fast-forwarded to REV3's `73e8dc7c`, at the commit carrying this order). Edits, builds, CPU runs
and GPU runs through `forge-gpu-run` are AUTHORIZED in:
- `tensor_cuda/src/apa_spo/` (every file there);
- `tensor_cuda/src/bindings.cpp`, only inside the `_tensor_cuda_spo2` module and the packed launch helpers it calls;
- `tensor_cuda/tensor_cuda/apa_packed.py`;
- new `tensor_cuda/tests/test_apa_p1f3_*.py`, new `scripts/apa_p1f3_*.py`, and `artifacts/apa_p1f3/` (new);
- scratch under `/tmp/apap1f3_*`. Point `TMPDIR` there for every command.

**Read-only:**
- every other file in this worktree, including every existing test, `tensor_cuda/tensor_cuda/__init__.py`,
  `apa_sp2.py`, `tensor_cuda/include/`, `kernels.cu`, and all of `artifacts/apa_p1f2/`, `artifacts/apa_p1_rev3/` and
  earlier rounds' folders (Build F2, your before-build, is `artifacts/apa_p1f2/arms/F2`);
- every other worktree and the production repository.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree only.

## Why

P1-F2 (`bc9e6f5e`) closed REV2's eight findings. The narrow Codex review of it, P1-REV3 (receipts `73e8dc7c`), found
three more:
- **P1R3-F02 (MAJOR).** A finite key row whose squared norm overflows fp32 stores an Inf norm, and the packed default
  can then return a finite row where dense SP is non-finite. The lead confirmed it on the GPU with every input finite
  (`artifacts/apa_p1_rev3/lead_p1rev3_gpu_repro.json`).
- **P1R3-F01 (MINOR).** R1's pin validates a snapshot, then `lookup` re-reads the caller's mapping.
- **P1R3-F03 (MINOR).** EB2b forms B*H in signed int before its range check. Attention extents above INT_MAX raise a
  generic RuntimeError instead of R3's operand-naming ValueError.

Read `artifacts/apa_p1_rev3/findings.json` and `REPORT.md`, and F2's order and report, first. Where this order and a
suggested fix differ, this order binds.

## Lead rulings (binding)

- **The non-finite contract (unchanged):** the packed route never hides a row that dense SP makes non-finite. For a NaN
  in q, k or v, the row sets are equal. A row whose stored key norm is non-finite, for any reason, is non-finite on the
  packed route.
- **R1′ (P1R3-F01).** After `check_pinned_margins` passes, `lookup` and `packed_attention` use the module's canonical
  pinned values (`PACKED_PINNED_ENTRIES`). They never read the caller's mapping again. A hand-built object whose
  mapping changes between reads therefore gets the pinned value at use, or a refusal.
- **R2′ (P1R3-F02).** `spq1::prepare` stores a quiet NaN norm for any key row whose computed norm is non-finite,
  including a norm that overflows from finite elements. Today it does this only for rows with a non-finite element
  (`packed.cuh:94-108`; its comment says the overflow case keeps Inf, which this ruling reverses). Rows whose computed
  norm is finite are unchanged, bitwise (LF1).
- **R3′ (P1R3-F03).**
  - EB2b computes B*H, and every other host product that is range-checked, in int64 before the check and before any
    narrowing (`eb2b.cuh:158-159` and any like site; census them).
  - A packed attention extent above INT_MAX raises a `ValueError` that names the entry and the operand, before any
    launch (R3's contract).
  - A zero or negative extent keeps F1's RuntimeError (dense parity, F2's D2, accepted).
- Nothing else changes.

## Sections (binding)

- **S1, implement R1′, R2′ and R3′.** Annotate prior art at each code site.
- **S2, builds.** Use F2's build script (`scripts/apa_p1f2_build.py`; import it, do not copy-edit it).
  - **Build F3:** this round's tree, default configuration (hooks off), built from **this worktree's** sources, as
    Build F2 was. Build F2's sources compiled from this same worktree path (`artifacts/apa_p1f2/build_F2/BUILD.json`
    `source_root`).
  - **Build F2** (`artifacts/apa_p1f2/arms/F2`, module sha256
    `6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579`) is the before-build. Check its sha before every
    use, and refuse on a mismatch.
- **S3, SASS (L09 as extended 11-16).** Before the first Build F3, write `artifacts/apa_p1f3/EXPECTED.json`. It holds:
  - each kernel you will change, add or remove, with its ruling;
  - every non-code source of difference between Build F2 and Build F3, with how it is removed or mapped:
    - the source path: equal, because both builds compile this worktree's sources, so nvcc's path-derived names
      (`_GLOBAL__N__<8 hex>_`, `__nv_static_<n>__<8 hex>_`) must be equal;
    - the build directory: the bijective `_cu_<8 hex>` plus optional `_<n>` suffix mapping, as F1 and F2 used it;
    - any other source you find.

  Compare Build F3 against Build F2 per kernel with `scripts/apa_p1f1_sass.py` (after F2's R7). `EXPECTED.json` is
  written once. Every later F3 build is compared against the same file, and you never amend it after any F3 build
  exists.
- **S4, CPU tests** (stubs where needed). Cover:
  - R1′: a hand-built object whose mapping returns the pinned values to validation and other values afterwards. At
    use, the recording backend receives the pinned e_q, or the call is refused. Also a loaded pinned table, unchanged;
  - R3′: EB2b's B*H product at and one past its bound, and an attention extent above INT_MAX. Each raises its
    `ValueError` (or F1's RuntimeError for non-positive extents) before any launch. Use host-level fakes or DLPack
    re-imports with large extents, as F2's LF3 did, and record which;
  - the per-ID CPU set: F2's CPU G3 set, plus `test_apa_p1d_policy.py`, `test_apa_p1f1_cpu.py` and
    `test_apa_p1f2_cpu.py` (as REV3's step T left them), before (this branch as committed, Build F2) and after (Build
    F3).
- **S5, mutants** (validity of the new tests). Each is one change in a patched copy, run against its named test or
  line, and must fail it with the assertion named in `artifacts/apa_p1f3/mutants/MUTANTS.json`, written before any
  mutant runs:
  - `m01_lookup_rereads_mapping`: `lookup` returns `self.entries[...]` again (S4, R1′);
  - `m02_overflow_norm_kept`: `prepare` keeps the Inf norm for an overflowing finite row (LF2, overflow cases);
  - `m03_eb2b_product_int32`: EB2b's B*H back to a signed int product (S4 or LF3).
- **S6, GPU lines, run by you** through `forge-gpu-run`, one at a time. Import F2's line machinery
  (`scripts/apa_p1f2_gpu.py`) where it fits. Write the lines' expectations into `EXPECTED.json` before the first line.
  Do one dry run with `FORGE_GPU_DRY_RUN=1` (returns 78) before each line.
  - **LF1, finite identity:** Build F3 against Build F2, bitwise, on F2's LF1 cases unchanged.
  - **LF2, non-finite:** F2's LF2 cases unchanged, plus overflow cases. Each overflow case is one key row with a
    finite element large enough that its squared norm overflows fp32 (REV3's r01: 1e20 at D64), at D64 and D128, with
    both signs of q. Run them on every packed entry that reads TQ4 keys, the public default route and dense SP. Gates
    as F2's G2.
  - **LF3, refusals** (R3′), where a GPU line is needed, with F2's launch-count method.
  - **LF6, regression:** F2's LF6 set, Build F2 against Build F3.
  - **MUT:** the device and binding mutants against their lines.

  Each line: `forge-gpu-run --min 30 --max <≤1800> --label p1f3-<line> -- timeout <≤1750>s ...`. Exit 76 means nothing
  ran: retry unchanged after at least 5 minutes, at most 6 times, each recorded. Never set an NVML or driver library
  override.

## Gates

- **G1, finite identity:** LF1 is bitwise on every case, and both sides are informative.
- **G2, non-finite:** LF2 holds, as F2's G2 defined it:
  - NaN row-set equality for NaN in q, k or v;
  - containment in every case;
  - no all-zero row where dense is non-finite.

  The overflow cases are included. Every case is informative (L19).
- **G3, refusals:** S4's R1′ and R3′ tests pass, and LF3 holds where run. Every new refusal is in `BOUNDARY.json`, with
  Build F2's behaviour on the same input.
- **G4, SASS:** every kernel difference between Build F3 and Build F2, after the registered mappings only, is on the
  expected list, and every listed kernel differs as stated.
- **G5, mutants:** 3 of 3 killed, each by its named assertion.
- **G6, no regression:** the CPU per-ID outcomes and LF6's outcomes are identical, except a listed set of intended
  changes, each with its ruling (registered in `EXPECTED.json`).
- **G7, scope:** writes only in the writable set. GPU only through `forge-gpu-run`. No git writes, nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G7 pass. Then the lead verifies on disk, and a narrow cross-model (Codex)
review covers this diff. QG0-R runs on Build F3.

## Mechanism & scope

- **Measures:** whether REV3's three findings are closed on the code that ships as the packed SP default, without
  changing any finite-norm result, and whether the new tests can detect each defect.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/artifacts/apa_p1_rev3/findings.json`.
- **Exercises:** `spq1::prepare`, EB2b's and the attention entries' host checks, `apa_packed.py`'s lookup and
  attention paths, Build F3 plus two mutant builds, CPU tests, mutants, and the GPU lines above.
- **Does not cover:**
  - model quality (QG0-R follows);
  - speed;
  - the dense production SP path (read-only, the comparator);
  - REV2's and F2's other items (closed or reviewed);
  - packed prefill (P1-RES, a temporary limitation);
  - promotion.
- **Required evidence:**
  - the diff;
  - Build F3 and the mutant build receipts, with module sha256s;
  - `EXPECTED.json` and its sha;
  - `SASS_COMPARISON_F3.json`;
  - `PRODUCT_CENSUS.json` (R3′'s range-checked host products);
  - `BOUNDARY.json`;
  - CPU results and per-ID before/after;
  - `mutants/MUTANTS.json` and `MUTANT_RESULTS.json`;
  - each GPU line's `RESULT.json`, log, start and exit files;
  - `REPORT.md`.
- **Operand census:** every host product that is range-checked in `src/apa_spo/` (R3′); `prepare`'s norm store
  (R2′); every read of the margin mapping in `apa_packed.py` (R1′).
- **Baseline isolation:** Build F2 and every earlier receipt are references, never edited. New builds go under
  `artifacts/apa_p1f3/`.
- **Forbidden configuration:**
  - edits outside the writable set;
  - any change to a finite-norm result;
  - restoring NaN masking for q, k or v;
  - an `EXPECTED.json` amended after an F3 build exists;
  - changing any existing test;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`. Start one only
  with `MemAvailable` at or above 12 GB. A full build compiles `kernels.cu` (about 5 to 18 minutes). Mutant rebuilds in
  a copied tree recompile only what changed, as F2's did.
- **Time:** every command under 30 minutes, except builds (under 60). Long commands run detached and are waited on in
  the foreground with `timeout <n> tail --pid`. The round's ceiling is 6 hours of wall time, with GPU use at most 1.5
  hours in total. If it cannot finish, stop and report `RED-BUDGET` with what was done.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free (121 GB at sealing).
- **GPU:** only through `forge-gpu-run`, one line at a time.

## Registration checklist (order-lint)

- **L01:** the instruments exist, each checked by running it first: F2's build script (import), `cuobjdump --version`,
  `nvcc --version`, pytest `--version`, `forge-gpu-run --status --min 30 --max 600`, and F2's `scripts/apa_p1f2_gpu.py`
  and F1's `scripts/apa_p1f1_sass.py` (import).
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. R1′ and R3′ refuse or use the pinned constant; R2′ carries NaN.
- **L07:** n/a. No prediction is registered.
- **L08:** builds and receipts go under `artifacts/apa_p1f3/`, and scratch under `/tmp/apap1f3_*`.
- **L09:** identity is compared per output array with `np.array_equal` on same-width unsigned views, after dtype and
  shape checks. Test outcomes are compared per ID, SASS per kernel. Logs and timings are never compared. The build
  comparison names its non-code sources of difference before the first build (S3).
- **L10:** finite-norm numerics must not change, so LF1 is bitwise (G1). The non-finite gate compares row sets.
- **L12:** the inherited gates are F2's LF1 (bitwise), LF2 (row sets) and the per-ID identity of LF6 and the CPU sets.
  R2′ contradicts F2's comment that an overflowing finite row keeps its Inf norm. LF1's cases have finite norms, so
  LF1's predicate is unchanged. Every other pair applies unchanged.
- **L14:** n/a. There is no gain gate.
- **L15:** n/a. No threshold is fitted.
- **L69:** one reference per operand class:
  - kernel outputs on finite inputs -> Build F2 (G1, LF1); excluded from G2, which compares row sets within one build;
  - test outcomes -> F2's tree per ID (G6), excluding the listed intended changes;
  - kernel code -> Build F2's SASS after the registered mappings (G4).

  "Nothing else changes" in the rulings is checked by these three and the diff, not by a further identity gate.
- **L16:** the cheapest passing work would special-case the one REV3 instance (D64, 1e20), check only EB2b's B*H, and
  snapshot only `lookup` while `packed_attention` re-reads. Excluded by:
  - LF2's overflow cases at both D, both signs, on every TQ4 reader;
  - `PRODUCT_CENSUS.json`, checked row by row;
  - S4's stateful-mapping test, run at `lookup` and at attention;
  - mutants `m01` to `m03`.
- **L17:** n/a. No ranking.
- **L18:** the shipped default stays packed, and every finite-norm result is unchanged.
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width. For LF2,
  each case shows that the dense side has at least one non-finite row (or declares it all-finite, as F2 did) and at
  least one finite row.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`. Your scripts must not
  contain a home-path literal.
- **L22:** the cases come from F2's LF1, LF2 and LF6 receipts and REV3's `repro/r01_finite_norm_overflow.py`.
- **L25:** bitwise comparisons put both sides on the same device, in the same process kind, from one build per side.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L32:** n/a. No bounded buffer is added or changed. F2's G4 filled `PackedSpKeys`' capacity, and R3′ changes no
  capacity rule.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool), because the round builds, runs GPU lines and kills device
  mutants. The review that follows is Codex, so it is cross-model.
- **L35:** your registrations (`EXPECTED.json`, `MUTANTS.json`, the census) may fix cases, seeds, positions and
  hashes. They may not remove, narrow or reclassify anything a gate covers. If a clause cannot hold as written, stop
  and report RED with the reason. Only the lead amends a gate. A rebuild never re-registers anything (S3).
- **L37:** no knob is added.
- **L38:** existing tests are not changed.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/apap1f3_*`.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json`, written before it runs. Show that each
  mutant's killing test executes the mutated line (a counter or log line with the mutant loaded).
- **L45:** each GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256. Exit 124 is a
  timeout, never a pass.
- **L47:** the mutants' expected failures come from this order's rules, never from this round's runs.
- **L48:** every GPU line runs as written, from the worktree root, through the real `forge-gpu-run`.
- **L55:** LF1's bitwise claim holds only if R2′ changes the stored norm only on rows whose computed norm is
  non-finite. R1′ and R3′ must change only host checks and the margin value source. Show both from the code, then
  measure LF1.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty. GPU commands run only inside `forge-gpu-run`.
- **L60:** for each refusal you add, record in `BOUNDARY.json` the input, the refusal, and Build F2's behaviour on the
  same input.
- **L64:** reads are this worktree only.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `artifacts/apa_p1f3/`. Stop if `/mnt/ForgeRealm` has under 40 GB free.
- **L72:** the write target is the named files, `artifacts/apa_p1f3/`, and scratch.
- **L74:** plain wording. These are engine guards and their tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per line before the real run.
- **L82:** the writable product files are `src/apa_spo/`, the spo2 region of `bindings.cpp`, and `apa_packed.py`.
  `rg -n 'numeric_limits<int>' tensor_cuda/src/apa_spo` starts R3′'s census.
- **L99:** the rulings rest on these lead readings, each run before sealing:
  - `packed.cuh:94-108` stores NaN only for `nonfinite` rows, and its comment keeps Inf for an overflowing finite row;
  - `apa_packed.py:440-446`: `lookup` calls `check_pinned_margins(self)` and then returns
    `self.entries[f"{bulk_bits}:{dim}"]`;
  - `eb2b.cuh:158-159`: `int B=...,H=...` and then `B*H>std::numeric_limits<int>::max()/4` in int;
  - `launcher.cuh:30-34` and `:132-134` raise `std::runtime_error` for extents above INT_MAX;
  - REV3's GPU repro confirmed the overflow case on Build F2;
  - Build F2's `BUILD.json` `source_root` is this worktree.

  Seat check that runs first: that `apa_packed.py`'s `packed_attention` reaches the margin values only through
  `lookup`. If it reads them another way too, R1′ covers that read as well.
- **L102:** no generated file, registry or ledger row depends on the writable files.
- **L103:** each build receipt (F3 and every mutant) records the module sha256 and a sha256 map of every engine source
  file at build time. Each GPU line refuses, before the lease, a module whose sha256 differs from its receipt, and, for
  Build F3, a working tree whose source map differs from the receipt's. Build F2 is checked against `6f03b2cd…`.
- **L104:** no ratio or fraction bar. Every gate is exact: bitwise per output array, row-set equality or containment per
  (batch, head, query) row per case, outcome per test ID or per kernel; slack 0 = 0 events. The containment boundary
  is covered by mutant `m02`.
- **L105:** n/a. No arm is compared with a non-reference baseline by a margin.
- **L106:** the gates are this order's lines. F2's line code is imported unchanged where reused, and its inputs (Build
  F2 and F2's receipts) exist in this worktree (lead `ls` of `artifacts/apa_p1f2/arms/F2`).
- **L107:** a "new or changed files" list compares sha256 against a base map taken at round start, never mtime. It is
  report-only.
- **L109:** your helpers run as `python3 -B -P` (or `-I`). Before the final message, list every `.py` you wrote and
  confirm none shares a standard-library module name.
- **L111:** n/a. No storage design is emulated.
- **Staged-script lesson (REV2):** any script you stage for a later GPU run first runs on CPU up to its first device
  call, with a receipt.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md`, `/mnt/ForgeRealm/Project-Tensor/AGENTS.md`, `docs/APA.md`, REV3's `findings.json`
  and `REPORT.md`, and F2's order and report first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout, and never start one while another is running.
- No deletes by you, in the leave-and-list form. Never kill a process you did not start.
- **The report:** try to write `artifacts/apa_p1f3/REPORT.md` early and keep it current. If the write is refused, say
  so; your final message is then the report, and the lead files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known: IEEE 754 NaN and overflow propagation;
  TurboQuant (Zandieh et al., 2025); time-of-check to time-of-use (CWE-367); ISO C++17 integer conversion; mutation
  testing (DeMillo, Lipton and Sayward, 1978). Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G7, each PASS or FAIL;
- the diff summary;
- every module sha256 and the SASS result against the expected list, with the non-code sources you registered;
- the product census (summary);
- the CPU and per-ID results;
- the mutant table;
- each GPU line's result, with its retries;
- the deviations;
- every file written.
