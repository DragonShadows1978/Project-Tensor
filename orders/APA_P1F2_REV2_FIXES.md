# APA-P1-F2: fix the eight P1-REV2 findings on the packed default (implementing round)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2`
(branch `apa-p1f2`, at the commit carrying this order; its code is P1-F1's `710cc0d4`, plus REV2's receipts
`71c77cb7`). Edits, builds, CPU runs and GPU runs through `forge-gpu-run` are AUTHORIZED in:
- `tensor_cuda/src/apa_spo/` (every file there);
- `tensor_cuda/src/bindings.cpp`, only inside the `_tensor_cuda_spo2` module (from `PYBIND11_MODULE(_tensor_cuda_spo2`
  to its closing brace) and the packed launch helpers it calls;
- `tensor_cuda/include/tc/ops.h`, **one edit only**: the stale 14-argument `tc::apa_eb2b_run` declaration (line 328)
  becomes the current definition's signature (`eb2b.cuh:138`);
- `tensor_cuda/tensor_cuda/__init__.py`, only the packed regions `APA_SPO2_D1_BEGIN` to `APA_EB4_END` and the `_spo2`
  loader lines just above them;
- `tensor_cuda/tensor_cuda/apa_packed.py`;
- `tensor_cuda/tensor_cuda/apa_sp2.py`, **only the function `quantize_sp2_keys`** (R8). The dense quantizer, the
  dense `RegisteredMargins` and every other line stay as they are;
- `scripts/apa_p1f1_sass.py`, for R7 only;
- new `tensor_cuda/tests/test_apa_p1f2_*.py`, new `scripts/apa_p1f2_*.py`, and `artifacts/apa_p1f2/` (new);
- scratch under `/tmp/apap1f2_*`. Point `TMPDIR` there for every command.

**Read-only:**
- every other file in this worktree, including `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/apa_sp2.cuh`,
  `tensor_cuda/CMakeLists.txt`, every existing test, and all of `artifacts/apa_p1d/`, `artifacts/apa_p1f1/` and
  `artifacts/apa_p1_rev2/`;
- `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d` (Build F, your before-build:
  `artifacts/apa_p1f1/arms/F` and `build_F_r2`; F1's GPU line receipts under `artifacts/apa_p1f1/gpu/`);
- `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1q` (P1-Q's append check,
  `scripts/apa_p1q_harness.py:235-260`);
- `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod` (REV1's receipts and
  the CPU G3 ID list);
- the production repository and every other worktree.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree and the named read-only paths only.

## Why

P1-D made packed SP keys the production default; P1-F1 (`710cc0d4`) closed REV1's twelve findings. The blind Codex red
team of both (P1-REV2, receipts `71c77cb7`) returned RED: 2 MAJOR, 4 MINOR, 2 NIT. The lead ran its three staged GPU
reproductions, and all three confirm (`artifacts/apa_p1_rev2/lead_p1rev2_gpu_repros.json`). The worst: on the default
route, a +Inf or -Inf in one key element makes the packed row finite where dense SP returns NaN. Bugs are fixed, not
waived. NITs are fixed once and closed, with no further review round for them.

Read `artifacts/apa_p1_rev2/findings.json` and `REPORT.md` first. Each finding has its file and line, mechanism and
suggested fix. Where this order and a suggested fix differ, this order binds.

## Lead rulings (binding)

- **The non-finite contract.** The packed route must never hide a row that dense SP makes non-finite. It may add
  non-finite rows where an input or an intermediate score is non-finite or overflows (the loud direction). For a NaN
  in q, k or v, the row sets are equal (F1's G2, unchanged). NaN masking of q, k or v is never restored.
- **R1 (F01, margin pin).** A `PackedRegisteredMargins` whose `sha256` or entries differ from the pinned table's
  (`PACKED_TABLE_SHA256`, the file `artifacts/apa_p1d/packed_e_q_table.json`) is refused before any device work, at
  `lookup` and at `packed_attention`. A hand-built instance carrying the pinned sha with other entries is refused too.
  `load()` on the public path cannot override the pin. If you keep a route for unpinned experimental tables, it is a
  separate type that `packed_attention` refuses. The dense `RegisteredMargins` is not changed (its caller-supplied sha
  is inherited behaviour; existing scripts load their own tables through it).
- **R2 (F02, non-finite TQ4 keys).** Every TQ4 key quantizer a packed entry owns stores NaN as the norm (or scale) of
  any key row with a non-finite input element, including a norm made infinite by an Inf element. Census the TQ4 key
  quantizers from the code (`packed.cuh` `prepare`/`pack_tq` is the public route's; check `eb3.cuh`, `spq4.cuh`,
  `spq6.cuh` and `packed_launcher.cuh` for their own or shared ones). On rows whose elements are all finite, nothing
  changes, bitwise (LF1).
- **R3 (F04, capacity range).** Every packed host entry and `PackedSpKeys`/`ApaSpq1Workspace` refuses, before any
  allocation or launch, a capacity, extent or required index product that does not fit the native index type that
  receives it. Census those types per launcher from the code (`packed_launcher.cuh`, `launcher.cuh` and the bulk
  launchers). The refusal is a `ValueError` naming the operand. Widening the indexing to 64 bits instead is allowed
  only where it changes no finite result (LF1 still holds).
- **R4 (F05, validation order).** Raw `packed_rotate`, `packed_prepare` and `packed_append`, and every other raw spo2
  helper (censused), refuse zero or negative leading extents on the host with a `ValueError` before any launch. EB3
  refuses `timeline=True` with a G it has no timeline template for (today G 3, 5, 6, 7 and 8) before any allocation or
  launch.
- **R5 (F06, the finite-identity claim).** No device numerics change for these cases. The merge comments
  (`merge.cuh:14-30` and the fused merges in `eb3.cuh` and `eb4.cuh`) and your report state the claim narrowly: the
  guard preserves every result whose intermediate scores and denominator are finite. Finite operands that overflow
  (FLT_MAX scale), NaN or +Inf sinks, and ±Inf queries over all-negative logits follow IEEE propagation, the loud
  direction. LF2 pins these cases.
- **R6 (F07).** The `ops.h` declaration matches the definition, so a native caller that includes `ops.h` links to the
  one `tc::apa_eb2b_run`. NIT.
- **R7 (F08).** `scripts/apa_p1f1_sass.py` records the expectation file it actually compared against (path and
  sha256) next to the original registration, and counts paired suffix renames separately from added functions. F1's
  filed receipts are not rewritten. NIT.
- **R8 (F09).** `quantize_sp2_keys(k, bulk_bits, *, storage=None, capacity=None)`. When the storage resolves to
  packed, `capacity` is forwarded to `PackedSpKeys`, with R3's bound. With dense storage, a capacity other than None
  raises `ValueError`. With `capacity=None`, every result is as today.

## Sections (binding)

- **S1, implement R1 to R8.** Annotate prior art at each code site (Rules).
- **S2, builds.** Use P1-D's build script and path adapter (`scripts/apa_p1d_build.py`; import it, do not copy-edit
  it), as F1 did (`scripts/apa_p1f1_build.py` shows how):
  - **Build F2:** this round's tree, default configuration (hooks off).
  - **Build F2H:** only if the round changes code compiled only with the test hooks on. If not, state that from the
    diff and skip it.
  - **Build F** is F1's, in the p1d worktree (`artifacts/apa_p1f1/arms/F`), module sha256
    `5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243`. Check the sha before every use, and refuse on
    a mismatch.
- **S3, SASS.** Before the first Build F2, write `artifacts/apa_p1f2/EXPECTED.json`: each kernel you will change, add
  or remove, with its ruling. Compare Build F2 against Build F per kernel with `scripts/apa_p1f1_sass.py` after R7 (the
  bijective `_cu_<8 hex>` plus optional `_<n>` suffix mapping only). `EXPECTED.json` is written once. Every later
  build of F2 (for example after a code fix) is compared against the same file. You never amend it after any F2 build
  exists.
- **S4, CPU tests.** The engine cannot build a tensor without a GPU (HOST-TENSOR-NEEDS-DEVICE), so CPU tests use stubs
  where needed. Cover:
  - R1: a hand-built table (wrong sha; pinned sha with other entries; `load()` with another sha) is refused before
    any device call;
  - R3: capacities at and one past the bound (refusal before allocation, by a stub that records allocation);
  - R8: the forwarded capacity, the dense refusal and the `capacity=None` default;
  - R6: a standalone compile of one translation unit that includes `ops.h` and `eb2b.cuh` and takes
    `&tc::apa_eb2b_run` with no target type. It fails to compile on the base tree (two overloads) and compiles after
    R6. Record both;
  - R7: the SASS script on a small synthetic pair of dumps, counting renames and additions apart.

  Also run the CPU G3 set (`artifacts/apa_p1_a1/g3_ids.tsv` in `tensor_fork_apa_wt_apa2pprod`),
  `test_apa_p1d_policy.py` and `test_apa_p1f1_cpu.py`, before (this branch's tree as committed) and after, per test ID.
- **S5, mutants** (validity of the new tests). Each is one change in a patched copy. Python-only mutants run against
  Build F2's module. Device or binding mutants are rebuilt incrementally in a copy of Build F2's tree. Each runs
  against its named test or line and must fail it with the assertion named in `artifacts/apa_p1f2/mutants/MUTANTS.json`,
  written before any mutant runs:
  - `m01_margin_pin_unchecked`: `lookup` and `packed_attention` accept any `PackedRegisteredMargins` (S4);
  - `m02_tq4_inf_norm_kept`: the public route's TQ4 quantizer stores the infinite norm unchanged (LF2, k ±Inf);
  - `m03_capacity_unbounded`: R3's bound removed from `PackedSpKeys` (S4);
  - `m04_raw_extent_unchecked`: R4's positive-extent check removed from `packed_rotate` (LF3);
  - `m05_eb3_timeline_g_late`: R4's EB3 timeline/G pre-check removed (LF3, by its launch count);
  - `m06_factory_capacity_dropped`: `quantize_sp2_keys` ignores `capacity` (S4 and LF4).

  NIT fixes (R6, R7) take no mutants.
- **S6, GPU lines, run by you** through `forge-gpu-run`, one at a time. Import F1's line machinery
  (`scripts/apa_p1f1_gpu.py`) where it fits, rather than copying it. Write the lines' expectations into
  `EXPECTED.json` before the first line, and give its sha256 in the report. Before each line, do one dry run with
  `FORGE_GPU_DRY_RUN=1`, which returns 78.
  - **LF1, finite identity.** Build F2 against Build F, bitwise (output, selection masks and diagnostics), on finite
    inputs. Use F1's LF1 cases unchanged: the P1-D L1 grid on the public route, and every raw packed entry on REV1's
    rule-line shapes, with every carry and delta argument passed explicitly and equal on both sides.
  - **LF2, non-finite.** For each packed entry that reaches a merge, the public default route (`storage=None`) and
    production's dense SP entry, from one build in one process:
    - NaN, +Inf and -Inf in q, in k and in v, at F1's LF2 positions and at one more key position each;
    - a NaN sink and a +Inf sink, on each entry that takes sinks;
    - FLT_MAX scale on finite operands (REV2's `r02` case);
    - +Inf and -Inf in q over all-negative logits (REV2's `r01` cases);
    - REV2's `r01` default-route cases, unchanged.

    A row is one (batch, head, query) output vector. It is non-finite if any element is.
  - **LF3, refusals**, Build F2:
    - the raw helpers on each zero leading extent raise R4's `ValueError` with no CUDA error;
    - in the same process, a valid control call made before the refusals is run again after them and returns a
      bitwise-equal output;
    - EB3 with `timeline=True` at G 3, 5, 6, 7 and 8 raises before any launch, shown by F1's launch-count method
      (`apa_p1f1_gpu.py` `launches()`), and at G 2 and 4 it still runs;
    - a hand-built margin table on the public route raises before any launch;
    - a capacity past R3's bound raises before any allocation (device memory in use is unchanged).
  - **LF4, append through the factory.** `quantize_sp2_keys(k, 4, capacity=N)`, then `append` one token at a time,
    against a per-step rebuild of the same keys, at D64 and D128: codes, norms and route output bitwise, by P1-Q's
    method (`apa_p1q_harness.py:235-260`). Use real-looking keys from a fixed seed, at least 64 steps per D. Fill
    the object exactly to capacity N, then show that one more append raises before any launch and leaves the object
    unchanged (codes, norms, visible rows).
  - **LF6, regression:** F1's LF6 set (P150's 44 files), Build F against Build F2. Per-ID outcomes are identical
    except a listed set of intended changes, each with its ruling.
  - **MUT:** the device and binding mutants against their lines.

  Each line: `forge-gpu-run --min 30 --max <≤1800> --label p1f2-<line> -- timeout <≤1750>s ...`. Exit 76 (lock not
  acquired) means nothing ran. Retry unchanged after at least 5 minutes, at most 6 times, each recorded. Never set an
  NVML or driver library override.

## Gates

- **G1, finite identity:** LF1 is bitwise on every case, and both sides are informative.
- **G2, non-finite** (LF2):
  - for a NaN in q, k or v, each entry's set of non-finite rows equals the dense entry's set on the same tensors;
  - in every LF2 case, each entry's set of non-finite rows contains the dense entry's set (never hides);
  - in every LF2 case, no row is all-zero in an entry where the dense row is non-finite;
  - rows non-finite in an entry but finite in dense are listed per case (allowed; the loud direction).
- **G3, refusals:** LF3 holds, and S4's R1, R3 and R8 tests pass. Every refusal row is in `BOUNDARY.json` (L60).
- **G4, append:** LF4 is bitwise on every step at both D.
- **G5, NITs:** R6 and R7 are implemented, each with its check from S4.
- **G6, SASS:** every kernel difference between Build F2 and Build F is on `EXPECTED.json`'s list, and every listed
  kernel differs, is added or is removed as stated.
- **G7, mutants:** 6 of 6 killed, each by its named assertion.
- **G8, no regression:** the CPU per-ID outcomes and LF6's outcomes are identical except the listed intended changes.
- **G9, scope:** writes only in the writable set. GPU only through `forge-gpu-run`. No git writes, nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G9 pass. Then the lead verifies on disk, a narrow cross-model (Codex)
review covers this round's diff, and P1-Q's QG0 rule reruns on Build F2 (QG0-R). NITs are closed by this round.

## Mechanism & scope

- **Measures:** whether the eight REV2 defects are closed on the code that ships as the packed SP default, without
  changing any finite-input result, and whether the new tests can detect each defect.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/artifacts/apa_p1_rev2/findings.json`.
- **Exercises:** device and host code under `src/apa_spo/`, the spo2 bindings, the packed Python regions,
  `quantize_sp2_keys`, one `ops.h` declaration, F1's SASS script, Build F2 plus mutant builds, CPU tests, mutants, and
  five GPU lines (finite identity, non-finite, refusals, append, regression) plus the mutant line.
- **Does not cover:**
  - model quality (QG0-R follows);
  - speed;
  - the dense production SP path and its `RegisteredMargins` (read-only, the comparator);
  - multi-stream support (refused at build time since F1);
  - multi-GPU device mismatch (one GPU);
  - packed prefill (refused; a later item, P1-RES);
  - column-level non-finite parity of the int8 V entries (REV2's withdrawn F03, a recorded behaviour);
  - DS1;
  - promotion.
- **Required evidence:**
  - the diff;
  - Build F2 (and F2H if built) and the mutant build receipts, with module sha256s;
  - `EXPECTED.json` and its sha;
  - `SASS_COMPARISON.json`;
  - `TQ4_CENSUS.json` (R2), `INDEX_CENSUS.json` (R3), `HELPER_CENSUS.json` (R4);
  - `BOUNDARY.json`;
  - CPU results and the per-ID before/after outcomes;
  - `mutants/MUTANTS.json` and `MUTANT_RESULTS.json`;
  - each GPU line's `RESULT.json`, log, start and exit files;
  - `REPORT.md`.
- **Source records (L81):** the findings and their mechanisms come from REV2's `findings.json`, `REPORT.md` and
  `repro/`. The lead's GPU confirmations are `lead_p1rev2_gpu_repros.json`. LF1, LF2 and LF6 cases come from F1's
  `scripts/apa_p1f1_gpu.py` and its receipts in the p1d worktree.
- **Operand census:** every TQ4 key quantizer a packed entry owns (R2), every native index type that receives a
  capacity or extent (R3), every raw spo2 helper and EB3's timeline templates (R4).
- **Baseline isolation:** Build F and every REV1, F1 and REV2 receipt are references, never edited. New builds go
  under `artifacts/apa_p1f2/`.
- **Forbidden configuration:**
  - edits outside the writable set;
  - any change to a finite-input result;
  - restoring NaN masking for q, k or v;
  - an `EXPECTED.json` amended after an F2 build exists;
  - changing any existing test;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, with at most 3
  build jobs. Start one only with `MemAvailable` at or above 12 GB. A full build compiles `kernels.cu` (about 18
  minutes under load). A mutant rebuild in a copied tree recompiles only what changed.
- **Time:** every command under 30 minutes, except builds, which stay under 60. Long commands run detached and are
  waited on in the foreground with `timeout <n> tail --pid`. The round's ceiling is 10 hours of wall time, with GPU use
  at most 2.5 hours in total. If it cannot finish, stop and report `RED-BUDGET` with what was done.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free (146 GB free at sealing). Mutant tree copies count.
- **GPU:** only through `forge-gpu-run`, one line at a time, as in S6.

## Registration checklist (order-lint)

- **L01:** the instruments exist, each checked by running it first:
  - P1-D's build script (import);
  - `cuobjdump --version`;
  - `nvcc --version`;
  - pytest `--version`;
  - `forge-gpu-run --status --min 30 --max 600`;
  - F1's `scripts/apa_p1f1_gpu.py` (import) and `scripts/apa_p1f1_sass.py` (import).
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. R1, R3 and R4 refuse; nothing is silently copied, clamped or rerouted.
- **L06:** LF4's 64 steps per D is a fixed floor on a fixed case, not a share of an operand. It covers growth from the
  case's S to S+64, at the scale of P1-Q's append check (192 checks per model). The case's S and N are written in
  `EXPECTED.json` before the line runs.
- **L07:** n/a. No prediction is registered; every gate is a pass/fail gate.
- **L32:** the one bounded buffer is the packed key store (`capacity` rows). When it is full, `append` raises and the
  object is unchanged. LF4 fills it to capacity and then checks that refusal (G4). Its consumers are the packed
  attention entries, which read only the visible rows (`apa_packed.py` `_packed_tuple`; the seat lists every reader
  by grep in `INDEX_CENSUS.json`). No sizing figure is handed to a consumer in this round. "Overflow" in R5 is
  floating-point overflow of a score, not a buffer.
- **L08:** builds and receipts go under `artifacts/apa_p1f2/`, and scratch under `/tmp/apap1f2_*`.
- **L09:** identity is compared per output array with `np.array_equal` on same-width unsigned views, after dtype and
  shape checks. Test outcomes are compared per ID, and SASS per kernel. Logs and timings are never compared. A field
  that holds timings or clock values is excluded by name, listed in the report.
- **L10:** finite numerics must not change, so LF1 is bitwise (G1). The non-finite gate compares row sets, not values.
- **L12:** the inherited gates are F1's LF1 (bitwise), its LF2 NaN row-set equality, and the per-ID identity of LF6
  and the CPU sets. R8 adds a keyword with a None default, so no existing call changes. No inherited predicate is
  contradicted, except F1's G2b prediction for ±Inf in k, which this round turns into G2's containment gate.
- **L15:** n/a. No threshold is fitted.
- **L16:** the cheapest passing work would carry the infinite norm only in `packed.cuh` and leave any other TQ4
  quantizer as it is, add the extent check only to `packed_rotate`, and check only the sha (not the entries) at use.
  Excluded by:
  - LF2 run on every packed entry with containment per row;
  - R2's code census, checked row by row in `TQ4_CENSUS.json`;
  - LF3 run for each helper in `HELPER_CENSUS.json`;
  - S4's case of a hand-built table that carries the pinned sha with other entries.
- **L17:** n/a. No ranking.
- **L18:** the shipped default stays packed, and the public route's carry (partition) is unchanged. `capacity=None`
  keeps every current result.
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width. For LF2,
  each case shows that the dense side has at least one non-finite row and at least one finite row, or, for cases where
  dense is all finite (sinks, all-negative logits), says so and gates containment only.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`. Your scripts must not
  contain a home-path literal; build paths from `Path.home()` or the environment.
- **L22:** the shapes and cases come from F1's LF1/LF2/LF6 receipts, REV2's `repro/` scripts and P1-Q's append check,
  named above.
- **L25:** bitwise comparisons put both sides on the same device, in the same process kind, from one build per side.
  LF2 compares entries within one build and one process.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool), because the round builds, runs GPU lines and kills
  device mutants itself. The review that follows is Codex, so it is cross-model.
- **L38:** existing tests are not changed.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/apap1f2_*`.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json`, written before it runs.
- **L42:** see Machine rules.
- **L45:** each GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit, and the log's sha256. Exit 124 is a
  timeout, never a pass.
- **L47:** the mutants' expected failures come from this order's rules, never from this round's runs.
- **L48:** every GPU line runs as written, from the worktree root, through the real `forge-gpu-run`. Its receipt lists
  each post-device step with a terminal label. A harness exception is not a device result.
- **L55:** LF1's bitwise claim holds only if R2 changes the stored norm on rows with a non-finite element alone, and R3
  and R4 add host checks that refuse only inputs that faulted or were out of range before. Show both from the code,
  then measure LF1.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty. GPU commands run only inside `forge-gpu-run`.
- **L60:** for each refusal you add, record in `BOUNDARY.json` the input, the refusal, and Build F's behaviour on the
  same input (fault, launch error, zero output, silent acceptance, or the same refusal).
- **L64:** reads are this worktree and the read-only paths above.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `artifacts/apa_p1f2/`. Stop if `/mnt/ForgeRealm` has under 40 GB free. Mutant tree copies
  count against that.
- **L72:** the write target is the named files and regions, `artifacts/apa_p1f2/`, and scratch.
- **L74:** plain wording. These are engine guards and their tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per line before the real run.
- **L20:** n/a. No parameter is picked or calibrated in this round.
- **L31:** n/a. No arm tests an alternative to an owner requirement. The packed default, the owner's requirement, is
  unchanged.
- **L37:** no knob is added. R8's keyword runs from `quantize_sp2_keys` (`apa_sp2.py`) to `PackedSpKeys`
  (`apa_packed.py`) to `ApaSpq1Workspace` and the packed launchers. All are writable.
- **L35:** your registrations (`EXPECTED.json`, `MUTANTS.json`, the censuses) may fix cases, seeds, positions and
  hashes. They may not remove, narrow or reclassify anything a gate covers. If a clause cannot hold as written, stop
  and report RED with the reason. Only the lead amends a gate. A rebuild never re-registers anything (S3).
- **L82:** the writable product files are the ones listed at the top. `rg -n 'pack_tq|tq4' tensor_cuda/src/apa_spo`
  lists R2's candidate quantizer sites. `rg -n 'int capacity|int dst_s|int S\b' tensor_cuda/src/apa_spo` starts R3's
  census.
- **L99:** the rulings rest on these lead readings, each run before sealing:
  - `findings.json` lists F01, F02, F04 to F09, and the withdrawn F03, with the sites quoted in each ruling;
  - the lead's GPU runs confirm F02 (both Inf signs, default route), F05 (B=0 rotate and prepare reach a CUDA
    invalid-configuration error; the control survives) and F06 (FLT_MAX scale and NaN/+Inf sinks):
    `lead_p1rev2_gpu_repros.json`;
  - `ops.h:328` declares the 14-argument `tc::apa_eb2b_run`; its only definition is `eb2b.cuh:138`, with delta and
    carry;
  - `apa_sp2.py:61` is `quantize_sp2_keys(k, bulk_bits, *, storage=None)`, and line 85 calls `PackedSpKeys(k)`;
  - `apa_packed.py:224-226` bounds capacity below only; `apa_packed.py:406-409` checks the margin table by `isinstance`
    only, and `lookup` (`:390-396`) never checks the sha;
  - `eb3.cuh:935` limits timeline to variants 0, 1 and 6, and the G check comes inside the dispatch after launches
    (`eb3.cuh:1060-1083`);
  - `rg -l 'pack_tq|tq4' tensor_cuda/src/apa_spo` lists `packed.cuh`, `eb3.cuh`, `spq4.cuh`, `spq6.cuh` and
    `packed_launcher.cuh`;
  - `rg -l 'GATE_FORMS|verify_all'` in this worktree finds nothing, so no registry lists tests or gates (L102).

  Two premises are seat checks that run first:
  - which of the five files above own a TQ4 key quantizer, and which only read codes made by another (R2's census);
  - the native index type at each site that receives a capacity or extent (R3's census).

  If either census changes what a ruling covers, the census governs the fix, and the report says so. It does not stop
  the round.
- **L102:** no generated file, registry or ledger row depends on the writable files (L99's grep). New tests are new
  files.
- **L103:** each build receipt (F2, F2H if built, and every mutant) records the module sha256 and a sha256 map of every
  engine source file (`tensor_cuda/src/`, `tensor_cuda/include/`, `tensor_cuda/tensor_cuda/`) at build time. Each GPU
  line refuses, before the lease:
  - a module whose sha256 differs from its receipt;
  - for Build F2, a working tree whose source map differs from the receipt's.

  Build F is checked against `5c171477…`. No copied file is named; F1's scripts are imported.
- **L104:** no ratio or fraction bar. Every gate is exact: bitwise per output array, row-set equality or containment
  per (batch, head, query) row per case, and outcome per test ID or per kernel; slack 0 = 0 events. The containment
  boundary is covered by mutant `m02` (one hidden row fails G2).
- **L105:** n/a. No arm is compared with a non-reference baseline by a margin.
- **L106:** the gates are this order's own lines. F1's line code is imported unchanged where reused, and its inputs
  (Build F, F1's receipts) exist on disk (lead `ls` of `artifacts/apa_p1f1/arms/F` in the p1d worktree).
- **L107:** a "new or changed files" list compares sha256 against a base map taken at round start, never mtime. It is
  report-only. You never move, rename or delete a file that was in the tree at round start.
- **L109:** your helpers run as `python3 -B -P` (or `-I`) and insert the repo root themselves. Before the final
  message, list every `.py` you wrote and confirm none shares a standard-library module name.
- **Staged-script lesson (REV2):** a script you stage for any later GPU run must first run on CPU up to its first
  device call (imports, table loads, argument parsing), with a receipt. REV2's staged `r01` passed `py_compile` and
  then failed on a call signature.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md`, `/mnt/ForgeRealm/Project-Tensor/AGENTS.md`, `docs/APA.md`, REV2's
  `findings.json` and `REPORT.md`, and F1's `artifacts/apa_p1f1/REPORT.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout, and never start one while another is running.
- No deletes by you, in the leave-and-list form. Never kill a process you did not start.
- **The report:** try to write `artifacts/apa_p1f2/REPORT.md` early and keep it current. If the write is refused, say
  so; your final message is then the report, and the lead files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known:
  - IEEE 754 NaN and overflow propagation;
  - TurboQuant (Zandieh et al., 2025), the rotated 4-bit key codes this route stores;
  - the online-softmax merge (Milakov and Gimelshein, 2018; FlashAttention, Dao et al., 2022);
  - SHA-256 pinning (NIST FIPS 180-4);
  - capacity reservation as in `std::vector::reserve`;
  - mutation testing (DeMillo, Lipton and Sayward, 1978).

  Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G9, each PASS or FAIL;
- the diff summary;
- every module sha256 and the SASS result against the expected list;
- the TQ4, index and helper censuses (summaries);
- the CPU and per-ID results;
- the mutant table;
- each GPU line's result, with its retries;
- the deviations;
- every file written.
