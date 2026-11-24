# BWD-F1: zero-extent broadcasting reads through a null pointer; reduce_to skips zero-extent axes; the BWD audit's hardening (implementing round)

Follows: BWD_AUD1_BLIND_AUDIT.md (branch `bwdaudit`, worktree `tensor_fork_apa_wt_bwdaudit`)
ID reuse: none

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1`
(branch `bwdf1`, from `eng4` at `fbd335ae7` = production main `f7282b1d` plus ENG4, at the commit carrying this order).
Edits, builds and CPU runs are AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`: only `broadcast_shape`, `bcast_strides`, `reduce_to`, and any other broadcast-shape
  computation Z1's census finds, plus host helpers you add for them;
- `tensor_cuda/src/ops.cpp`, `tensor_cuda/src/autograd.cpp`, `tensor_cuda/src/bindings.cpp`: only the broadcast-shape
  sites Z1 finds, the messages of H3, and the comments and docstrings of H2;
- `tensor_cuda/tensor_cuda/*.py`: only broadcast-shape sites Z1 finds;
- `tc_signal/tc_signal/__init__.py` (H1, H4), `tc_signal/src/bindings.cpp` (H3 messages only), `tc_signal/README.md`
  (H2, H4, H5);
- `tensor_cuda/include/tc/*.h`: only declarations of helpers you add;
- existing tests: only the changes Z4 allows, each listed;
- new `tensor_cuda/tests/test_bwdf1_*.py`, new `tc_signal/tests/test_bwdf1_*.py`, new `scripts/bwdf1_*.py`, everything
  under `artifacts/bwdf1/`;
- the build directories `tensor_cuda/build-bwdf1/` and `tc_signal/build-bwdf1/` and the modules they install into this
  worktree's packages;
- `logs/bwdf1_*`, and scratch under `/tmp/bwdf1_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here; every other worktree (the audit's receipts and witness scripts are in
`tensor_fork_apa_wt_bwdaudit/artifacts/bwd_aud1/`; ENG4's build tooling is in `tensor_fork_apa_wt_eng4/scripts/`); the
production repository; the consumer roots `/mnt/ForgeRealm/Project-Voice/engine` and `/mnt/ForgeRealm/GRAPA-Native-LLM`
(grep only).
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** you run no GPU work of any kind. You stage GPU lines and dry-run them; the lead runs them (Machine rules).

## Why

A blind audit of production's backward pass (BWD-AUD1, receipts `7977c1586`, `bbc1e7833`, `09630623c` on `bwdaudit`)
found 11 defects. The lead ran its GPU witness lines on production's build (module `8a46ceee`) on 2026-11-23:
- the F01 witness (float16 `a[4,3] @ b[3,0]`, `sum`, `backward`) and the F03 witness (float32
  `a[2,0,4,3] @ b[1,3,5]`, `backward`) both ended in an illegal memory access on the device;
- under `compute-sanitizer` memcheck, inside the lease, the fault is `binary_kernel` reading from address `0x0`
  upward, a null operand pointer (14 errors in the F01 run).

**The lead's reading of the cause** (at `fbd335ae7`, `kernels.cu:1028-1039`): `broadcast_shape` sets each output extent
to `max(a, b)`. With one extent 0 and the other 1 it returns 1, where NumPy's rule gives 0. `ew_binary`
(`kernels.cu:1147-1165`) then allocates a non-empty output and launches `binary_kernel` over it with the empty operand,
whose storage is a null pointer. Any public binary op on such shapes should reach it, for example a `[4, 0]` tensor
plus a `[4, 1]` tensor. That is a reading, not yet a measurement; Z2 measures it.

`reduce_to` (`kernels.cu:1364-1376`) is the same rule in reverse: it sums only axes whose extent is above 1, so a
gradient with a zero-extent axis against a target extent of 1 is never summed, and the reshape raises (audit F03;
item REDUCE-TO-ZERO-EXTENT).

The audit's other findings this round takes, as hardening: F06, F07, F08, F09, F10 and F11. Not taken here:
- F01 and F02 are closed by ENG4's forward on this branch (zero fill for K = 0; one operand predicate); Z3 shows it;
- F04 (`NDArray::zeros` ignores its `cudaMemset` status) is inside ENG5's status census;
- F05 (layer_norm, group_norm, lstm and rint refuse under grad mode) is the documented scope of the BWD contract.

## Lead rulings (binding)

### Part Z: zero extents

- **Z1, census first.** Write `artifacts/bwdf1/CENSUS.json` with:
  - every place in the engine that computes a broadcast output shape or broadcast strides: C++ (`kernels.cu`,
    `ops.cpp`, the other `src/` files) and Python (`tensor_cuda/tensor_cuda/*.py`), with file:line and the rule each
    uses for an extent of 0 against 1;
  - every public op that reaches each site, with the call path;
  - every caller of `reduce_to`, with file:line;
  - the consumer roots (read-only grep): any site that can pass a zero-extent tensor into a broadcasting op, else
    "not found".
- **Z2, the broadcast rule.** Every site in Z1 follows NumPy's rule: two extents are compatible when they are equal or
  one of them is 1, and the output extent is the other one. So 0 against 1 gives 0, and 0 against an extent above 1
  is refused with `RuntimeError` naming both shapes, before any CUDA call. In addition, the binary dispatch refuses,
  before any launch, an operand with zero elements when the output has elements (it cannot be read), so a future
  shape bug fails on the host instead of on the device.
- **Z3, `reduce_to`.** It sums every axis where the target extent is 1 and the source extent is not 1, including 0. A
  sum over an empty axis gives zeros. Results where no axis has extent 0 are bitwise unchanged (G4).
- **Z4, tests that pin the old behaviour.** An existing test that asserts the old output shape or an old refusal for
  a zero-extent broadcast changes to assert NumPy's rule. List each in `artifacts/bwdf1/INTENDED_CHANGES.json` with
  the old and new assertion, before any after-run. No other existing test changes.

### Part H: the audit's hardening (one fix each, lead-verified, no re-review)

- **H1 (F06).** In `tc_signal/tc_signal/__init__.py`, an op with a tracked path that is called with frozen inputs and
  an `out=` destination requiring gradients refuses it as the tracked path and the four refused ops do
  (`README.md:97-99`).
- **H2 (F07, F09, F10).** Documentation only:
  - the process-wide host dispatch lane (`autograd.cpp:28-46`): the `attach_vjp` docstring and the README
    "Gradients" section say a second thread's backward raises while one runs;
  - the tracked VJPs need the operand's device to be current (F09); say so in the README;
  - core's per-device 32 MiB backward workspace, next to tc_signal's (`README.md:87-90`).
- **H3 (F08).** Each message the audit lists names the operand and the shapes: `tensor_cuda/src/bindings.cpp`'s
  `attach_vjp` argument error, `matmul_backward`'s messages in `ops.cpp`, and `tc_signal/src/bindings.cpp`'s private
  entry messages. Messages only; no behaviour change.
- **H4 (F11).** `tc_signal/README.md:416-419` says `instance_norm1d` raises for a trainable scale or bias; the code
  differentiates both (`_norms.py:90-108`) as `:92-95` says. Rewrite `:416-419` to match the code. Voice trains
  through `instance_norm1d`.
- **H5.** The README has two "Gradients" sections (`:69` and `:409`). Make the second point to the first, or merge
  them, without changing any claim except H4's.

## Gates

- **G1, CPU only** (`CUDA_VISIBLE_DEVICES` empty). Through the public API, every Z2 refusal (0 against an extent
  above 1, in each position and rank) raises naming both shapes, before any CUDA call. Use ENG3's CUDA call counter
  (`tensor_cuda/tests/support/cudart_count.c`) to show no CUDA call. Also every H3 message, where it is reached before
  device work.
- **G2, GPU line, the null read.** The audit's F01 and F03 witness scripts, copied with only their module pin changed,
  plus a direct witness (`[4, 0]` plus `[4, 1]`, and the same for every binary op in Z1's census, each dtype), run under
  `compute-sanitizer --tool memcheck` inside the lease:
  - on Build B: record what happens (the reading predicts a null read);
  - on Build E: 0 memcheck errors; each output has NumPy's shape; each gradient is zeros of its operand's shape.
- **G3, GPU line, zero-extent gradients.** For every op in Z1's census whose backward goes through `reduce_to`, a
  zero-extent broadcast gives a gradient of zeros of the operand's shape on Build E, compared with NumPy. Record Build
  B's behaviour.
- **G4, no regression.** Production's test suite, per test ID, Build B against Build E, on CPU, and staged as GPU lines.
  Outcomes are identical except the intended changes listed before the after-run, each with its ruling. Results for
  shapes with no zero extent are bitwise unchanged; G4's suite covers them, and G2's direct witness adds one
  nonzero-shape control per op compared bitwise.
- **G5, mutants** (validity of the new tests). Each one fails its named assertion in `artifacts/bwdf1/MUTANTS.json`,
  written before any mutant runs:
  - `m01_max_rule`: `broadcast_shape` returns `max` again (G1 or G2);
  - `m02_no_empty_operand_guard`: the dispatch guard is removed (a test that drives the dispatch with an empty operand
    and a non-empty output; a staged GPU line if it cannot be reached on CPU);
  - `m03_reduce_gt1`: `reduce_to` tests `> 1` again (G3);
  - `m04_h1_dropped`: H1's destination check is removed (its test; staged if it needs the GPU).
- **G6, scope.** Writes only in the writable set. No GPU use. No git writes, nothing deleted.

**Registered acceptance:** the seat's part passes when G1, G4's CPU half, the CPU-killable mutants and G6 pass, and
G2, G3, G4's GPU half and the GPU mutants are staged and dry-run clean. The round is ACCEPTED when the lead's GPU lines
pass too. Part Z then goes to a cross-model (Codex) review; Part H is lead-verified. Promotion needs the owner's
verbatim approval and the review receipt.

## Builds

- **Build B:** this branch as committed (`fbd335ae7`), default configuration, both `tensor_cuda` and `tc_signal`.
- **Build E:** this round's tree.

Build out of tree under the build directories above, the way ENG4 built its arms (`artifacts/eng4/REPORT.md` and
`scripts/eng4_build.py` in `tensor_fork_apa_wt_eng4`). Each build receipt records the module sha256 and a sha256 map of
every engine source (`tensor_cuda/src/`, `tensor_cuda/include/`, `tensor_cuda/tensor_cuda/`, and for `tc_signal` its
`src/`, `include/` and `tc_signal/`).

## Mechanism & scope

- **Measures:** whether broadcasting follows NumPy's rule for zero extents everywhere in the engine, with no device read
  of an empty operand; whether `reduce_to` sums zero-extent axes; and whether the audit's hardening holds.
- **Mechanism doc:** `tc_signal/README.md` (sections "Gradients" at `:69` and `:409`) and the audit's report,
  `tensor_fork_apa_wt_bwdaudit/artifacts/bwd_aud1/REPORT.md`.
- **Exercises:** every broadcast-shape site; `reduce_to`; the binary dispatch guard; H1's destination check; Builds B and
  E plus mutant builds; CPU tests with no visible GPU and the CUDA call counter; staged GPU lines for G2, G3, G4 and the
  GPU mutants.
- **Does not cover:**
  - ENG5's host-operand refusals and status checks (its own lane);
  - F04 and F05 (above);
  - new broadcasting beyond NumPy's rule;
  - performance;
  - promotion.
- **Required evidence:** `CENSUS.json`; the diff; build receipts with module sha256s; G1's CPU result; `MUTANTS.json`
  and `MUTANT_RESULTS.json`; `INTENDED_CHANGES.json`; G4's CPU per-ID before and after; the staged GPU lines with their
  dry-run exits; `REPORT.md`.
- **Operand census:** Z1's sites, ops, `reduce_to` callers and consumer sites; G2's ops and dtypes; G1's refusal
  positions and ranks.
- **Baseline isolation:** Build B is the reference for G4 and for every "record Build B" clause. Production is
  read-only.
- **Forbidden configuration:**
  - any change to a result for shapes with no zero extent;
  - a refusal that runs after a CUDA call;
  - edits outside the writable set;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`. Start one only
  with `MemAvailable` at or above 12 GB. Other seats build on this machine: builds at most `-j4`.
- **Time:** every command under 30 minutes, except builds (under 60). The round's ceiling is 6 hours of wall time.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** none of any kind for you:
  - never call a GPU step function or a staged script directly, never run `compute-sanitizer` yourself (it removes
    `CUDA_VISIBLE_DEVICES` from its target, so the target sees the GPU), never remove or override
    `CUDA_VISIBLE_DEVICES` in a child of yours;
  - a smoke of a staged line is its `FORGE_GPU_DRY_RUN=1` dry run (returns 78), nothing else.
  Stage each GPU line as one command in `artifacts/bwdf1/GPU_LINES.json`:
  `forge-gpu-run --min <≤480> --max 540 --label bwdf1-<line> -- timeout <≤500>s ...` (a memcheck line puts
  `compute-sanitizer --tool memcheck` after the `timeout`). Every step script starts with
  `if subprocess.run(["forge-gpu-lease-check"]).returncode != 0: raise SystemExit("not inside a forge-gpu-run lease")`
  before it imports the engine or changes `CUDA_VISIBLE_DEVICES` (order-lint L125). Each line writes under
  `artifacts/bwdf1/gpu/<line>/` and refuses to start if that directory exists.

## Registration checklist (order-lint)

- **L01:** the instruments are cmake, nvcc, gcc, pytest, python3 with NumPy, `compute-sanitizer` (`--version` only)
  and `forge-gpu-run` (`--status --min 30 --max 540`). Run each first and record it.
- **L02:** GPU lines carry an explicit `--min`; each lane's smoke is its dry run.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. A shape that NumPy refuses is refused, not reshaped.
- **L07:** n/a. Nothing visual.
- **L08:** builds under the build directories above, receipts under `artifacts/bwdf1/`, scratch under `/tmp/bwdf1_*`.
- **L09:** G4 compares outcomes per test ID; G2's controls and G3's gradients compare per element on same-width
  unsigned views (bitwise) or against NumPy's zeros, after dtype and shape checks. Logs and timings are never compared.
- **L10:** results for shapes with no zero extent are held to identity with Build B.
- **L12:** inherited gates: production's suite per ID; intended changes listed first (Z4).
- **L14:** n/a. No gain gate.
- **L15:** n/a. No threshold is fitted.
- **L16:** the cheapest passing work would fix `ew_binary` alone and leave other broadcast sites on the `max` rule.
  Excluded by Z1's whole-engine census, G1 over every refusal position, G2 over every binary op in the census, and
  `m01`.
- **L18:** n/a. No owner requirement behind an option.
- **L19:** every bitwise comparison first checks each side is nonzero and non-constant, at full width; zero-gradient
  checks compare against NumPy zeros of the expected shape.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L22:** the shapes and dtypes are defined by this order and `MUTANTS.json`; inputs use recorded seeds.
- **L25:** each build runs in its own process, with the same inputs from the same seeds, compared afterwards.
- **L32:** n/a. No bounded buffer.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool). Part Z's review is Codex, so it is cross-model.
- **L35:** your registrations may fix cases and seeds; they may not remove, narrow or reclassify anything a gate
  covers. If a clause cannot hold as written, stop and report RED.
- **L37:** n/a. No knob is added.
- **L38:** existing tests change only as Z4 allows, each listed.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/bwdf1_*`. Restore the tracked
  `artifacts/wo*.ppm` files that `test_terrain_render.py` writes (read-only git: `git show HEAD:<path>`), or exclude
  that file and list it.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json` before it runs; show each killing test
  executes the mutated line.
- **L45:** each staged GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256. Exit 124
  is a timeout, never a pass.
- **L47:** the mutants' expected failures come from this order.
- **L48:** every GPU line runs as written, from the worktree root, through the real `forge-gpu-run`. Before staging,
  drive each line's post-device code (readback, memcheck-summary parse, comparison, verdict) on constructed results of
  each dtype and shape class it will see, including the empty and one-element cases, and list those steps in
  `GPU_LINES.json`. No trailing command masks a line's exit.
- **L53:** n/a. Nothing is built for another project's run.
- **L55:** for one witness per Z2 class (0 against 1, 0 against an extent above 1, equal extents, 1 against an extent
  above 1), show the shapes, `broadcast_shape`'s result on Build B and Build E, and the strides the kernel then uses.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your own
  scripts is repaired at most 3 times, each recorded.
- **L58:** G1 runs with `CUDA_VISIBLE_DEVICES` empty. No child of yours removes or overrides it.
- **L60:** for each refusal Z2 adds, record in `artifacts/bwdf1/BOUNDARY.json` the call, the refusal, each term of its
  predicate with the rows that trigger it, and Build B's behaviour for the same operands (on CPU where Build B reaches no
  CUDA call; otherwise "source reasoning, file:line" or "GPU line <name>").
- **L64:** reads are this worktree, the audit's and ENG4's worktrees named above, ENG3's counter source, and the two
  consumer roots (grep only). Record the paths the inherited suite reads outside the worktree in one CPU run under an
  audit hook on `open`, with counts.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds under the two build directories; the audit's two builds were 45 MB and 8.7 MB, and ENG4's whole build
  tree with every arm was 989 MB. Free space at sealing: 52 GB on `/mnt/ForgeRealm` (`df -h`, 2026-11-23 19:32; disk-watch WARN at 49.9 GB, 62 GB of retired worktrees pending flush in DELETE_ME). Check
  before each build and stop below 40 GB.
- **L69:** one reference per operand class: shapes -> NumPy's broadcasting rule; refusals -> Z2; zero-extent gradients
  -> NumPy zeros; other results -> Build B (G4); the null read -> memcheck on Build B (G2).
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail and each is run (GPU ones as staged lines).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per staged line.
- **L81:** the null-read expectation comes from two sources: the lead's memcheck runs on production's build (source
  records: kernel name `binary_kernel<__half>` and `<float>`, addresses from `0x0`, thread and block, error count; not
  the host call stack, which is unsymbolized) and the lead's reading of `broadcast_shape`. G2 re-measures it on Build B.
- **L82:** `grep -n "broadcast_shape\|bcast_strides\|reduce_to\|max(" tensor_cuda/src/*.cu tensor_cuda/src/*.cpp` and
  `grep -n "broadcast" tensor_cuda/tensor_cuda/*.py` start Z1's census. The lead's run at `fbd335ae7` found
  `broadcast_shape` at `kernels.cu:1028`, `bcast_strides` at `:1041`, `ew_binary` at `:1147`, and `reduce_to` at
  `:1364` with callers `ops.cpp:873`, `ops.cpp:1240` and `autograd.cpp:90`.
- **L89:** evasion: a broadcast computed by a helper with another name, or inline. Excluded by Z1's census of every
  output-shape computation (by behaviour, not name) and by G2 over every binary op; control `m01`.
- **L90:** roles: product (engine C++ and Python), test (existing tests), consumer (Voice and GRAPA, read-only),
  harness (your scripts). Count per role in `CENSUS.json`. Product sites are fixed per Z2 and Z3; consumer sites are
  report-only; tests change only per Z4.
- **L99:** the rulings rest on the lead's readings at `fbd335ae7`, quoted above under Why and L82, and on the lead's
  GPU runs on production's build (the audit's commits `09630623c`, logs held back with sha256 in
  `artifacts/bwd_aud1/LEAD_HELD_BACK.tsv` on `bwdaudit`). Seat check that runs first: re-read the four functions and
  record any difference.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each staged GPU line refuses, before the lease, a module whose sha256 differs from its build receipt; the
  receipt carries the full-tree source map.
- **L104:** no ratio bar. Every gate is exact (per op, per shape, per test ID).
- **L109:** your helpers run as `python3 -B -P` (or `-I`); none shares a standard-library module name.
- **L111:** n/a.
- **L114:** builds run in their own scope (`MemoryMax=12G`). Staged GPU lines set no RLIMIT_AS; the lead runs them in a
  scope with `MemoryMax=12G` and `OOMPolicy=continue`. `GPU_LINES.json` lists each line's expected peak host bytes.
- **L117:** H4 changes only documentation of `instance_norm1d`'s gradients: Voice trains its scale and bias through
  adain (the audit's consumer reach for F11); the code already differentiates them. No gradient is added or removed.
- **L118:** the consumers of the changed rules are Z1's ops, the `reduce_to` callers and the tests that reach them;
  all run in G4.
- **L119:** every write-once path a staged line creates is listed in `GPU_LINES.json`, so a retry can move it first.
- **L125:** every staged step script starts with the `forge-gpu-lease-check` refusal above, before any engine import or
  `CUDA_VISIBLE_DEVICES` change; cite its file:line in `GPU_LINES.json`. Seats exercise staged lines only by dry run.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **The report:** if writing `artifacts/bwdf1/REPORT.md` is refused, your final message is the report, and the lead
  files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known: NumPy's broadcasting rules (the NumPy
  documentation, "Broadcasting"; extents of 0 broadcast only against 0 or 1); PyTorch's `sum_to_size` for reducing a
  gradient to a broadcast operand's shape; mutation testing (DeMillo, Lipton and Sayward, 1978). Write "no prior art
  known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS, FAIL or STAGED;
- the Z1 census summary, and the Z2 classes each with its witness;
- the diff summary;
- every module sha256;
- G4's CPU per-ID result and the intended changes;
- the mutant table;
- each staged GPU line with its dry-run exit;
- the deviations;
- every file written.
