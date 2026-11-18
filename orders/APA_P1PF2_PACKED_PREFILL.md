# APA-P1-PF2: implement prefill on packed SP keys (route a-S, direct streaming from the 4-bit codes)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2`
(branch `apa-p1pf2`, from P1-F3's `2945c3e9`, at the commit carrying this order). Edits, builds, CPU runs and GPU runs
through `forge-gpu-run` are AUTHORIZED in:
- `tensor_cuda/src/apa_spo/` (a new prefill kernel and its launcher; existing kernels' code stays as it is);
- `tensor_cuda/src/bindings.cpp`, only inside the `_tensor_cuda_spo2` module, for the new entry;
- `tensor_cuda/tensor_cuda/apa_packed.py`, only `packed_attention`'s L > 1 branch (`apa_packed.py:514-525`, which
  refuses today) and its docstring;
- `tensor_cuda/tensor_cuda/__init__.py`, only the packed regions (`APA_SPO2_D1_BEGIN` to `APA_EB4_END`) and the
  `_spo2` loader lines above them, if the new entry needs a wrapper;
- new `tensor_cuda/tests/test_apa_p1pf2_*.py`, new `scripts/apa_p1pf2_*.py`, and `artifacts/apa_p1pf2/` (new);
- scratch under `/tmp/apap1pf2_*`. Point `TMPDIR` there.

**Read-only:**
- every other file in this worktree, including every existing kernel, test, `kernels.cu`, `apa_sp2.py` and every
  earlier round's artifacts;
- `tensor_fork_apa_wt_p1pf`: the design (`artifacts/apa_p1pf/DESIGN.md`, `REPORT.md`, `MEASUREMENTS.json`,
  `apap1pf_emulate.py`);
- `tensor_fork_apa_wt_p1f2`: Build F3, your before-build (`artifacts/apa_p1f3/arms/F3/`, module sha256
  `2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9`);
- `tensor_fork_apa_wt_p1q`: P1-Q's model-quality harness (`scripts/apa_p1q_*.py`) and its QR driver
  (`scripts/apa_p1qr_*.py`);
- `tensor_fork_apa_wt_spq3`: the captures;
- every other worktree and the production repository.

(The worktrees are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/`.) **Searching:** never walk
`/`, `/mnt` or `/mnt/Shared` recursively; search inside the named roots.

## Why

The owner's requirement (11-02) is APA memory back at the two-pass level, with packed SP keys as the shipped default.
P1 shipped that default, but prefill (L > 1) on packed keys still refuses (P1-RES). A caller who prefills must hold a
dense fp32 copy, which is the memory the default exists to save.

The P1-PF design round (receipts `086ea24c`) found that no prefill route can be bitwise equal to dense prefill without
changing existing numerics. It recommends route (a-S): the dense prefill kernel's structure, with each key's bulk
score taken from its 4-bit codes in rotated space (`q·kq_j = n_j·(qrot·c_j)`), the same running-maximum refine rule,
exact refine from `k`, and one softmax over every key. It needs no key-shaped transient and changes no existing
numerics.

On four Qwen3 layers at 8,192 rows, its emulation flips 4.1e-8 to 1.5e-7 of refine decisions against dense SP
prefill. Its output difference is 0.04% to 0.21% of SP's own error. Read `DESIGN.md` §1, §2.1 and §5 first.

## Lead rulings (binding; ledger 2026-11-16 09:20)

- **R-E is replaced.** Prefill on packed keys is served by route (a-S) when this round passes every gate below.
  Otherwise it keeps refusing, as today.
- **Carry rule:** the whole-row running maximum, the dense prefill rule (`kernels.cu:8415-8416`). The public decode
  route keeps partition.
- **The packed margin table** (pinned) applies to prefill unchanged. PF-A1 checks the decisions empirically.
- **The non-finite contract** (P1, as narrowed after REV4) applies:
  - NaN in q, k or v gives equal row sets with dense;
  - a non-finite input is never hidden;
  - a key whose stored norm is non-finite makes every row that reads it non-finite;
  - finite inputs whose score arithmetic overflows fp32 are outside both routes' contract.
- **The design's gate set (`DESIGN.md` §5.1) is adopted, with these changes:**
  - **PF-Q1 is aligned to P1-Q:**
    - AB-1's 10 decode windows per model, S0 = 7,936;
    - arms: dense SP prefill (`storage="dense"`) and packed prefill (default), layer 0 exact in both;
    - two populations: (i) the prefill's own logits at the last 1,024 positions of each window, N = 10,240 per model;
      (ii) 256 teacher-forced decoded tokens per window, N = 2,560 per model;
    - clauses, per population and model, packed against dense:
      - mean KL(exact‖packed) ≤ 1.10 × mean KL(exact‖dense) + 2e-5;
      - top-1 against the exact reference, one-sided: top1(packed) ≥ top1(dense) − 0.003, which is a slack of 30
        positions at N = 10,240 and 7 tokens at N = 2,560;
      - pooled paired dNLL(packed − dense) ≤ +0.002 nats per token, with the paired-bootstrap 95% CI upper bound
        ≤ +0.005 (10,000 draws over windows, seed 0).
  - **PF-I2's fallback:** if the two kernels' fp32 contraction differs so that the all-refined outputs cannot be
    bitwise equal, do not stop. Report it, show the contraction difference from the SASS or the PTX, and hold PF-I3's
    tolerance form for that comparison too. Name the instruction-level cause.
  - **The speed line is informative,** not gated. Report packed prefill against dense prefill at S = 8,192 and 32,768,
    on both geometries, as multiples of the flash reference.
  - **Mutants:** the design's 16 are registered as written.

## Gates (as `DESIGN.md` §5.1, with the rulings above)

- **G1, identity:** PF-I1 (decisions and bulk against packed decode, `carry="row"`, bitwise); PF-I1n (non-finite
  keys); PF-I2 (output against the dense prefill kernel when every key refines, bitwise, or the fallback above); PF-I3
  (output against packed decode within τ, registered before the run).
- **G2, attention quality:** PF-A1 (decision flips ≤ 1e-6 per layer, on the four SPQ3 layers the design used) and
  PF-A2 (pooled output difference ≤ 0.02 of SP's error; pooled packed error ≤ 1.01 × dense SP's; rows with d > e_SP
  among rows with e_SP ≥ 1e-4·‖out_exact‖ at most 1e-4 of N; the floor rows held by d ≤ 1e-4·‖out_exact‖).
- **G3, model quality:** PF-Q1 as aligned above, on both models and both populations.
- **G4, refusals:** the design's 15 refusal rows, each raised before any device work. P1-D's
  `RE_prefill_on_packed_refused` row becomes an acceptance row, listed as the one intended change.
- **G5, memory:** P1-D's L4 extended. The allocator's high-water mark during a packed prefill call, minus resident
  memory, is at most the output, the optional mask and one allocation granule. No S × D key-shaped allocation
  (mutant 15's census).
- **G6, regression:** P1's finite-identity gates and P1-D's L1, L2 (one intended change), L4 and L6, against Build F3.
  Every pre-existing kernel's object code is unchanged (P1-D G7 form). Dense SP goldens and the packed margin pin are
  unchanged.
- **G7, mutants:** 16 of 16 killed, each by its named assertion.
- **G8, scope:** writes only in the writable set; GPU only through `forge-gpu-run`; no git writes; nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G8 pass. Then the lead verifies on disk, and a cross-model (Codex) review
follows before any promotion.

## Builds and SASS (L09 as extended 11-16)

- **Build F3** is the before-build (above), checked by sha before every use.
- **Build PF** is this round's tree, built with P1-F2's build script (`scripts/apa_p1f2_build.py`, imported) from this
  worktree's sources.
- Before the first PF build, write `artifacts/apa_p1pf2/EXPECTED.json`, once. It holds:
  - the kernels you will add, with their rulings; there are no changes to existing kernels;
  - every non-code source of difference between the builds. This worktree's source path differs from F3's (built in
    `tensor_fork_apa_wt_p1f2`), so nvcc's path-derived names (`_GLOBAL__N__<8 hex>_`, `__nv_static_<n>__<8 hex>_`,
    and the mangled length counts) differ. Register a mapping of those tokens: the lead's
    `artifacts/apa_p1f3/lead_f3_sass_check.py` in `tensor_fork_apa_wt_p1f2` is a working example. Also register the
    build-directory `_cu_` suffix mapping.

  Compare per kernel. Never amend `EXPECTED.json` after a PF build exists.

## Mechanism & scope

- **Measures:** whether prefill on packed keys can serve the packed default with no key-shaped transient, matching
  packed decode's decisions bitwise and dense SP prefill's quality.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf/artifacts/apa_p1pf/DESIGN.md`
  (route a-S, §2.1; gates, §5.1).
- **Exercises:** a new prefill kernel and entry; the L > 1 branch of `packed_attention`; Build PF plus mutant builds;
  CPU tests; GPU lines for G1 to G6; P1-Q's harness for G3.
- **Does not cover:**
  - grouped reuse for G in {4, 8} (a later optimisation);
  - speed as a gate;
  - DS1;
  - promotion.
- **Required evidence:**
  - the diff;
  - Build PF and mutant build receipts with module sha256s and source maps;
  - `EXPECTED.json` and its sha;
  - `SASS_COMPARISON.json`;
  - each GPU line's `RESULT.json`, log, start and exit files;
  - G3's per-model, per-population table;
  - `MUTANTS.json` and `MUTANT_RESULTS.json`;
  - the speed line;
  - `REPORT.md`.
- **Operand census:** the design's PF-I1 grid; the four SPQ3 layers; AB-1's decode windows per model; every caller of
  `packed_attention` (G4 caller census).
- **Baseline isolation:** Build F3, the design's receipts, P1-Q's harness and the captures are read-only references.
- **Forbidden configuration:**
  - any change to an existing kernel's code or to a finite-input result of an existing route;
  - a key-shaped transient;
  - reconstructing dense keys and calling the dense kernel;
  - re-tuning δ, ε or the margin tables;
  - changing an existing test;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`. Start one only
  with `MemAvailable` at or above 12 GB.
- **Time:** every command under 30 minutes, except builds (under 60). Long commands run detached and are waited on in
  the foreground with `timeout <n> tail --pid`. The round's ceiling is 12 hours of wall time, with GPU use at most 5
  hours. Time one PF-Q1 window first and project. If the project exceeds the GPU cap, stop and report `RED-BUDGET`
  with the projection.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free (85 GB at sealing).
- **GPU:** only through `forge-gpu-run`, one line at a time:
  `forge-gpu-run --min 30 --max <≤1800> --label p1pf2-<line> -- timeout <≤1750>s ...`. Do one dry run with
  `FORGE_GPU_DRY_RUN=1` (returns 78) before each line. Exit 76 means nothing ran: retry unchanged after at least 5
  minutes, at most 6 times.

## Seat

Claude Opus 5.5, high effort (Agent tool): kernel work with builds and GPU lines. Under SC's 14:30 limit, this is
Tensor's one Claude seat. The review that follows is Codex, so it is cross-model.

## Registration checklist (order-lint)

- **L01:** the instruments exist, each run first: P1-F2's build script (import), `nvcc --version`, `cuobjdump
  --version`, pytest `--version`, `forge-gpu-run --status --min 30 --max 600`, P1-Q's harness (import), and the
  design's `apap1pf_emulate.py` (import).
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. A failed gate keeps the refusal; mutant 15 excludes a hidden dense fallback.
- **L06:** the bars are the design's (§5.1) and P1-Q's, set before any implementation. G3's slack in events: 30 of
  10,240 and 7 of 2,560.
- **L07:** n/a. No prediction is gated. PF-I4 (the emulation's prediction) is informative.
- **L09:** identity is compared on same-width unsigned views after dtype and shape checks. Test outcomes are compared
  per ID, SASS per kernel with the registered mappings. Logs and timings are never compared.
- **L10:** G1 holds deterministic stages to identity (decisions against packed decode; outputs against the dense
  kernel when every key refines). G2 and G3 hold the adaptive route to quality bars.
- **L12:** the inherited gates are P1's finite identity, P1-D's L1, L2, L4, L6 and G7, and P1-Q's bars, unchanged. The
  one contradicted predicate is L2's `RE_prefill_on_packed_refused` row, whose replacement is G4's acceptance row.
- **L14:** n/a. No gain gate.
- **L15:** n/a. No threshold is fitted.
- **L16:** the cheapest passing work would reconstruct dense keys and call the dense kernel. Excluded by mutant 15's
  allocation census, PF-I1's decode oracle and G5. A correct but slow route passes every gate; the speed line shows it.
- **L17:** n/a. No ranking.
- **L18:** the shipped default stays packed. This round removes its last refusal.
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`.
- **L22:** the cases come from the design's grid and P1-Q's windows, named above.
- **L25:** each bitwise comparison puts both sides on the same device, in one process, from one build per side.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L31:** this round serves the owner's requirement (the packed default). It tests no alternative to it.
- **L32:** n/a. No bounded buffer is added.
- **L34:** a Claude seat (Opus 5.5 high). The review that follows is Codex.
- **L35:** your registrations (`EXPECTED.json`, `MUTANTS.json`, τ, seeds, rows) may fix cases. They may not narrow a
  gate. If a clause cannot hold as written, stop and report RED, except for PF-I2's registered fallback. Only the lead
  amends a gate.
- **L37:** no knob is added. The new route is reached through `packed_attention`'s existing L > 1 branch.
- **L38:** existing tests are not changed, except the one L2 row's intended change, which is listed.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/apap1pf2_*`.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json`, written before it runs. Show that each
  mutant's killing test executes the mutated line.
- **L42:** time one PF-Q1 window first, and project the whole run against the 5-hour GPU cap.
- **L45:** each GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256. Exit 124 is a
  timeout, never a pass.
- **L47:** the mutants' expected failures come from this order and the design, never from this round's runs.
- **L48:** every GPU line runs as written, from the worktree root, through the real `forge-gpu-run`.
- **L55:** PF-I1's identity holds only if the prefill kernel reproduces packed decode's per-pair bulk arithmetic
  exactly. Show one constructed pair (the design's L55 instance, `DESIGN.md` §2.1), then measure.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty. GPU commands run only inside `forge-gpu-run`.
- **L60:** for each refusal, record in `BOUNDARY.json` the input, the refusal, and Build F3's behaviour.
- **L64:** reads are this worktree and the read-only paths above.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `artifacts/apa_p1pf2/`. Stop if `/mnt/ForgeRealm` has under 40 GB free.
- **L69:** one reference per operand class:
  - decisions -> packed decode (G1);
  - all-refined outputs -> the dense kernel (G1);
  - quality -> dense SP prefill (G2, G3);
  - existing results -> Build F3 (G6).
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78. Use it once per line before the real run.
- **L82:** the writable product sites are named at the top. `grep -n 'L != 1' tensor_cuda/tensor_cuda/apa_packed.py`
  gives line 514, the refusal this round replaces.
- **L90:** the G4 caller census of `packed_attention` classifies each caller by role (product, test, harness, script)
  with counts. Treatment: product callers must reach the new L > 1 route, or record why not. Test, harness and script
  callers are listed only; none is changed (L38).
- **L99:** the lead's readings, run before sealing:
  - `apa_packed.py:514-525` refuses L > 1 on packed keys;
  - `launcher.cuh:184` refuses L != 1 in `apa_sp_d1`;
  - the design's `REGISTRATION.json` (`d7fc80c1`) and its measured flip counts (63 / 22 / 91 / 165 at ε = 0.01) were
    checked by the lead;
  - Build F3's module sha256 was recomputed by the lead;
  - `/mnt/ForgeRealm` had 85 GB free at sealing.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each build receipt records the module sha256 and a full engine-source sha256 map. Each GPU line refuses,
  before the lease, a module whose sha256 differs from its receipt.
- **L104:** every bar states its unit, N and slack:
  - PF-I1 and PF-I2: per decision and per element, slack 0;
  - PF-A1: per layer, N = 536,936,448 or 1,073,872,896, slack ≤ 1e-6;
  - PF-A2: per row;
  - G3: per population and model, N = 10,240 or 2,560, top-1 slack 30 or 7 events.

  Boundary mutants: QR3's form, on G3's gate code (an arm one event past the slack fails).
- **L105:** G3's top-1 clause is one-sided against the dense SP baseline: top1(packed) ≥ top1(dense) − 0.003.
- **L109:** your helpers run as `python3 -B -P`. List every `.py` you wrote, and confirm none shares a
  standard-library module name.
- **L111:** stage, reads, stored as:
  - bulk: rotated query, codes, codebook and norm, stored as the packed store's codes and scales;
  - refine: delta, from the pinned table;
  - exact score: `q` and `k`, from the caller's exact keys;
  - softmax: `v` and sinks, from the caller.

  Mutant 11 (exact score read from a reconstruction) and mutant 15 (the allocation census) are the accessor checks.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md`, `/mnt/ForgeRealm/Project-Tensor/AGENTS.md`, `docs/APA.md`, and the design's
  `DESIGN.md` and `REPORT.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **The report:** write `artifacts/apa_p1pf2/REPORT.md` early. If the write is refused, say so; your final message is
  then the report, and the lead files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known: TurboQuant (Zandieh et al., 2025);
  asymmetric distance computation in product quantization (Jégou, Douze and Schmid, 2011); BLASST (arXiv 2512.12087);
  online softmax (Milakov and Gimelshein, 2018); FlashAttention (Dao et al., 2022); mutation testing (DeMillo, Lipton
  and Sayward, 1978).

## Done

Your final message contains, verbatim:
- G1 to G8, each PASS or FAIL;
- the diff summary;
- every module sha256 and the SASS result with the registered mappings;
- G1's per-case counts;
- G2's per-layer table;
- G3's per-model, per-population table;
- G4's refusal rows;
- G5's memory figures;
- the mutant table;
- the speed line;
- each GPU line's result;
- the deviations;
- every file written.
