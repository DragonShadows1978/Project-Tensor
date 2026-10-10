# APA-REGT2: port the packed 0/1 identity probe (pIdV) to production as regression tests

David, 2026-10-09, verbatim:
- the standing rule: "On things like regression tests going into production, that's fine unless it really
  significantly hurts performance";
- answering "Should I port the 0/1 probe into the regression tests?": "Yes, yes, and yes."

## Background

APA-LD3 (`apa-ld3` 24f8b11) built pIdV as its strongest non-power-of-two baseline, and pIdV beat everything:
- **Wrong-source and interior leaks:** it killed 28/28 at production shapes, 1 run each, median 0.037 s. These are
  wrong KV head, wrong batch, batch stride, and wrong-head interior tiles, at B 2, Hq 16, KVH 8, D 128, up to
  S = 98,304.
- **LD2's edge leaks:** it killed 30/30, including both CU6 tile-edge cells (A30, A31). The REGT1 tests in production
  do not catch CU6.
- **False alarms:** 0 on 35 main hosts.
- **Oracle:** V ∈ {0, 1}, and each output entry must be exactly zero or exactly nonzero. It needs no reference run,
  no repeat run, and no determinism.

Code to port: `scripts/apa_ld3_pdwl2.py` at 24f8b11, namely `pidv_layouts` (:337), `pidv_operand` (:358),
`pidv_check` (:381), and the helpers they use (`spread`, `_empty_layout`, the kind constants). Also
`scripts/apa_ld3_run.py:133` `pidv_cell`.

Production main (27d87fd) already has the LD1/LD2 engine adapter and the cut generator. These were ported by
APA-REGT1, at `scripts/apa_ld1_pdwl.py` (`Attn`, `up`, `down`, `expand_heads`) and `scripts/apa_ld2_pdwl.py`
(`frontier_cuts`). Reuse them; do not duplicate them.

The REGT1 review left one nit, F01. The CUDA-availability guard in REGT1's tests catches every exception and skips.
Your new test module must not repeat this. Skip only on a recognised no-device condition: extension import failure,
or the "no CUDA-capable device" error. Anything else must fail. REGT1's existing files stay untouched.

## Mechanism & scope

- **Measures:**
  - (a) whether the ported pIdV tests pass on the production engine;
  - (b) whether each engine-level test fails on its known-bad builds, including CU6;
  - (c) their added default-suite wall time.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ld3/docs/APA_LD3_REPORT.md`
  (the pIdV design, the W/I/E tables, and the leak census). Also `scripts/apa_ld3_pdwl2.py` at 24f8b11.
- **Exercises:**
  - new `scripts/apa_pidv.py`;
  - new `tensor_cuda/tests/test_apa_pidv.py`;
  - the production `scripts/apa_ld1_pdwl.py` / `apa_ld2_pdwl.py` adapters (import only);
  - sdpa, the fused causal path, apa_quant, apa_selective (fused and split-K), apa_int4 and apa_sp, on the pinned
    main engine. That engine is the same binary as production's installed `_tensor_cuda`, d8fda9a8.
- **Does not cover:**
  - the GRM gemma4 sliding path (production tests must not import GRM);
  - in-slab reordering of visible keys (pIdV is blind to it by design; say so in the module docstring);
  - fp16 tiny-weight underflow beyond the registered cells;
  - edits to existing tests or engine source;
  - backward passes.
- **Required evidence:**
  - the diff stat against 27d87fd;
  - the per-test durations on main;
  - the known-bad table (test, build path, sha256, verdict);
  - the full suite on the pinned main engine, with and without the new files;
  - every difference from the LD3 source, with its reason;
  - a demonstration that the new availability guard skips on no-device and fails on an injected non-device
    exception.
- **Operand census:**
  - **Edge cells:** LD2's small-geometry cells (QW (64,1025), MC3/G4G (3,33), G4G (3,1025)) for every kernel above.
  - **Production-layout cells:** B 2, Hq 16, KVH 8, D 128 at S = 8192, L ∈ {1, 64}, for sdpa and apa_selective.
  - **Optional:** one cell at S = 98,304, only if it fits the budget. Otherwise put it behind `TC_SLOW_TESTS=1` and
    document that in the module.
  - **Dtypes:** bf16, plus fp32 on at least one cell per kernel, and fp16 where LD3 ran it.
- **Baseline isolation:**
  - **Main:** `…/WorkTree/tensor_fork_apa_wt_mf2/artifacts/apa_mf_main/build` (sha256
    `d8fda9a8879b2c0de1b659653e8899aa6fbb89da36af03d1339e2a1c9b011e31`).
  - **Known-bad builds**, used in place and read-only, one engine per process:
    - LD1's CU1_gqa_tile, CU3_wrong_batch_v and CU4_batch_stride_off1_v (`…/tensor_fork_apa_wt_ld1/artifacts/apa_ld1/mutants/`);
    - LD2's CU5_offby1 and CU6_tile_edge (`…/tensor_fork_apa_wt_ld2/artifacts/apa_ld2/mutants/`);
    - LD3's I1_tile_head, I2_splitk_batch and I3_slab_tile_head (`…/tensor_fork_apa_wt_ld3/artifacts/apa_ld3/mutants/`).
  - **Python-level mutants** (bstride, offby1) belong only inside the tests, as positive controls.
- **Forbidden configuration:**
  - editing existing tests, REGT1's files or engine source;
  - widening the exact zero/nonzero oracle to a tolerance;
  - hardcoded `Fork/` or `WorkTree/` paths in shipped files;
  - a GPU test that errors, rather than skips, without CUDA;
  - a catch-all exception skip;
  - a test that passes on a known-bad build it is registered to catch;
  - the retired 595.84 driver override.

**Budget.** The new tests should add no more than 30 s to the default suite. Anything slower goes behind
`TC_SLOW_TESTS=1`, or is dropped; state which.

Seat: Claude Opus 5.5, high (an Agent-tool subagent; it has the GPU).

## Where to work

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_regt2`
(branch `apa-regt2` at 27d87fd, which is production main). Edits, CPU runs and GPU runs are AUTHORIZED there, in
these paths only:
- new `scripts/apa_pidv.py`;
- new `tensor_cuda/tests/test_apa_pidv.py`;
- `docs/APA_REGT2_REPORT.md` (new);
- `artifacts/apa_regt2/` (new; logs and evidence only);
- `logs/`.

Everything else is READ-ONLY: the rest of this worktree, every other worktree (read and import only; mutant builds
are used in place), the production repo `/mnt/ForgeRealm/Project-Tensor` outside `Fork/`, GRM, GRAPA,
`/mnt/ForgeRealm/reference`, the models and the HF cache.

NO git (the lead commits). NO subagents. NO DELETES: use `delete-me <path> --reason "..." --session apa-regt2`. Never
kill a process you did not start. Run everything synchronously.

**GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, on the native driver only
(595.99.02; never set the 595.84 override). If a lock wait times out, retry. Load the engine with
`APA_LD1_ENGINE=<build> source <ld2 worktree>/scripts/apa_ld1_env.sh` and
`PYTHONPATH=$APA_LD1_ENGINE:<this worktree>/tensor_cuda`. If the AFTERIMAGE memory hook denies a Write, read what it
shows you and retry the Write. Do not route around it through the shell.

## Required

1. **Read** the LD3 report's pIdV sections, the LD3 source, and REGT1's ported adapters in this worktree.
2. **Port** pIdV and write the tests, with the narrowed availability guard.
3. **Main:** run the new tests on the pinned main engine and record per-test durations.
4. **Known-bad:** run them on every known-bad build listed above, one command and its output per build.
5. **Full suite** on the pinned main engine: `pytest tensor_cuda/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
   The FAILED/ERROR set must equal the 52 baseline ids (`…/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log`;
   compare with `…/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py`), plus the new tests passing.
   Report wall time with and without the new files.

## Gates

- **G1 (port):**
  - `git diff 27d87fd --stat` shows only the writable paths;
  - no `Fork/` or `WorkTree/` string appears in the two shipped files;
  - the GPU tests skip cleanly with `CUDA_VISIBLE_DEVICES=""`;
  - an injected non-device exception in the availability probe FAILS rather than skips.
- **G2 (main):** every new test passes on the pinned main engine, and the full suite's FAILED/ERROR set equals the
  52 baseline ids.
- **G3 (teeth):** every listed known-bad build fails at least one new test, and CU6 must be among them. Any build
  that survives is RED for that build, and you report why.
- **G4 (cost):** added default-suite wall time is ≤ 30 s; anything slower sits behind `TC_SLOW_TESTS=1`.

## Rules

- HOUSE_RULES (`/mnt/Shared/HOUSE_RULES.md`) apply. RED honesty.
- Prior Art Directive (AGENTS.md): keep LD3's annotations at the code site. The main ones are Dorfman (1943) and
  non-adaptive group testing (Du & Hwang), identity-V (LD1/LD2), and DeMillo, Lipton & Sayward (1978).
- Keep to about 1.5 hours.

## Done

Write `docs/APA_REGT2_REPORT.md`. If the harness refuses report files from subagents, put the report in your final
message. The final message gives:
- G1–G4;
- the per-test duration table;
- the known-bad table;
- the full-suite comparison and wall times;
- the guard demonstration;
- every difference from the LD3 source, with its reason;
- the files and diff stat;
- deviations;
- prior art;
- receipts cited as `path:line "quote"`.
