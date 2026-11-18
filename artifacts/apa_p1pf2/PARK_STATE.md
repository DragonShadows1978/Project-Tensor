# APA-P1-PF2 park state (2026-11-16 ~16:40 EST, parked on SC's 16:30 usage stop)

## Has run
- r1: Build PF (`c3c27f86…`), Build PFH (`e54b39c3…`), SASS PASS (781 equal, 32 added, as registered),
  G1 lines (GPU 1.03 h): PF-I1 PASS, PF-I1n PASS, PF-I2 PASS, PF-I3 FAIL under the 1e-5 cap (r1 report: REPORT_r1.md).
- Amendment 1, A1.1 (CPU, from r1's G1 parts): PF-I3 **PASS**: fp32 tau 1.564e-4 (max at delta 4.11e-5), bf16 tau
  1.351e-2 (max 3.17e-3), 0 rows above. Receipt `a1/PF_I3_A1.json` (sha256 b83f8bf2…), script `scripts/apa_p1pf2_a11.py`.
- G7 mutants: `mutants/MUTANTS.json` (4771bd3a…, written before staging). Staged: m01–m06. Built (PASS): m01 b081a6d3,
  m02 807fbbf0, m04 ca6c5e8d, m05 00d9d6a0, m06 382e0461 (incremental, base Build PF). m03 is a Python mutant (staged,
  no build needed). The build driver was stopped by me after m06 (no build running).
- CPU: `CALLER_CENSUS.json` (L90: 1 product caller, apa_sp2.apa_selective_attention_sp; tests/scripts listed only).
- Scripts written, not yet run on GPU: `apa_p1pf2_g2.py` (G2), `apa_p1pf2_q1.py` (G3; top-1 boundary selftest passed on
  CPU), `apa_p1pf2_g45.py` (G4 rows + BOUNDARY.json, G5 census with an LD_PRELOAD allocation/launch shim),
  `apa_p1pf2_g6.py` (LF1, L2/L4, LF6, pins), `apa_p1pf2_mut.py` (G7 runner, mutant + control per mutant),
  `apa_p1pf2_speed.py`.

## Pending (GPU budget left: 5 h - 1.03 h = 3.97 h)
1. A1.2: m12 must fail A1.1's PF-I3 (G7 line) — G1's verdict under A1 depends on it.
2. Builds: m07–m14, m16 (m13 on Build PFH). Python: m15 stage.
3. GPU lines: G2, G3 (time one window first, L42), G4, G5, G6 (LF1, L2/L4, LF6 with F3/PF; pins on CPU), G7, speed.

## Exact next step to resume (from the worktree root, CUDA_VISIBLE_DEVICES empty for CPU steps)
1. Edit `/tmp/apap1pf2_main/mutbuild.sh` (or recreate it from the loop in this file's history: stage + build per
   mutant) to start at m07, and run it detached:
   `setsid nohup /tmp/apap1pf2_main/mutbuild.sh > artifacts/apa_p1pf2/mutants/BUILD_DRIVER_r2.log 2>&1 < /dev/null &`
   (stage commands fail on m01–m06 because their trees exist; that is expected — skip them in the list).
2. Stage arms: `python3 -B -P scripts/apa_p1pf2_gpu.py stage mutant:<name>` for each mutant; build the shim:
   `python3 -B -P scripts/apa_p1pf2_g45.py shim`.
3. GPU lines, each via `scripts/apa_p1pf2_line.py <LINE> <attempt> <max> <timeout> <arms> -- <script> ...`:
   G7 `apa_p1pf2_mut.py mut --out artifacts/apa_p1pf2/gpu/G7` (m12 first: `--only m12_unrefined_dropped`);
   G4 `apa_p1pf2_g45.py g4 --out .../gpu/G4`; G5 `apa_p1pf2_g45.py g5 --out .../gpu/G5`;
   G6 `apa_p1pf2_g6.py lf1|l2l4|lf6 --out ...` (LF6 files: P1-F3's LF6a/LF6b lists); `apa_p1pf2_g6.py pins` (CPU);
   G2 `apa_p1pf2_g2.py --out .../gpu/G2`; G3 `apa_p1pf2_q1.py job --model <M> --out .../gpu/G3 --windows 0`
   (time it), then the remaining windows, then `apa_p1pf2_q1.py report --out .../gpu/G3` (CPU);
   speed `apa_p1pf2_speed.py artifacts/apa_p1pf2/gpu/SPEED`.
None of these new scripts has run yet; expect first-run repairs (L56b budget used so far: 2 of 3 in r1).

## Deviations since r1
- The REPORT.md write was refused by the harness in r1. This file is the park record.
- Mutant build driver stopped by me after m06 (park order); m06's in-flight build was allowed to finish (PASS).
- `scripts/apa_p1pf2_build.py`: mutant base selection (m13 on PFH); `scripts/apa_p1pf2_gpu.py`: MUTANTS.json path fix.
- LF6 runs both arms from this tree's tests (identical to 2945c3e9's tests, checked by `diff -rq` against a git
  archive at /tmp/apap1pf2_tree0).
- G4 evidence for native refusals: 0 launches and 0 allocations (shim), not 0 spied calls (the native entry is
  called and refuses before any allocation); Python refusals: 0 spied calls.
