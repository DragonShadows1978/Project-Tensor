# APA-SPD1 implementation ledger

Append-only. Model id: `gpt-6-astra`; effort: high. Writable worktree:
`/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1`. Branch `apa-spd1` is supplied by
dispatch; no git commands were used to verify or change it. Production read-only.

## 2026-09-06 — Orientation and registration, before gates

- Read local `AGENTS.md`, `/mnt/Shared/HOUSE_RULES.md`, and lead's prior-art comparison.
- Evidence class: source/receipt inspection. Reused SP1's 48 shape dictionaries,
  calibration values and inherited build. All 17 source hashes in the inherited
  build manifest matched current production sources. No rebuild needed; this seat
  builds the benchmark plumbing, not a new kernel generation.
- Evidence class: CPU import probe. Installed torch reports `2.11.0+cu130`, CUDA
  build `13.0`, `torch.cuda.is_available() == False`, `flash_attn` absent. Runtime
  imports from `artifacts/apa_sp1/build/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`.
  An orientation probe calling `tc.tensor([1.0], device="cpu")` failed verbatim:
  `RuntimeError: CUDA error at from_host: no CUDA-capable device is detected`.
  Engine CPU allocation is not a usable oracle; G1 uses a NumPy test double for
  composition and real torch CPU math. This was an environment probe, not G1/G2.
- Evidence class: filesystem inspection. No
  `/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2` directory exists at inspection.
  Only that authorized discovery path and SP2 script-name search were read in main;
  no writes to main or another worktree.
- Registration created exclusively before any gate. SHA-256:
  `178a45cb99cb091b8514f05a1270baba53ae12786bba66054bd73c1c8e8b95d0`.
  File: `artifacts/apa_spd1/registration.json`; 50 cells; 24 production file pins;
  lead P1–P4 and seat A1–A5 are inside. Registration remains immutable.
- Interpretation: two E1 extras are H=16/KV=4, causal, D=128, L=512,
  S=8192/32768, preserving E1's actual query-head geometry. SP1's 48 cells include
  both causal flags for decode too.
- RED calibration premise: SP1 used K+0.1-noise, not TurboQuant; E1 used symmetric
  INT4 r=.10, not SP1 r=.15. Main grid uses registered TurboQuant4 MSE reconstruction
  and r=.15; E1 extras use symmetric INT4/r=.10. Frozen SP1 deltas are retained;
  no recalibration or timing-guided tuning. Rectangular extras transfer full-context
  decode deltas, explicitly UNCALIBRATED. Optional clarification was posted; after
  an opportunity to respond, transfer was stated as the assumption before registration.
  Fraction mismatch is measured/flagged; no guaranteed matched-budget claim.
- Registered measurement execution detail: 12 rows/cell (FP32/BF16 engine dense,
  torch math, APA two-pass, APA-SP; BF16 torch efficient and flash; optional BF16
  flash_attn; optional FP32 SP2). Three peak calls per available row, separate from
  nine timed calls. Full output metrics use engine FP32; same-dtype timing ratios
  use added BF16 baselines. Inputs are identical rounded values in separate runtime
  allocations, not zero-copy aliases. Default stream 0 selected explicitly.
- Stop rails: no GPU gate in this seat; source drift, busy lease, failed nvidia-smi,
  non-sm_89 device, missing runtime or nonfinite reference blocks the job. No
  automatic retries. SP2 absence skips; appearance without an actual pinned launcher
  adapter reports `BLOCKED_SP2_INTERFACE`, never substitutes SP1 or guesses a δ.

## Prior art — implementation sites

`apa_spd1_common.py`: E1 grouped dense (Project-Tensor, 2026), Vaswani et al.
(2017); PyTorch SDPA/backend and lower-right causal APIs (2023–2026); SP1/BLASST
running-max rule (Yuan et al., 2025/v3 2026); standard Frobenius norms, NumPy type-7
quantiles (Hyndman & Fan, 1996), SHA-256 (NIST, 2015). Own work: registry, common
inputs, validation and receipt contracts; no new attention mathematics.

`apa_spd1_bench.py`: FlashAttention-2 (Dao, 2023), TurboQuant (Zandieh et al., 2025),
Lloyd-Max (Max 1960/Lloyd 1982), E1 uniform INT4 reconstruction (2026), APA two-pass
and SP1/SP1.1 (David and seats, 2026), online normalizer (Milakov & Gimelshein 2018),
Flash-Decoding (2023), related ThriftAttention (Sharratt, 2026). Own work: common
measurement integration, separate peak pass and explicit calibration qualifications.

`apa_spd1_lead_gpu.sh` / report: SP1 one-cell leased execution, util-linux flock,
GNU timeout, NumPy median/IQR and E1/SP1 receipts; own work: cross-contender table,
optional rows, append-only attempts, missing-data-aware predictions. Counterbalanced
order uses a standard experimental-design idea (Fisher 1935); no novelty asserted.

G1 tests/mutations: independent dense SDPA oracle and SP1 boundary test patterns;
HOUSE_RULES §8 mutation discipline. Own adversarial fixtures, no blind validation
claim. Bibliographic verification and unverified search leads for all of the above
are recorded in `artifacts/apa_spd1/PRIOR_ART.md` with primary-source URLs.

## 2026-09-06 — G1 execution receipts and final handoff

- Evidence class: CPU suite. Initial command:
  `timeout 60s env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B -m pytest -q -p no:cacheprovider tensor_cuda/tests/test_apa_spd1.py --junitxml=artifacts/apa_spd1/G1_BASELINE.xml`.
  Exit 0; `G1_BASELINE.log`: **22 passed in 2.12s**. No failures/skips.
- Evidence class: CPU mutation suite, run only after the passing baseline:
  `timeout 290s env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B scripts/apa_spd1_mutations.py`.
  Initial receipt `G1_MUTATIONS.json`: all five registered non-error mutants
  killed; **1.0 ≥ .80**; source unchanged. Mutants ran in temporary copies, with
  40-second subprocess bounds. Mutation prior-art lead: DeMillo, Lipton & Sayward
  (1978), unverified — search “Hints on Test Data Selection Help for the Practicing
  Programmer 1978”. This is an author baseline, not blind verification.
- Evidence class: code inspection and stronger plumbing checks. Final review found
  that “all worker collections complete” alone was insufficient to declare G2 PASS:
  a numerical RED or an optional SP2 table with a missing interface must block it.
  Added an explicit aggregate verdict guard and one test; no prediction tolerance
  or registration changed. Also pin all protected source/receipt bytes in the
  runtime fingerprint, reject unsupported status values/required unavailability,
  retain partial timing samples on failure, and reject GPU peak telemetry below
  the still-live output's bytes. The GPU telemetry guard is not GPU-verified here.
- Evidence class: CPU suite. `G1_FINAL_BASELINE.log`: **23 passed in 2.07s**.
  Final common-source mutation run:
  `artifacts/apa_spd1/mutations/1788669994816705287/result.json` — five non-error
  mutants killed, **1.0**, unchanged common source hash
  `69a62a676b2b98697978735e64ce77230ad66fee231d7b09211b682661f95a5e`.
  Mutation runner now stores each new run under a unique timestamp; initial
  receipts remain intact. This is an artifact-retention improvement, no gate change.
- Evidence class: final CPU suite on delivered benchmark code. Same timeout/env
  command as initial baseline, XML target `G1_RELEASE_BASELINE.xml`;
  `G1_RELEASE_BASELINE.log`: **23 passed in 2.09s**, 0 failures/errors/skips.
  `G1_RESULTS.json` records exact counts, duration 2.089 seconds, current source
  fingerprints, final mutation receipt and all validation checks.
- Evidence class: dry-run and shell syntax. `timeout 10s bash -n
  scripts/apa_spd1_lead_gpu.sh` exited 0. `timeout 55s bash
  scripts/apa_spd1_lead_gpu.sh --dry-run` exited 0 and enumerated **50 unique cells**,
  exactly matching registration; output `G1_FINAL_DRY_RUN.jsonl`. The suite also
  runs dry-run with torch import blocked entirely.
- Evidence class: report writer. `timeout 55s bash scripts/apa_spd1_lead_gpu.sh
  summary` exited 0; `G1_RELEASE_SUMMARY.log` reports **0/50** GPU receipts, no
  rejected receipts. `SPEED_CHAIN.md` and `speed_chain.json` contain **600** explicit
  blocked row slots; no timing values were fabricated. Writer fingerprint equals
  the delivered code/source fingerprint. Numerical RED and blocked SP2 coverage
  cannot become a green G2 from receipt completeness alone.
- Evidence class: final CPU inventory. `timeout 55s bash
  scripts/apa_spd1_lead_gpu.sh inventory` exited 0; `G1_RELEASE_INVENTORY.log`
  and `CPU_INVENTORY.json`: torch 2.11.0+cu130, CUDA unavailable, flash_attn absent,
  no frozen SP2 file. Entry-point availability is source/import evidence only.
- Evidence class: integrity. Registration SHA remains
  `178a45cb99cb091b8514f05a1270baba53ae12786bba66054bd73c1c8e8b95d0`;
  SP1 calibration's own SHA sidecar matches; all **24 production files** and all
  inherited source/receipt pins remain unchanged. No production or other-worktree
  write, git command, subagent, background wait, service operation, GPU gate,
  or process kill occurred. Every compute invocation had an explicit timeout.
- G2 remains **BLOCKED**, expected by dispatch. Exact lead commands and safety
  envelope are in `artifacts/apa_spd1/G2_BLOCKED.md`; machine-readable receipt in
  `G2_BLOCKED.json`. Final synthesis is `artifacts/apa_spd1/REPORT.md`; scope,
  calibration transfer, allocator limits and absent actual SP2 interface remain RED
  residuals. No model-quality, GPU support, timing, memory-win, or blind-review
  result is claimed from G1.
