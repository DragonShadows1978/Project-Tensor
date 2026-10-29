# APA-DK-STE5 — STAGED; GPU acceptance PENDING

S5-G1 PASS: 34 CPU tests. S5-G2 commands staged; G-G1..G-G7 and timing NOT RUN by this CPU seat. S5-G3 PASS for complete staged disclosure and actual bundle hashes. Overall **PENDING, not ACCEPT**. No GPU, git, subagents, kernel change, .so change, engine installation or production promotion.

Immutable plan: orders/APA_DK_STE5.md, with STE4 mechanism and REV1 findings. Thresholds are the lead's; this seat registers no gate. INPUT_PINS.json precedes CPU gates and future lead captures; immutable INPUT_PINS_AMENDMENT_001.json records the isolated-subprocess dependency fix and effective provenance digest. No threshold was changed.

## Evidence and limits

- CPU unit/integration evidence: receipts/S5_CPU.json, cpu.xml and cpu.log: 34 passed. Required one-bit G-G2, slope .97, missing governor negatives rejected. Real old small h_g1 exports and independent REV1 oracle agree; these are CPU reference checks, not fresh G-gates.
- Initial CPU attempt RED: 13 failed, 21 passed because Python -I hid user-site NumPy (`ModuleNotFoundError: No module named 'numpy'`). Retained cpu_attempt1.log/xml. Explicit existing NumPy parent supplied to isolated subprocesses; all rejection assertions preserved.
- F4 fixed at source and copied byte-identically into engine: real subprocess value 2 raises `ImportError: tensor_cuda: TC_APA_DK_STE must be unset, 0, or 1`, never the fallback ModuleNotFoundError. Empty, whitespace, true and stray fallback tested too; 0/1/unset still import. F4.diff and receipts/F4_BEFORE.txt/F4_AFTER.txt retain the treatment evidence. Only import-error handling changed.
- CPU full GRAPA fixture float64 reference: receipts/REFERENCE_PREFLIGHT.json PASS, 134,250,496 visible pairs; selective and all controls healthy, all visible pairs selected for -1e6. 44.25 s CPU preflight; no GPU result implied.
- Pinned inputs, staged shell syntax, manifest package hashes and both actual .so hashes checked. receipts/VERIFY5_PENDING.json returns rc 2, PASS false, missing base governor rows; no vacuous acceptance.
- F1 at actual GRAPA shape is **not claimed fixed** until lead captures and G-gates pass. F5 GRAPA environment scrubbing remains outside scope. F3 arm step time comes from GRAPA own smoke; no step projection from APA timing.

## Mechanism / comparisons

GRAPA launcher cc78_launch.py:230-234 selects backward g1 / forward h, a_base included. Every worker sets/asserts both. Direct h forward and g1 backward calls use the same kernels as training dispatch; ops.cpp:789-795, 804-813, 1184-1188. Environment -> ops.cpp:21-31 -> bindings.cpp:221 -> kernels.cu:3157-3159 is unchanged. G1 uses saved-output rowdot and BF16 coefficient rounding; no f_pass_a bound is transferred to it.

OPERAND_CENSUS.md enumerates every operand/rule: 756 byte rows, 27 full-element signal rows, 27 whole-key witnesses across three shapes, six configurations, three fresh processes/config/suite (36 processes and 54 files). Full uint16 BF16 and uint32 FP32 bytes, including signed zeros. G-G5 ideal float64 reference uses BF16-decoded inputs; no captured gradient or saved statistic defines it, and no uncertain/zero element is masked. G-G7 alone uses a conservative score envelope against the actual saved threshold to certify an entirely unselected key. Every capture needs a unique governor ran/rc0 row and matching provenance.

G-G5 thresholds unchanged: cosine >= .995, zero-intercept LS slope in [.98,1.02], on relative-L2 <= off relative-L2 + .01 against their own full references. On/off dk per-element BF16-ULP percentiles 0/50/90/95/99/99.9/100 will be reported. No current G-gate numbers or witness exist.

## Lead execution

Run GPU_COMMANDS.sh exactly in its listed order (verbatim below). Base-only jobs and CPU premise check precede every arm; any failed premise stops the shell, and each arm worker requires its matching premise receipt. Each GPU job uses forge-gpu-run --max 600 and timeout --foreground 300s; capture batch has a 270s internal budget. No budget is a measured runtime promise. Labels are unique and receipts refuse overwriting.

Timing is an h forward plus g1 backward pair at GRAPA shape, switch off then on, one warmup and two samples each. verify5 recomputes means and difference. Lead own GRAPA smoke supplies arm step time. verify5 then rehashes stored operands, recomputes all G-gates, and refreshes manifest/BUNDLE only on full PASS plus CPU receipt checks. Nothing is published.

```bash
#!/usr/bin/env bash
# LEAD ONLY: do not run on the CPU seat.
# Prior art: STE1/4 governed captures (2026); GNU timeout (2008+, unverified - lead to check).
set -euo pipefail
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1
export TMPDIR="$PWD/artifacts/apa_ste5/scratch"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/apa_ste5/lead
export FORGE_GPU_RUN_LOG="$PWD/artifacts/apa_ste5/lead/governor.tsv"
CUDA_VISIBLE_DEVICES= timeout --foreground 120s python3 -B artifacts/apa_ste5/stage5.py --check
# 1. Base only, then G-G1 premise. STOP on any failure.
forge-gpu-run --min 300 --max 600 --label ste5v1-base-small -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config base --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-base-grapa -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config base --suite grapa
CUDA_VISIBLE_DEVICES= timeout --foreground 300s python3 -B artifacts/apa_ste5/verify5.py --premise --output artifacts/apa_ste5/lead/PREMISE.json
# 2. Arm configurations only after premise PASS.
forge-gpu-run --min 300 --max 600 --label ste5v1-off-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-off-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-unset-small -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config unset --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-unset-grapa -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config unset --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-on-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-on-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-off-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off-all --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-off-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off-all --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-on-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on-all --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-on-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on-all --suite grapa
# 3. Timing: one h forward + g1 backward pair; warmup, then 2 samples.
forge-gpu-run --min 300 --max 600 --label ste5v1-timing-0 -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/timing5.py --switch 0
forge-gpu-run --min 300 --max 600 --label ste5v1-timing-1 -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/timing5.py --switch 1
# 4. CPU verify recomputes gates, timing difference, then refreshes bundle only on PASS.
CUDA_VISIBLE_DEVICES= timeout --foreground 1800s python3 -B artifacts/apa_ste5/verify5.py --output artifacts/apa_ste5/lead/VERIFY5.json
```

## S5 status

| gate | current result |
|---|---|
| S5-G1 | PASS: 34 CPU tests |
| S5-G2 | STAGED; G-G1..G-G7 and timing pending lead |
| S5-G3 | PASS: complete staged disclosure and actual bundle hashes |
| Registered acceptance | PENDING, not ACCEPT |

G-G1..G-G7 are all NOT RUN. The premise will stop and report RED if base/base differs. Costs, witnesses and new ULP percentiles remain unmeasured until lead captures. No lead captures were awaited.

CPU command (from target; CUDA hidden):

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR="$PWD/artifacts/apa_ste5/scratch" timeout --foreground 180s python3 -B -m pytest -q -o cache_dir=artifacts/apa_ste5/scratch/pytest_cache --basetemp=artifacts/apa_ste5/scratch/pytest-r2 --junitxml=artifacts/apa_ste5/receipts/cpu.xml artifacts/apa_ste5/tests
```

## F4 diff

```diff
--- a/tensor_cuda/tensor_cuda/__init__.py
+++ b/tensor_cuda/tensor_cuda/__init__.py
@@ -14,9 +14,16 @@
 
 import numpy as np
 
+# Prior art: Python importlib.import_module / ModuleNotFoundError (Python 3.6,
+# 2016; unverified - lead to check). Preserve extension initialization errors;
+# only a missing packaged extension permits the existing top-level fallback.
+from importlib import import_module as _import_module
+
 try:
-    from . import _tensor_cuda as _C
-except ImportError:  # pragma: no cover
+    _C = _import_module(f"{__name__}._tensor_cuda")
+except ModuleNotFoundError as _extension_error:  # pragma: no cover
+    if _extension_error.name != f"{__name__}._tensor_cuda":
+        raise
     import _tensor_cuda as _C  # fallback when the .so sits on PYTHONPATH
 
 Tensor = _C.Tensor
```

## Manifest disclosure

MANIFEST.diff contains the complete manifest diff. DISCLOSURE.json is the exact disclosure section; variant-a metrics distinguish per-element ULPs and common BF16 accumulation error from run ordering. Earlier STE4 text naming a as GRAPA's variant is superseded by REV1 and this disclosure. f_pass_a exact controls and conditional bound are kept separate from pending g1/h evidence.

```json
{
  "1_variants": {
    "GRAPA": "backward g1 / forward h, including a_base; scripts/cc78_launch.py:230-234 in grapa_wt_cc80",
    "a": "STE3 envelopes and float64 comparisons; REV1 oracle and accumulation-error disclosure below. Not GRAPA arm settings.",
    "f_pass_a": "STE4 lead/VERIFY4.json ACCEPT: all D-G1..D-G6 PASS on small1, small2 and GRAPA shape, with forward a. Exact byte controls plus conditional elementwise bound; not a g1 proof.",
    "g1": "STE5 stages g1/h on small1, small2 and B1 H16 KVH16 L=S4096 D128 VD64 causal BF16; 3 fresh processes/config. GPU evidence PENDING lead; no GRAPA-shape g1 claim yet. Receipt after run: artifacts/apa_ste5/lead/VERIFY5.json.",
    "others": "STE1 small-shape evidence only; no GRAPA-shape validation or compatibility claim for other variants."
  },
  "2_switch": {
    "inherited_environment": "An inherited 1 enables STE even if an arm omits the variable. GRAPA-side scrubbing/assertion is F5 and remains outside this engine change.",
    "invalid": "Other strings raise ImportError naming TC_APA_DK_STE and allowed unset, 0, or 1; no fallback masks extension initializer errors.",
    "path": "ops.cpp:21-31 -> bindings.cpp:221 -> kernels.cu:3157-3159 (g1). Unchanged .so af9ffce4; F4 changes only Python import-error handling.",
    "rule": "On adds dScore*scale*q for visible unselected pairs to dK through identity STE of quantized K. Selected pairs, forward, dQ and dV keep their rules; finite gates test resulting bytes.",
    "semantics": "TC_APA_DK_STE unset or 0 = off (shipped default); 1 = on. Read once during extension import, cached process-wide. Change requires a fresh process. Experiment engine."
  },
  "3_variant_a_BF16_accumulation": {
    "delta_certain_key_elements": {
      "cosine": 0.9968871417811772,
      "n": 6943232,
      "ref_norm": 4.202263381360506,
      "relL2": 0.07884180873722425,
      "slope": 0.9937189568366868
    },
    "envelopes": "artifacts/apa_ste4/VARIANT_A_DISCLOSURE.json and .md (5 runs/config, all within/cross pairs); REV1 findings F2 supersede their atomic-order-only wording.",
    "grapa_off_certain_key_elements": {
      "maxabs": 0.0025969468999907375,
      "n": 6943232,
      "relL2": 0.03189101334890789,
      "ulp_elem_max": 6487891.2712819055,
      "ulp_elem_p50": 2.813392389381022,
      "ulp_elem_p99": 110.09266765295605,
      "ulp_elem_p999": 1121.9107775427544
    },
    "grapa_on_certain_key_elements": {
      "maxabs": 0.008524584587828912,
      "n": 6943232,
      "relL2": 0.070597991543809,
      "ulp_elem_max": 13238061.688818395,
      "ulp_elem_p50": 7.030622852382848,
      "ulp_elem_p99": 272.34479887405763,
      "ulp_elem_p999": 2732.5062667774428
    },
    "grapa_run0_all_elements": {
      "off_common_part_relL2_mean2": 0.02959131304783189,
      "off_error_relL2_run0": 0.030358750436514658,
      "off_runtorun_relL2": 0.01395537416542289,
      "off_slope_vs_R_off": 0.9987426752873071,
      "on_common_part_relL2_mean2": 0.06356319024441412,
      "on_error_relL2_run0": 0.06564382621163437,
      "on_runtorun_relL2": 0.03358182632852732,
      "on_slope_vs_R_on": 0.9946628044220391
    },
    "scope": "REV1 oracle.json certain-key metrics exclude 11,292 keys with selection uncertainty: 6,943,232 elements retained; order_vs_bias.json split uses all elements of two runs. On rel-L2 about 6.56-7.06%, off 3.04-3.19%; per-element on ULP median 7.03, p99 272.34 on certain keys. Do not confuse ULP at max reference magnitude with per-element percentiles.",
    "summary": "BF16-destination accumulation error, with both common error and run-to-run ordering spread; not merely atomic-order noise. Base dK/dV are not bitwise repeatable. At refine-all the on/off dK computation is the same, subject to accumulation ordering."
  },
  "4_STE4_bound": {
    "BOUND_sha256": "136dfcd30dec86444bd8a79500066b04b592bc2f7a37ffa683bba90cd1896628",
    "conditions": "f_pass_a only: FP32 accumulation/key ownership then one BF16 RN store; decoded BF16 inputs, finite no-overflow IEEE RN arithmetic; stated CUDA exp/log/sqrt contracts and domains. Not a g1 bound. No mask in the STE4 element gate. Finite validation does not prove all-input or library-contract correctness.",
    "formula": "For every element: |decode(dk)-R_rule| <= Epre + 2^-8*(|R_rule|+Epre) + 2^-134, with outward float64 guards. Epre includes coefficient, selection-interval, FP32 reduction, float64 evaluation and FTZ floors; see complete derivation BOUND.md.",
    "receipt": "artifacts/apa_ste4/BOUND.md; lead/VERIFY4.json; REV1 K3 confirms zero element failures and byte controls"
  },
  "5_cost": {
    "g1_h": "PENDING lead timing: one APA forward h + backward g1 pair at GRAPA shape, off/on, 2 samples after one warmup; verify5 reports raw samples, means and on-minus-off difference.",
    "prior_variant_a": "REV1 K5: about +10.2% backward single-call kernel delta (n=2, 14 ms spread), inapplicable to g1/h. No g1 step cost inferred.",
    "step_time": "GRAPA arm step time comes from GRAPA own smoke. A single APA-call delta is not a step-time projection."
  },
  "6_provenance": {
    "arm": "af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760; no .so or kernel change in STE5. Identical F4 Python file at source and in engine package.",
    "base": "69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c; source baseline f2c686e plus three restored untracked files, byte-identical to base build inputs.",
    "registration": "STE4 registration was committed with data; temporal ordering rests on registered_utc, mtimes and capture-recorded hashes (REV1 F6). STE5 thresholds are the immutable lead order; INPUT_PINS.json is written before CPU gates and any lead capture. This seat registers no new gate; future amendments must be separate files.",
    "restored_sources": {
      "include/tc/deterministic_gather.h": {
        "from": "/mnt/ForgeRealm/Project-Tensor/WorkTree/tensor_wt_det_cert/tensor_cuda/include/tc/deterministic_gather.h",
        "sha256": "0bab127a26b3c775024ea8ffa0c5c5e629b2cf2609f42611c120d70b68aff95c"
      },
      "src/embed_deterministic.cu": {
        "from": "/mnt/ForgeRealm/Project-Tensor/WorkTree/tensor_wt_det_cert/tensor_cuda/src/embed_deterministic.cu",
        "sha256": "44e7182a5e00fec534f681dabc4349c3bef433612f4e7355e0cb2731f70529b7"
      },
      "src/gather_deterministic.cu": {
        "from": "/mnt/ForgeRealm/Project-Tensor/WorkTree/tensor_wt_det_cert/tensor_cuda/src/gather_deterministic.cu",
        "sha256": "05ce1d3e814521823f8f77bdff347c632112f4ba3f290246eaf9fa77870b4758"
      }
    }
  }
}
```

## BUNDLE.json

```json
{
  "manifest": "artifacts/apa_ste1/engine/ENGINE_MANIFEST.json",
  "manifest_sha256": "5adcaa54901bb80f7a6b8fa24da511f176b481c3e8d12c7c2d4dd68e7c7488d4",
  "schema": "apa-ste5-bundle-v1",
  "so_sha256": "af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760",
  "status": "STE5 staged; g1/h GPU gates and timing PENDING; not accepted or promoted"
}
```

## Prior art

Bengio, Leonard and Courville (2013), STE identity derivative; Dao et al. (2022), FlashAttention attention VJP/recomputation; Higham (2002), gamma_n / dot error and float64 reference analysis; IEEE 754-2019, widths and ULP spacing; NIST FIPS 180-4 (2015), SHA-256; DeMillo, Lipton and Sayward (1978), injected-defect tests; Python importlib/ModuleNotFoundError (Python 3.6, 2016) and isolated interpreter (3.4, 2014); NVIDIA CUDA events (2007+); GNU timeout (2008+). External bibliography **unverified - lead to check** these names and years. Existing STE1/STE4/REV1 fixture, capture, byte-gate and oracle machinery is reused. This round contributes g1/h harness integration, full unmasked signal accounting, import-error propagation and disclosure; no new kernel algorithm. No prior art known to me for the lead's particular G-gate protocol. Code sites and ledger carry these annotations.

# Every file written or retained by APA-DK-STE5

All new files are under artifacts/apa_ste5/. Three authorized existing files changed: source __init__.py, identical engine __init__.py, ENGINE_MANIFEST.json.
Scratch and first failing CPU receipts are retained. No deletions or DELETE_ME moves. Pytest-current symlinks are listed separately as filesystem entries.

- `artifacts/apa_ste1/engine/ENGINE_MANIFEST.json`
- `artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/__init__.py`
- `artifacts/apa_ste5/BUNDLE.json`
- `artifacts/apa_ste5/DISCLOSURE.json`
- `artifacts/apa_ste5/F4.diff`
- `artifacts/apa_ste5/FILES_WRITTEN.json`
- `artifacts/apa_ste5/FILES_WRITTEN.md`
- `artifacts/apa_ste5/FINAL_REPORT.md`
- `artifacts/apa_ste5/GPU_COMMANDS.sh`
- `artifacts/apa_ste5/INPUT_PINS.json`
- `artifacts/apa_ste5/INPUT_PINS_AMENDMENT_001.json`
- `artifacts/apa_ste5/LEDGER.md`
- `artifacts/apa_ste5/MANIFEST.diff`
- `artifacts/apa_ste5/OPERAND_CENSUS.md`
- `artifacts/apa_ste5/STAGING.json`
- `artifacts/apa_ste5/bundle5.py`
- `artifacts/apa_ste5/capture5.py`
- `artifacts/apa_ste5/common.py`
- `artifacts/apa_ste5/pin5.py`
- `artifacts/apa_ste5/receipts/F4_AFTER.txt`
- `artifacts/apa_ste5/receipts/F4_BEFORE.txt`
- `artifacts/apa_ste5/receipts/INIT_BEFORE.py`
- `artifacts/apa_ste5/receipts/ORIGINAL_ENGINE_MANIFEST.json`
- `artifacts/apa_ste5/receipts/REFERENCE_PREFLIGHT.json`
- `artifacts/apa_ste5/receipts/S5_CPU.json`
- `artifacts/apa_ste5/receipts/S5_STATUS.json`
- `artifacts/apa_ste5/receipts/VERIFY5_PENDING.json`
- `artifacts/apa_ste5/receipts/VERIFY5_PENDING.md`
- `artifacts/apa_ste5/receipts/cpu.log`
- `artifacts/apa_ste5/receipts/cpu.xml`
- `artifacts/apa_ste5/receipts/cpu_attempt1.log`
- `artifacts/apa_ste5/receipts/cpu_attempt1.xml`
- `artifacts/apa_ste5/receipts/reference_preflight.log`
- `artifacts/apa_ste5/receipts/verify5_pending.log`
- `artifacts/apa_ste5/reference5.py`
- `artifacts/apa_ste5/scratch/add_timing.py`
- `artifacts/apa_ste5/scratch/amend_test.py`
- `artifacts/apa_ste5/scratch/cpu_receipts.py`
- `artifacts/apa_ste5/scratch/finish_code.py`
- `artifacts/apa_ste5/scratch/implement.py`
- `artifacts/apa_ste5/scratch/make_verify.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_duplicate_governor_fails0/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest-r2/test_duplicate_governor_failscurrent`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc0/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc1/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc2/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc3/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc4/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc5/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc6/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc7/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc8/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproc9/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest-r2/test_f4_real_extension_subproccurrent`
- `artifacts/apa_ste5/scratch/pytest-r2/test_missing_governor_failscurrent`
- `artifacts/apa_ste5/scratch/pytest-r2/test_missing_governor_whole_ve0/missing.json`
- `artifacts/apa_ste5/scratch/pytest-r2/test_missing_governor_whole_ve0/missing.md`
- `artifacts/apa_ste5/scratch/pytest-r2/test_missing_governor_whole_vecurrent`
- `artifacts/apa_ste5/scratch/pytest-r2/test_wrong_governor_outcome_ra0/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest-r2/test_wrong_governor_outcome_ra1/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest-r2/test_wrong_governor_outcome_racurrent`
- `artifacts/apa_ste5/scratch/pytest-r2/test_wrong_governor_outcome_re0/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest-r2/test_wrong_governor_outcome_recurrent`
- `artifacts/apa_ste5/scratch/pytest/test_duplicate_governor_fails0/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest/test_duplicate_governor_failscurrent`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc0/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc1/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc2/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc3/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc4/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc5/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc6/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc7/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc8/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproc9/_tensor_cuda.py`
- `artifacts/apa_ste5/scratch/pytest/test_f4_real_extension_subproccurrent`
- `artifacts/apa_ste5/scratch/pytest/test_missing_governor_failscurrent`
- `artifacts/apa_ste5/scratch/pytest/test_missing_governor_whole_ve0/missing.json`
- `artifacts/apa_ste5/scratch/pytest/test_missing_governor_whole_ve0/missing.md`
- `artifacts/apa_ste5/scratch/pytest/test_missing_governor_whole_vecurrent`
- `artifacts/apa_ste5/scratch/pytest/test_wrong_governor_outcome_ra0/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest/test_wrong_governor_outcome_ra1/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest/test_wrong_governor_outcome_racurrent`
- `artifacts/apa_ste5/scratch/pytest/test_wrong_governor_outcome_re0/governor.tsv`
- `artifacts/apa_ste5/scratch/pytest/test_wrong_governor_outcome_recurrent`
- `artifacts/apa_ste5/scratch/pytest_cache/.gitignore`
- `artifacts/apa_ste5/scratch/pytest_cache/CACHEDIR.TAG`
- `artifacts/apa_ste5/scratch/pytest_cache/README.md`
- `artifacts/apa_ste5/scratch/pytest_cache/v/cache/lastfailed`
- `artifacts/apa_ste5/scratch/pytest_cache/v/cache/nodeids`
- `artifacts/apa_ste5/scratch/reference_preflight.py`
- `artifacts/apa_ste5/scratch/report5.py`
- `artifacts/apa_ste5/stage5.py`
- `artifacts/apa_ste5/tests/test_verify5.py`
- `artifacts/apa_ste5/timing5.py`
- `artifacts/apa_ste5/verify5.py`
- `tensor_cuda/tensor_cuda/__init__.py`
