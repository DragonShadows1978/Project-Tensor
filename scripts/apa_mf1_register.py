"""APA-MF1 registration writer: create-only JSON + .sha256 (refuses to overwrite)."""
import hashlib, json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts/apa_mf1")
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
Ls = [1, 7, 64, 512]
shapes = [[L, S] for L in Ls for S in (L, L + 1, L + 100, 4 * L + 3)]
reg = {
  "order": "APA-MF1 rectangular causal masks (orders/APA_MF1_RECTANGULAR_CAUSAL_MASKS.md)",
  "seat": "Claude Opus 5.5 (Agent-tool subagent)",
  "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "worktree": ROOT,
  "base_commit": "fd32b6d (production; before copies byte-equal to the production checkout)",
  "before_copies_sha256": {f: sha(os.path.join(ART, "before", f)) for f in sorted(os.listdir(os.path.join(ART, "before"))) if f != "SHA256SUMS"},
  "engine_build": {"script": "scripts/apa_mf1_build.sh", "module": "artifacts/apa_mf1/build/_tensor_cuda.cpython-312-x86_64-linux-gnu.so",
                    "module_sha256": sha(os.path.join(ART, "build/_tensor_cuda.cpython-312-x86_64-linux-gnu.so")),
                    "note": "no C++/CUDA source is edited by MF1; same .so for before/A/A+B"},
  "reference": "scripts/apa_mf1_ref.py fp64 numpy from the definition (row i sees cols <= S-L+i); never tensor_gpu_v2 or code under test; no triu(k=1) on a rectangle",
  "G1": {
    "command": "pytest tensor_cuda/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py",
    "ignore_reason": "test_selector_accuracy.py is a CLI script (reads sys.argv[1] at import); collection error on the unchanged tree",
    "baseline": {"file": "artifacts/apa_mf1/g1/baseline_unchanged.xml", "sha256": sha(os.path.join(ART, "g1/baseline_unchanged.xml")),
                 "run": "unchanged tree, 2026-10-07 11:28-11:30 (BEFORE this registration; recorded as the lead-baseline input, not a gate verdict)",
                 "counts": {"passed": 719, "failed": 37, "error": 15, "skipped": 18}},
    "pass_rule": "post-change run has the identical outcome per test id over the baseline ids; new test_apa_mf1_* ids reported separately",
    "apa_cuda": "pytest apa_cuda/tests/test_parity.py before and after: same outcomes (extension apa_attention_cuda not built here)",
  },
  "G2": {
    "shapes_L_S": shapes,
    "quant_py": {"B": 1, "H": 2, "D": 32, "refine_percentile": 0.15, "cases": [["float32", 2], ["float32", 8], ["bfloat16", 8]], "seed_rule": "default_rng(1000*L+S)"},
    "tensor_gpu_v2": {"device": "cuda (cupy)", "dtype": "float32", "B": 1, "H": 2, "D": 16,
                       "paths": ["sdpa(use_flash=False)", "flash_attention(block_size=32)", "apa refine=1.0 (block_size=32)",
                                 "apa refine=0.15 bulk 2 non-tiled", "apa refine=0.15 bulk 2 use_tiled (L=512 shapes only, H chosen so B*H*L*S*4 > 256 MiB, D=8)"],
                       "checks": "forward and grads (q,k,v)"},
    "apa_cuda_reference": "test_parity.reference_forward/backward (causal) vs fp64 definition on the same grid (B=1,H=2,D=16, bulk 2)",
    "tolerance": {"float32": "max|err|/max|ref| <= 1e-5", "bfloat16": "max|err|/max|ref| <= 2e-2", "max_abs": "reported"},
    "ties": "selection entries with |absb-thr| <= 1e-5*|thr| (fp64 threshold) are fp ties: the code's choice is taken for the output reference; their count is reported; all non-tie entries must agree exactly",
    "square": "S == L: tree output (and tensor_gpu_v2 grads) byte-identical to the before copy (fix-A state; under A+B the quant.py square case changes by design and is receipted in the A-only set)",
    "before_receipt": "logs/apa_mf1_before.log must show the rectangular failures",
  },
  "G3": {
    "rule": "fp64 fused-kernel rule on the code's own |bulk| (captured fp32 absr): visible & |bulk| >= mean + z*std over visible keys, population std; z = quant._norm_ppf(1-0.15)",
    "metric": "agreement = mean over visible (row,col) of [code refine mask == rule mask]",
    "gate": ">= 0.999 per cell for A+B; before and A agreement reported",
    "cells": "G2 quant.py synthetic grid (all dtype cases) + every G4 real-data cell",
    "kernel_diag": "tc._C.bp_kernel_4_diagnostic variants 'a' (shipped apa_selective_fwd_train + a_mask) and 'h' (BK4) on real captures with one query head per KV head (kq shared): agreement of each arm's quant.py mask vs the kernel mask, reported (not gated)",
  },
  "G4_real": {
    "captures": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures/ (read-only)",
    "cells": {"Qwen3-1.7B": [0, 16, 27], "Qwen3-4B": [0, 21, 35]},
    "L": [64, 512, 2048], "S": 8192, "rows": "last L query rows against all S keys", "bulk_bits": [4, 8], "refine_percentile": 0.15,
    "arms": ["before (artifacts/apa_mf1/before quant.py)", "A (artifacts/apa_mf1/variants/quant_A.py snapshot)", "A+B (tree quant.py)"],
    "operands": "bf16 exact (raw bits); K/V expanded to query heads (GQA repeat, head h -> kv h//(Hq/KVH)); calls in groups of 8 query heads (rotation tables indexed within the group, identical across arms)",
    "metrics": ["eligible fraction of visible keys", "refined fraction of visible keys", "output rel err ||out-ref||_F/||ref||_F and max-abs vs fp64 exact attention", "G3 agreement"],
  },
  "G4_grm": {
    "model": "Qwen3-1.7B bf16, GRM core/qwen3_1p7b_tc.Qwen3_1p7b_TC.from_pretrained (bulk_bits 8 as GRM sets it), read-only GRM import, engine = this worktree (asserted)",
    "mode": "attention_mode 'apa', refine_percentile 0.15 (set_attention_mode), GRM default quant_kv_cache", "reference": "GRM 'standard' mode, same chunking",
    "text": "wikitext-2-raw-v1 test parquet (local HF cache), ''.join(text), tokenized without special tokens, first 8192+33 ids",
    "chunked": "prefill 16 chunks of 512 with KV cache, then 32 teacher-forced single-token steps (feed ids[8192+j])",
    "metrics": "per step KL(P_standard || P_arm) and top-1 agreement over the 32 teacher-forced step distributions (prefill-final reported separately)",
    "control": "unchunked prefill (S == L) of the first 4096 tokens (fallback 2048 on OOM; 8192 unchunked does not fit 12 GiB in the composed apa path) then 32 teacher-forced steps; A logits must equal before bytewise",
    "arms": ["standard", "before", "A", "A+B", "standard (repeat, determinism)"], "time_box_min": 45,
  },
  "mutation": {
    "kill": "a mutant is killed when the MF1 suite (tensor_cuda/tests/test_apa_mf1_quant.py + apa_cuda/tests/test_apa_mf1_tgv2.py + apa_cuda/tests/test_apa_mf1_parity_ref.py) has >= 1 failure; gate killed/(non-error) >= 0.80",
    "mutants": {
      "M1": "quant.py stats mask back to top-left np.triu(ones((L,S)),1)",
      "M2": "quant.py stats mask k = S - L (off by one)",
      "M3": "quant.py stats mask k = 1 + L - S (sign flip)",
      "M4": "tensor_gpu_v2 shared causal helper offset S - L - 1 (off by one)",
      "M5": "tensor_gpu_v2 shared causal helper offset L - S (sign flip)",
      "M6": "tensor_gpu_v2 sdpa site reverted to triu(k=1)",
      "M7": "tensor_gpu_v2 APA non-tiled backward site reverted to triu(k=1)",
      "M8": "tensor_gpu_v2 APA tiled forward site reverted (no S-L row offset)",
      "M9": "fix B count off by one (S-L+i)",
      "M10": "fix B count off by one (S-L+i+2)",
      "M11": "fix B sample variance (count-1)",
      "M12": "test_parity reference mask reverted to triu(k=1)",
      "M13": "tensor_gpu_v2 flash_attention backward site reverted",
    },
    "applied_to": "temporary copies swapped into the tree one at a time; originals restored and sha256-verified after every mutant (receipted)",
  },
  "predictions": {
    "P1_lead": "before A, rectangular quant.py excludes exactly S-L of each row's visible columns from eligibility; mean eligible fraction at L=512,S=8192 = 0.03198 (computed mean_i (i+1)/(S-L+i+1)); L=64: 0.00398; L=2048: 0.13701",
    "P2_lead": "real captures bulk 8: fix A changes output error a little; bulk 4: larger",
    "P3_lead": "GRM chunked prefill: KL vs standard drops after A (modest at bulk 8)",
    "P4_lead": "tensor_gpu_v2 score sites fail the rectangular reference grossly before the fix, rel err >> 1e-2",
    "P5_seat": "A+B G3 agreement >= 0.999 in every cell; before/A agreement < 0.999 in causal cells with S >> L",
    "P6_seat": "eligible fraction after A = 1.0 exactly in every cell",
    "P7_seat": "kernel diag mask vs A+B quant.py agreement >= 0.99 (residual = bf16 rounding of the cuBLAS bulk output vs the kernel's fp32 dot)",
    "P8_seat": "GRM unchunked control: A logits byte-identical to before",
  },
  "prior_art": "63960a4 (Project-Tensor 2026-06-09); FlashAttention-2 bottom-right causal (Dao 2023); GRAPA attention.py _visible_counts; kernels.cu bk4 forward count=S-L+qi+tid+1; M4a 2^E tag detector optional (not used unless stated)",
}
p = os.path.join(ART, "registration.json")
with open(p, "x") as f:
    json.dump(reg, f, indent=2); f.write("\n")
with open(p + ".sha256", "x") as f:
    f.write(f"{sha(p)}  registration.json\n")
os.chmod(p, 0o444); os.chmod(p + ".sha256", 0o444)
print(sha(p))
