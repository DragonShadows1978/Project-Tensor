# APA-MF1 — rectangular causal masks: finish the June 9 bottom-right fix (63960a4) everywhere it was missed

Source: System-Claude mail, 2026-10-07 10:55, 11:05 and 11:10. A review seat flagged it, System-Claude confirmed it,
and David pointed out the history. The lead re-verified it at production `fd32b6d`.

## The defect

63960a4 (2026-06-09) made `tensor_cuda/tensor_cuda/functional.py::_causal_mask` BOTTOM-RIGHT aligned
(`k = 1 + S − L`). It was the KV-Graft killer: a graft predicted " briefing" instead of recalling. The fix changed only
that function. Four inline top-left copies (`triu(..., k=1)` on an (L, S) rectangle) survive:

| Site | What it masks | Live? (lead's trace) |
|---|---|---|
| `tensor_cuda/tensor_cuda/quant.py:174` (`apa_quant_attention`) | the refine-STATISTICS mask; the score mask at :155 already uses the fixed `F._causal_mask` | **Yes.** GraftRepository `core/mistral7b_tc.py:485` (`attention_mode == "apa"`, the pure-APA path used by the Qwen3 family) calls it with `is_causal = (L > 1)`. It applies whenever S > L: chunked prefill (NC17 P3y built 32K with chunk 512), and grafts or injected prefixes. |
| `tensor_gpu_v2/_core.py:1660` | the standard-attention SCORE mask | legacy engine. Importers: `apa_cuda/apa_attention/{ops,quant_tables}.py`, `validate_turboquant.py`, `reproduce_bugs.py`, `adversarial_test_tensor_gpu_v2_phase5.py`, and `apa_cuda/tests/test_parity.py` |
| `tensor_gpu_v2/_core.py:2063` | the APA bulk SCORE mask | same |
| `tensor_gpu_v2/_core.py:2249` | the APA backward SCORE mask | same |

The test references at `apa_cuda/tests/test_parity.py:109` and `:153` are top-left as well. Parity against
`tensor_gpu_v2` can therefore pass while both sides are wrong.

**Effect when S > L.** Query row i is the absolute position S−L+i, so it should see columns 0..S−L+i. A top-left mask
shows it only 0..i:
- at the `tensor_gpu_v2` score sites, the attention itself is blind to most of the prefix;
- at `quant.py:174`, the scores are correct, but columns (i, S−L+i] are zeroed in `absr_f`, so they can never be
  refined and they skew the threshold.

When S == L the two alignments agree, which is why square tests never caught it.

**Already correct (lead's trace; verify, do not edit):**
- every CUDA kernel in `tensor_cuda/src/kernels.cu`, which uses `s_max = (S − L) + i + 1`;
- GRM's `_cublas_blend_attention` (bottom-right bounds);
- GRAPA's own composed path (`GRAPA-Native-LLM/grapa/attention.py:94`).

## The second divergence at the same site (separate fix B, semantic, David decides)

`quant.py:175–176` takes the mean and std over ALL S columns, with the masked entries set to 0 and counted. The fused
kernel takes them over the VISIBLE keys only: `kernels.cu:8113`, `count = causal ? S−L+qi+tid+1 : S`, a population mean.
GRAPA's corrected path does the same (`_visible_counts`, `attention.py:94–105`). The function's own comment requires
"Selection criterion must match the fused apa_selective_kernel". The research board lists this as known item 1:
"composed-APA causal stat dilution… patch before anything trains on the ENGINE's composed path".

Fix B changes the selection even when S == L. It is therefore a separate commit, and its promotion is David's call.
Fix A (alignment) restores the intended semantics and is the bug fix.

## Mechanism & scope

- Measures:
  - **A.** After the fix, every listed site produces bottom-right causal semantics for S ≥ L. Measured against an fp64
    numpy reference on rectangular and square shapes, forward and (`tensor_gpu_v2`) backward.
  - **B.** With fix B, `apa_quant_attention`'s refine selection agrees with the fused kernel's selection rule on the
    same operands.
  - **Both.** The before/after consequence on real attention data, and on a live GRM path.
- Mechanism doc: `tensor_cuda/tensor_cuda/functional.py` (the `_causal_mask` comment at lines 26–37: the convention
  and its failure mode). Also commit 63960a4, and `docs/APA_PAPER_DRAFT.md` §2.1 (refinement selection).
- Exercises:
  - `tensor_cuda/tensor_cuda/quant.py::apa_quant_attention` (causal, S > L and S == L);
  - `tensor_gpu_v2/_core.py` standard attention, APA forward and APA backward, causal; include every causal branch,
    for example `use_tiled`, and not only the three `triu` lines;
  - `apa_cuda/tests/test_parity.py` references;
  - real Q/K from `.../WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures/` (Qwen3-1.7B and Qwen3-4B,
    S=8192);
  - the GRM `"apa"` mode on Qwen3-1.7B with chunked prefill.
- Does not cover:
  - the CUDA kernels (already bottom-right; verify by test only);
  - the SP, packed and K1 entries;
  - GRAPA (already correct);
  - S < L (out of contract; it should raise or stay as is, so report it);
  - any promotion to production main (David's call, after the cross-model review).
- Required evidence:
  - failing-before / passing-after receipts for each site on rectangular cases (L ∈ {1, 7, 64, 512}, S ∈ {L, L+1,
    L+100, 4·L+3});
  - square-case identity before vs after for fix A (byte-identical outputs when S == L);
  - fix B's selection-agreement numbers against the fused-kernel rule;
  - the real-capture before/after table;
  - the GRM before/after numbers;
  - the mutation receipt.
- Operand census:
  - real captures: Qwen3-1.7B (28 layers × 16 query / 8 KV heads × D128) and Qwen3-4B (36 × 32/8 × D128), S=8192, bf16;
  - rectangular slices take the LAST L query rows against all S keys, L ∈ {64, 512, 2048};
  - synthetic shapes as listed under Required evidence;
  - the GRM model is Qwen3-1.7B bf16 from the HF cache, `bulk_bits` 8 as GRM sets it for the Qwen family.
- Baseline isolation:
  - before/after runs use the same operands and the same process type;
  - the "before" side is production `fd32b6d`'s file, imported from a read-only copy saved under
    `artifacts/apa_mf1/before/`;
  - references are fp64 numpy written from the mathematical definition (row i sees columns ≤ S−L+i), never from
    `tensor_gpu_v2` or from the code under test.
- Forbidden configuration:
  - a reference built with `triu(k=1)` on a rectangle;
  - square-only tests offered as evidence for the rectangular case;
  - comparing the fixed code against the old `tensor_gpu_v2` as the reference.

Seat: Claude Opus 5.5, high (an Agent-tool subagent; it has the GPU). A blind Codex cross-model review follows.

## Where to work

YOUR WRITABLE TARGET is the fork worktree `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf1`
(branch `apa-mf1`, forked from production `fd32b6d`). Edits, builds and GPU runs are AUTHORIZED there, and only in:
- `tensor_cuda/tensor_cuda/quant.py` (lines 170–177 region only);
- `tensor_gpu_v2/_core.py` (causal-mask constructions only);
- `apa_cuda/tests/test_parity.py` (reference masks only);
- new `tensor_cuda/tests/test_apa_mf1_*.py` and `apa_cuda/tests/test_apa_mf1_*.py`;
- `scripts/apa_mf1_*`, `artifacts/apa_mf1/`, `docs/APA_MF1_LEDGER.md`, `docs/APA_MF1_REPORT.md`, `logs/`.

Build the engine into `artifacts/apa_mf1/build/` with `tensor_cuda/build.sh` or an `scripts/apa_sp*_build.sh`
pattern (copy it as `scripts/apa_mf1_build.sh`). Use the existing `.so` only if its sha256 is recorded and it is
built from this tree's unchanged C++.

**READ-ONLY, FORBIDDEN TO WRITE:**
- the production repo `/mnt/ForgeRealm/Project-Tensor` (everything outside `Fork/`);
- `/mnt/ForgeRealm/GraftRepository` (import it read-only: put this worktree's engine first on `PYTHONPATH` and assert
  `tensor_cuda.__file__` is this worktree's);
- `/mnt/ForgeRealm/GRAPA-Native-LLM`;
- every other worktree;
- the HF cache.

Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`.

NO git (the lead commits). NO subagents. No background waits. NO DELETES: move anything unwanted with
`delete-me <path> --reason "..." --session apa-mf1`. Never kill a process you did not start.

## Required

1. **Audit.** Find every causal-mask construction in `tensor_cuda/tensor_cuda/*.py`, `tensor_gpu_v2/*.py`,
   `apa_cuda/**`, including its C++/CUDA sources, and the tests named above. Look beyond `triu(`:
   - `tril`;
   - index comparisons such as `j > i` or `j <= i`;
   - `arange` broadcasts;
   - "causal" branches in kernels.

   Classify each one: bottom-right / top-left / square-only / not a mask. Write it to
   `artifacts/apa_mf1/audit.md` with `path:line "quote"`. Report sites beyond the four above; fix them only if they
   are in the writable list, and otherwise list them.
2. **Tests first.** Write the rectangular tests against the fp64 reference, run them against the unchanged code, and
   save the failing receipt (`logs/apa_mf1_before.log`).
3. **Fix A.** Bottom-right alignment at every writable top-left site. Derive it from the same convention as
   `F._causal_mask` (`k = 1 + (S − L)`). In `quant.py`, prefer deriving the boolean from one shared helper rather
   than another inline copy, without changing `functional.py`'s public behavior. Re-run the tests and save the
   passing receipt. Square-case outputs must be byte-identical to before.
4. **Fix B** as a separately recorded change: visible-count population mean and std in `apa_quant_attention`,
   matching `kernels.cu:8113`. Use the GRAPA path as prior art, and do not import from GRAPA. Measure the selection
   agreement before and after against an fp64 numpy implementation of the fused-kernel rule (`|bulk| ≥ mean + z·std`
   over the visible keys, population std). Also against the fused kernel's own diagnostic mask where an entry exposes
   it; find out which entries do.
   - Keep A-only and A+B as two receipt sets.
   - Put the B diff in `artifacts/apa_mf1/fixB.patch` so the lead can split the commits.
5. **Real data.** On the captures (both models, 3 layers each: L0, a middle layer and the last), with rectangular
   slices L ∈ {64, 512, 2048} against S=8192 and `bulk_bits` ∈ {4, 8}, report before, A and A+B:
   - the fraction of visible keys that are eligible for refinement;
   - the refined fraction;
   - the output relative error against exact fp64 attention.
6. **Live GRM path.** Qwen3-1.7B through GRM's `attention_mode="apa"`, `refine_percentile` 0.15, bulk 8, chunked
   prefill (chunk 512) of one 8,192-token wikitext-2 test window, then 32 teacher-forced steps. Report next-token mean
   KL and top-1 against GRM's own standard mode on the same chunking, for before / A / A+B. Repeat with an unchunked
   (S == L) prefill as the control: A must equal before there.
   - If GRM cannot run against this worktree's engine without editing GRM, stop this item and report why.
   - Time box: 45 min.
7. **Mutation.** At least 5 registered mutants, killed ≥ 0.80:
   - quant.py back to top-left;
   - `k = S − L` (off by one);
   - `k = 1 + L − S` (sign flip);
   - one `tensor_gpu_v2` site reverted;
   - fix B's count off by one.

## Gates — registered BEFORE any GPU gate

Write `artifacts/apa_mf1/registration.json` and its `.sha256` create-only, with the shapes, the tolerances below, the
real-data cells, the predictions and the mutants.
- **G1:** a clean build; the existing `tensor_cuda/tests` suite (`pytest tensor_cuda/tests -q`, with any
  `--ignore`s stated) shows the same pass and fail set as production `fd32b6d` on this machine (lead baseline: run it
  on the unchanged tree first); the `apa_cuda` tests pass or fail as before, apart from the reference fix.
- **G2 (fix A):**
  - every rectangular case matches the fp64 reference within the dtype tolerance (fp32 ≤ 1e-5 relative; bf16/fp16
    ≤ 2e-2 relative, max-abs reported);
  - every square case is byte-identical to before;
  - the before-receipt shows the failures.
- **G3 (fix B):** selection agreement with the fused-kernel rule is ≥ 0.999 per cell (the rule is deterministic; the
  remainder is fp ties at the threshold), against the before-B agreement as reported.
- **G4:** the real-data table (item 5) and the GRM table (item 6) are complete, or stopped with the reason.
- **Mutation:** ≥ 0.80 killed.
- **Sanity/adjudication:** every anomalous number (agreement < 0.999, a square-case diff, a test changing state) is
  explained in one line.

Lead predictions:
- P1: before fix A, rectangular `apa_quant_attention` excludes exactly S−L of each row's visible columns from
  eligibility (columns i+1..S−L+i, leaving only the first i+1). The mean eligible fraction at L=512, S=8192 is
  therefore about 3.2% (mean over i of (i+1)/(S−L+i+1));
- P2: on real captures with bulk 8, fix A changes the output error by a little, because the excluded keys are scored
  at 8 bits; with bulk 4 the change is larger;
- P3: in the GRM chunked prefill, the KL against standard drops after A (`bulk_bits` 8, so the drop is modest);
- P4: the `tensor_gpu_v2` score sites fail the rectangular reference grossly before the fix (blind prefix), with
  relative error ≫ 1e-2.

## Rules

- HOUSE_RULES apply.
- GPU: `flock -w 900 /tmp/forge-gpu.lock timeout 600 …` per command, with the NVML exports
  (`LD_LIBRARY_PATH=/mnt/ForgeRealm/engines/nvidia-userspace-595.84/root/usr/lib/x86_64-linux-gnu`, and
  `/mnt/ForgeRealm/engines/nvidia-userspace-595.84/root/usr/bin` prepended to `PATH`).
- Registration before gates. RED honesty.
- Prior Art Directive:
  - 63960a4 (Project-Tensor, 2026-06-09);
  - FlashAttention-2's bottom-right causal convention for rectangular blocks (Dao, 2023);
  - GRAPA's visible-count stats (`attention.py`);
  - the fused kernel `kernels.cu:8113`;
  - System-Claude's M4a "2^E tag" leak detector (`/mnt/ForgeRealm/reference/openai-math-notes/receipts/tx2_opus/MUTATIONS.md`),
    optional as an extra check.
- Keep to about 2.5 hours.

## Done

Write `docs/APA_MF1_REPORT.md` and `artifacts/apa_mf1/verdict.json` first. Then give the final message:
- the audit summary, with every site and its classification;
- G1–G4 and mutation verdicts, with the registration SHA-256;
- the before / A / A+B tables (synthetic, real captures, GRM);
- predictions held or missed;
- the files changed, with the diff stat, and `fixB.patch`;
- deviations and open items;
- receipts cited as `path:line "quote"`.
