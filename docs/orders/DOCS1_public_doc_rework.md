# DOCS1 — Project-Tensor Public Documentation Rework

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor` — edits under
`README.md` and `docs/` AUTHORIZED. This is a documentation-only order:
no code, no kernels, no build-system changes, no GPU runs, no model
workloads.

## Boundaries

- WRITE: `README.md`, `docs/*.md` (new files allowed in `docs/`).
- READ-ONLY: everything else in the repo (code, tests, artifacts, ledgers —
  read them for evidence, never modify).
- NO git — the lead commits. NO subagents. No network.
- RED honesty: if an acceptance check cannot be met, say so with receipts;
  a documented failure is a valid result.
- No monitor-idling: work to completion, then stop.

## Mission

Execute the Project-Tensor portion of
`docs/orders/REFERENCE_PUBLIC_DOC_RECOMMENDATIONS.md` (read it in full
first — it is the spec for this order; its "Specific edits →
Project-Tensor", "Recommended documentation architecture → Project-Tensor
README", "APA overview", "Required public status vocabulary", and
"Recommended master evidence matrix" sections are binding). Deliverables:

1. **README.md rework** — target 100–150 lines before links/license, in the
   reference doc's recommended order (identity paragraph → install + native
   smoke test → "What runs today" table → "Research systems built on it"
   table → 3–4 performance receipts → honest limitations → links). The
   suggested opening paragraph in the reference doc is pre-approved; use it.
   Include, near the top, the mandated sentence: "**The named models below
   are validation points, not a compatibility whitelist.** Support status is
   reported separately for TensorCUDA execution, APA evaluation, and GRM
   end-to-end certification."
2. **`docs/LEGACY_TENSOR_GPU_V2.md`** — move ALL legacy CuPy /
   `tensor_gpu_v2` installation and usage material out of the README into
   this clearly-labeled historical doc. The README's primary install path
   must build/import the current native `tensor_cuda` engine. Verify the
   documented install and import commands against the repo's ACTUAL build
   files (setup/pyproject/Makefile, test invocations) by inspection; if you
   can run the import smoke test CPU-side without network or GPU, do so and
   record the output; otherwise label it inspect-verified in ## Done.
3. **`docs/APA.md`** — the concise standalone APA overview, in the reference
   doc's 7-part order. The suggested opening blockquote is pre-approved.
   Say "attention families" (MHA/MQA/GQA/MLA), never "four architectures"
   next to a model list. Include positive AND boundary/negative results.
4. **`docs/SUPPORT_MATRIX.md`** — the master evidence matrix, seeded from
   the reference doc's starting structure and the status vocabulary table
   (reproduce the vocabulary table verbatim at the top). Every cell must
   either cite a receipt (link to the in-repo ledger/receipt doc — use the
   real filenames in `docs/`) or say `unconfirmed`. Do NOT invent receipts:
   if you cannot find an in-repo document supporting a cell, mark it
   `unconfirmed` even if the reference doc's starting table claimed more.
5. **`docs/GEMMA4_MQA_ADJUDICATION.md`** — the reconciliation note required
   by reference-doc §5, using the adjudicated facts below (these are the
   lead-supplied operative verdicts — do not re-litigate them, do not
   soften them). README, APA.md, and SUPPORT_MATRIX.md must all point at
   this note wherever Gemma/MQA is mentioned.
6. **Repo-local consistency sweep** (reference doc Phase 4): grep every
   public doc you touched (plus any remaining `docs/*.md` that a README
   link reaches) for `any transformer`, `architecture-agnostic`,
   `supported models`, `four architectures`, `unbounded`,
   `infinite context`, `no PyTorch`, `CuPy`, `MQA`, `Hunyuan`, `HY3D`,
   `DiT` — and fix each occurrence to the qualified vocabulary. List every
   hit and its disposition in ## Done. Historical ledgers/plans (immutable
   under house rules) are exempt — do not edit ledgers, plans, or the paper
   draft (`APA_PAPER_DRAFT.md` is OUT OF SCOPE for this order; the READMEs
   and new docs may link to it with correct framing).

## Gemma-4 / MQA adjudication (lead-supplied, binding)

- Model: google/gemma-4-12B-it. 48 layers = 40 sliding-window(1024) + 8
  global MQA layers; globals are 1 KV head × head_dim 512 with shared K=V
  projection.
- Engine port: COMPLETE and gated (parity vs QAT ground truth top-1 70/80
  with near-tie flips, bit-identical state gate, ~28.5 tok/s serving on the
  consumer card; exact QAT q4_0 import via symmetric-8 INT4 kernels;
  bf16-origin post-hoc INT4-g128 collapse was proven pure quantization
  noise — that is why only exact QAT import is supported).
- APA verdict: NEGATIVE, with law (mission closed 2026-07-04). The earlier
  paper-draft "near-tie" perplexity readings were a protocol artifact: that
  sweep's 2048-window scoring never actually engaged APA during scoring.
  The operative measurement (2026-07-04 re-probe, 12GB card, APA-engaged
  scoring, both modes OOM-free at the full 32K trained window): APA
  as-built on this model is net cost — +0.4–3.3% peak VRAM, ~3.5× prefill
  time, +8% decode ms/tok, perplexity +1.55–1.92%.
- The law: **APA is a multi-KV-head mechanism.** Its statistics need key
  population (sliding windows of ≤1024 keys fail that axis); its economics
  need KV-head multiplicity (a single shared KV head makes quantization
  noise coherent across all 16 query heads, head_dim 512 exceeds the fused
  kernel's supported head_dim, and MQA already pre-spent the KV slack at
  training time). Gemma-4 12B fails both axes — an accidental anti-APA
  architecture.
- Scope law: "saving memory on storage is not APA" — APA is precision in
  the attention computation; storage quantization is a separate residency
  decision. Do not present storage results as APA results.
- Selection rule: APA belongs on multi-KV-head (GQA-family),
  sufficiently-long-context sites. External boundary validation: a 26B
  kv=2 GQA model, collaborator-reported (label: External receipt) — APA
  clean, 64K context alive where the baseline attention OOMs.
- Resulting labels for the matrix: Gemma-4 12B = Engine port (parity-gated)
  + APA negative (adjudicated). GRM status: label strictly per receipts you
  find in-repo; a demonstrated mount is not lifecycle certification.

## Acceptance checks (from the reference doc — verify before ## Done)

- A new reader following only the README reaches the current native engine,
  and cannot mistake the headline rows for a model whitelist.
- Engine ports, research systems, and evaluation targets are separate
  tables.
- No unqualified "any transformer" / "architecture-agnostic" /
  "four architectures" survives in touched docs.
- Every SUPPORT_MATRIX cell has a receipt link or `unconfirmed`.
- Hunyuan3D appears by name; the 2.0 APA E2E receipt is never attributed
  to 2.1.

## Done

Your final message MUST contain, verbatim:

1. `wc -l` for README.md and every doc you created or modified.
2. The full list of files created/modified/moved.
3. The Phase-4 sweep table: every phrase hit, file:line, disposition.
4. For each SUPPORT_MATRIX row: the receipt file it cites, or
   `unconfirmed`.
5. Smoke-test status: executed (with output) or inspect-verified (with the
   build files you checked).
6. Any acceptance check you could NOT satisfy, stated RED with the reason.
