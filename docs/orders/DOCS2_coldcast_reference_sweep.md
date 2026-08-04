# DOCS2 — Remove ColdCast-as-Product References from Public Docs

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor` — edits to
`README.md`, `docs/APA.md`, `docs/SUPPORT_MATRIX.md`,
`docs/APA_VISUAL_TOKENS_PRIMER.md` AUTHORIZED. Documentation-only; no code,
no GPU runs, no git, no subagents, no network. RED honesty; no
monitor-idling.

## Directive (operator, 2026-08-04)

ColdCast is NOT publicly available. Public docs must not present
"ColdCast" as a named product/repo a reader could go find. The work it
names is publicly described ONLY as: **the Hunyuan3D pipeline with its
transformer (shape-generation runtime) replaced by a torch-free TensorCUDA
implementation.** The technical claims (engine port, APA E2E visual
receipt on 2.0, 2.0-vs-2.1 separation) are unchanged — only the product
name goes.

## Known hits (lead-surveyed; re-grep to catch any others in the four files)

- `README.md:6` — "the complete ColdCast Hunyuan3D image-to-3D pipeline" →
  e.g. "the complete Hunyuan3D image-to-3D pipeline (its transformer
  replaced by TensorCUDA)".
- `README.md:49` — "Complete ColdCast **Engine port**" → drop the name;
  the row already says what it is.
- `README.md:58` — the "Research systems built on it" row named "ColdCast"
  → rename the row to something like "Hunyuan3D transformer replacement
  (torch-free)"; description already correct.
- `docs/SUPPORT_MATRIX.md:40` — "complete ColdCast runtime report" → drop
  the name.
- `docs/APA_VISUAL_TOKENS_PRIMER.md:8` — "inside ColdCast" → "inside the
  TensorCUDA Hunyuan3D runtime replacement" or similar.
- `docs/APA_VISUAL_TOKENS_PRIMER.md:56` — heading "## 3. The modality jump
  (ColdCast, 2026-07-13/14)" → rename (e.g. "The modality jump —
  Hunyuan3D transformer replacement (2026-07-13/14)"). WARNING: this
  changes the heading anchor. `docs/SUPPORT_MATRIX.md` row 40 links to
  `#3-the-modality-jump-coldcast-2026-07-1314` four times — update EVERY
  anchor to the new heading's slug and verify each resolves.
- `docs/APA_VISUAL_TOKENS_PRIMER.md:148` — "Receipts: ColdCast LEDGER
  (EXP-APA-1/2, K1–K3)" → rephrase without the product name, e.g.
  "Receipts: EXP-APA-1/2 and K1–K3 in the runtime-replacement project's
  internal ledger (not publicly available)". Keep it honest that the
  receipt is not publicly readable.

## Rails

- QUOTES ARE IMMUTABLE: never edit text inside a blockquote or anything
  cited to a file/line — including David's verbatim formulation quote in
  the primer. If a quote contains "ColdCast", leave the quote intact and
  handle the naming in surrounding prose. Report any such case in ## Done.
- Do NOT touch `docs/APA_PAPER_DRAFT.md`, ledgers, plans, or `docs/orders/`.
- Do not weaken any claim; this is a naming sweep only.

## Done

Your final message MUST contain verbatim:
1. Final `grep -n ColdCast` output over the four files (expected: zero
   hits, or only quote-protected hits with explanation).
2. Every changed line, before → after.
3. Proof each SUPPORT_MATRIX anchor resolves against the primer's new
   heading slug.
4. Anything you could NOT fix, stated RED with the reason.
