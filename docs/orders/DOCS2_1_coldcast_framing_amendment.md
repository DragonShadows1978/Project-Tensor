# DOCS2.1 — ColdCast Framing Amendment (follow-up to DOCS2)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor` — edits to
`README.md`, `docs/APA.md`, `docs/SUPPORT_MATRIX.md`,
`docs/APA_VISUAL_TOKENS_PRIMER.md` AUTHORIZED. Documentation-only; no
code, no GPU runs, no git, no subagents, no network. RED honesty; no
monitor-idling.

## Directive update (operator, 2026-08-04, supersedes DOCS2's total ban)

ColdCast MAY be named in public docs, but only framed as WHAT it is:
**a personal project testing APA in Project-Tensor in a non-text-LLM
environment** — concretely, Hunyuan3D's shape transformer replaced with a
torch-free TensorCUDA implementation. Constraints that still hold:

- ColdCast is NOT publicly available — never present it as a repo,
  product, or download a reader could obtain; keep the "(not publicly
  available)" style labeling on receipt pointers.
- It is a validation/testing project, not a product line.
- All technical claims stay exactly as they are (2.0 APA E2E receipt,
  2.1 unconfirmed, etc.).

## Task

Reintroduce the ColdCast name where it aids the narrative — typically ONE
naming per document, at first mention, with the framing attached. Suggested
shape: "ColdCast, a personal (non-public) project that tests APA on
Project-Tensor outside text LLMs by replacing Hunyuan3D's shape
transformer with TensorCUDA". Natural sites:

- `README.md` — the identity paragraph mention of the Hunyuan3D pipeline,
  and/or the "Research systems built on it" row DOCS2 renamed.
- `docs/APA_VISUAL_TOKENS_PRIMER.md` — the §3 modality-jump narrative
  (first mention; the receipt pointer keeps its "internal ledger, not
  publicly available" phrasing but may name ColdCast as whose ledger it
  is).
- `docs/APA.md` / `docs/SUPPORT_MATRIX.md` — only if a mention genuinely
  helps; a matrix note naming ColdCast as the testing project is fine.

Do NOT rename the primer §3 heading again (anchors were just fixed —
leave the heading and all anchors untouched). Subsequent references within
a doc can stay generic ("the runtime replacement").

## Rails

- QUOTES ARE IMMUTABLE: no edits inside blockquotes or file:line-cited
  text.
- Do NOT touch `docs/APA_PAPER_DRAFT.md`, ledgers, plans, `docs/orders/`.
- No claim changes; framing only.

## Done

Your final message MUST contain verbatim:
1. Every changed line, before → after.
2. Final `grep -in coldcast` output over the four files, confirming every
   occurrence carries (or sits under) the personal-project framing and no
   occurrence implies public availability.
3. Confirmation the §3 heading/anchors are byte-unchanged.
4. Anything you could NOT fix, stated RED with the reason.
