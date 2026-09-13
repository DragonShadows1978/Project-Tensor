# BP-SCOUT-1 — how could backprop be made cheaper on this stack? (read-only scout, 2026-09-12)

Seat: Codex Astra (`gpt-6-astra`, reasoning high). **READ-ONLY. No code
changes, no file writes outside `/mnt/ForgeRealm/Project-Tensor/artifacts/bp_scout_1/`.**
Your WRITABLE TARGET is exactly that directory (create it). Read anything
under `/mnt/ForgeRealm/Project-Tensor`, `/mnt/ForgeRealm/GraftRepository`,
`/mnt/ForgeRealm/GRAPA-Native-LLM`, `/mnt/Shared/HOUSE_RULES.md`. No git,
no subagents, no GPU, foreground, every Bash call < 10 min, never kill or
signal any process (a GPU campaign and two other seats are running on this
machine). Prior Art Directive applies to every idea you propose (paper /
system / year; what is taken vs what would be ours; "unverified — lead to
check" where you cannot verify — no network here).

## David's question (verbatim)
"looking at the backprop and if we can optimize it so that it isn't as
resource intense as current backprop." Then: "Ideas on how to make the
Back Prop cheaper" — code RECOMMENDATIONS, not code.

## The machine and the receipts you must read first
- One RTX 4070 SUPER (12 GB); sustained dual-GPU is FORBIDDEN (near
  electrical fire, 2026-06-30; single card, short leased cells only).
- Engine: `tensor_cuda` (Project-Tensor; autograd loop exists, see
  `tensor.py`, `tensor_gpu_v2/`, `test_autograd_regressions_phase6.py`);
  APA = selective-precision attention (`docs/APA.md`, `docs/APA_SP1_LEDGER.md`,
  `apa_cuda/`): most attention mass sits on a small head set; the tail
  runs at reduced precision with no output change; SP1–SP5 ledgers hold the
  single-pass running-max criterion and split-K decode numbers.
- GRAPA training state: `GRAPA-Native-LLM` (232.6M MLA, paused at step
  1720; memory `project_grapa_training`: no training driver, no gradient
  checkpointing, and "APA costs MORE memory in TRAINING, not less (backward
  materializes …)" — read `GRAPA-Native-LLM/docs` and the training receipts
  under `checkpoints/`, `eval_runs/`, `logs/` for what was actually measured).
- Bolt-on experts (GraftRepository `docs/`, memory `project_moe_bolton`):
  rank-64 adapters trained by distillation from captured pairs — the base
  never runs backward.

## Mission (all read-only; output = one report)
1. **Cost model of backward on THIS stack**, from the code and receipts:
   per-layer FLOPs and activation memory for the 232.6M GRAPA model and
   for a 7B-class model at seq 2048 on 12 GB, split into attention
   (Q·Kᵀ, softmax, P·V and their gradients), MLP, norms, embeddings;
   where the peak lives (saved activations vs gradient buffers vs optimizer
   state); which of these `tensor_cuda` materializes today (cite lines).
   Every number labelled with its evidence class (code reading / receipt /
   reasoning). If a number is unmeasured, say so.
2. **Idea ledger — ranked**, each with: mechanism; what it saves (memory /
   FLOPs / wall) and at what cost (recompute, approximation, code); the
   prior art (paper/year; what we'd take vs what would be ours); the
   cheapest EXPERIMENT that would falsify it on this card (model, seq,
   metric, registered threshold, GPU-hours); code-site recommendations
   (file:line in tensor_cuda/apa_cuda where it would attach). Must
   cover at least: gradient checkpointing variants (selective /
   attention-only); activation compression / quantized activations;
   FlashAttention-style recomputed softmax in backward; **APA-keyed
   selective backward** (full precision on the attention head set, reduced
   or dropped on the tail — the lead's candidate; treat it skeptically:
   does the gradient through softmax tails actually vanish, or does the
   Jacobian of softmax couple them?); low-rank / sparse gradient methods
   (GaLore-class, sparse updates); forward-mode / zeroth-order gradients
   (MeZO-class) and their sample-efficiency cost; reversible layers;
   mixed-precision / 8-bit optimizer state; distillation-only training
   (the bolt-on precedent) as the "avoid backward through the base"
   route; anything else you find in the literature you know.
3. **The honest map**: which ideas attack FLOPs vs memory vs wall-clock;
   which are generic vs specific to APA/MLA/GRAPA; which are already
   standard (and so not a result) vs open; where the field's known
   failure modes are (approximate gradients diverging, checkpointing
   recompute wall, sparse gradients missing rare features).
4. **Recommended first experiment**, one paragraph: the single measurement
   that most reduces uncertainty, with its registered prediction and
   falsifier, ≤ 2 GPU-h, single card.

## Done (verbatim, in `artifacts/bp_scout_1/REPORT.md` and in your final message)
1. Cost model table (with evidence classes).
2. Ranked idea ledger (≥ 10 entries) with prior art and falsifying experiments.
3. The honest map + the recommended first experiment.
4. Process safety (read-only confirmed: `git status` clean apart from
   `artifacts/bp_scout_1/`), model id + effort.
