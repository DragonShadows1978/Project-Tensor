# P1-T promotion record: TensorCUDA DLPack interop (core) plus the `tc_signal` add-on, into Project-Tensor main

Promoted tip: fork `tensor_fork_voice` branch `voice-fix8` at e4cf1d6 (FIX8 plus lead send-back L2), plus this record.
The only difference from the reviewed tip is this record (docs and artifacts); no source or test changed.

## David's words (verbatim)

1. **Conditional approval, 2026-10-12 at about 21:55 EDT:** "I am going to bed, If the Blind Review passes, I approve the merger."
2. **Bar raised, 2026-10-13 evening:** "On THIS project which has bare access to metal, bugs can not be acceptable."
3. **Override, 2026-10-13 at about 19:05 EDT,** after round 9 rejected on 2 MINOR findings and the lead explained their
   impact: "I'm gonna command decision override I don't wanna burn any more tokens. You've seen where we're at. We've
   used like 20% of the week on this one freaking project actually it's probably more like 12% on this one project
   alone so I'm gonna authorize this ain't [expletive] worth the token."
4. **Confirmation, through the lead's readback question:**
   - Question: "Merge the verified FIX8 build (e4cf1d6) into Project-Tensor main now, with your override recorded and
     V9-01/V9-02 listed as known issues?" Answer: "Yes, merge it".
   - Engine install: "I meant to select yes to the second one", which answers "After the merge, should I also install
     the new engine .so into production?" with yes.

**Review status.** The review was WAIVED by David's override (quote 3). No round passed. The last review, round 9,
gave REJECT with 0 BLOCKING, 0 MAJOR and 2 MINOR.

## Review history (Codex gpt-6.1-sol, high; author Claude Opus 5.5)

| Round | Verdict | Findings | Fixed by |
|---|---|---|---|
| 1–3 | REJECT | F01–F07, N1–N5, V3-01, V3-02 | FIX1–FIX3 |
| 4 | REJECT | V4-01 MAJOR + 3 | FIX4 |
| 5 | REJECT | V5-01 MAJOR + 3 MINOR | FIX5 |
| 6 | REJECT (content filter stopped it early) | V6-01, V6-02 MAJOR, V6-03 MINOR | FIX6 |
| 7 | REJECT | V7-01 MAJOR, V7-02 MINOR | FIX7 + lead L1 |
| 8 | REJECT | V8-01, V8-02 MINOR | FIX8 + lead L2 |
| 9 | REJECT, on e4cf1d6 | V9-01, V9-02 MINOR (OPEN, below) | not fixed: waived by David's override |

Round 9's verdicts: R1 SAFE (no effect on existing core behaviour), R2 UNSOUND (V9-01, V9-02 only), R3 SOUND (the
add-on), R4 CORRECT (kernels and arithmetic), R5 PASS (package separation and build). Receipts: `docs/P1T_REV*_REPORT.md`,
`artifacts/p1t_rev*/` and `artifacts/p1t_rev9/lead/` on fork branch `voice-p1trev9` (fd5fcb6).

## Known open issues (KNOWN; not fixed)

- **V9-01 (MINOR), `tensor_cuda/src/bindings.cpp` around 465–489 (the export fence).** A failing fence call, followed by
  a failing cleanup call and then failure to allocate the error report, loses the first CUDA error. The later one
  stays in the slot. This affects diagnostics only, under compound CUDA failure plus host memory exhaustion. It causes
  no wrong data, no early release and no crash.
- **V9-02 (MINOR), `tensor_cuda/src/bindings.cpp` around 1632–1653 and 1715–1721, and `kernels.cu` around 411–438.**
  At interpreter exit, a CUDA failure that another native thread records between the core's two snapshot reads can be
  printed unnamed, when the exit hook cannot write to stderr. This affects diagnostics only.
- **Reproductions:** `artifacts/p1t_rev9/probes/core_fence_cpu.py` and `core_snapshot_cpu.py` on `voice-p1trev9`.
  The lead confirmed both: rc 1 on e4cf1d6.
- **Unfinished fix:** a FIX9 order and a partial edit exist on fork branch `voice-fix9`, uncommitted and stopped by
  David's override.

## Lead verification of the promoted build (`artifacts/p1t_fix8/lead/LEAD_VERIFICATION.md`)

- **Binaries:**
  - core `_tensor_cuda` `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24`;
  - `_tc_signal` `30aee4e67fbe80320863cf9d0d4b7748dca46c0db7a3689f70bcc9d9e8dc782b`.
- **Probes:** every probe from rounds 1–8 passes, or is INCONCLUSIVE for lack of a second GPU.
- **Regression lanes:** `gpu_regressions` fix8 39, fix7 115, fix56 157 and semantics 504, all with zero skips.
- **Voice runner:** PASS 249/249 (T1–T6, T1R, T4S). Voice confirmed it independently (their LD94).
- **Combined core plus `tc_signal` session:** the FAILED/ERROR set equals the 52 baseline ids, with 0 new.

## Scope

- **Covers:**
  - core `tensor_cuda/`: the P1-T DLPack surface (`__dlpack__`, `__dlpack_device__`, `from_dlpack`, `data_ptr`,
    `fill_`, `mark_modified`, `revision`, device-side `empty`), its release lifecycle and exit hook, and P1-T error
    recording;
  - the new package `tc_signal/` (1-D convolutions, ONNX-semantics LSTM, three normalizations, nearest upsample,
    reflection pad, round-half-to-even), which exchanges tensors with core only through DLPack;
  - the receipts (`docs/`, `artifacts/`, `logs/`, `orders/`).
- **Does not cover:**
  - pre-existing engine behaviour (round 9's R1 was SAFE);
  - performance;
  - real two-GPU execution;
  - V9-01 and V9-02 (above).
