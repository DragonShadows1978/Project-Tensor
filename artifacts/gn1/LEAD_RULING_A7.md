# Lead ruling on GN1 amendment A7: INCOMPLETE; the scorer's contract changes in A8

- **Order:** `orders/GN1_A7_MODE_EVIDENCE.md` (`00cb7a99`).
- **Run:** Codex seat, run `20261101T211135Z-3009553`, 1,423 s, `SHIM-DONE exit=1`.
- **How it ended:** the model provider's content filter stopped the seat while it was inspecting failed replays of
  the inherited edited-receipt controls (the REV4 control script). The seat wrote no final report. Its `REPORT.md`
  says IN PROGRESS. Its `LEDGER.md` and receipts are the record.

## What A7 established

- **D1:** A6's sources were copied into `a7/`, with only namespace, order and scratch relocation (`LEDGER.md`).
- **D2, device evidence:**
  - **Implementation:** `device_evidence.py`. The pinned engine has no Python device-query call, so the evidence
    comes from the CUDA runtime the engine's native module links: `cudaRuntimeGetVersion`, `cudaDriverGetVersion`,
    `cudaGetDevice` and `cudaGetDeviceProperties`. The header lines are cited in `DEVICE_API_SOURCES.json`, and the
    module's dependency symbols in `native_dependency_symbols.json`.
  - **Probe with the GPU hidden:** it records runtime 12060 and driver 13020, `cudaGetDevice` status 100 (no
    device), and no device name (`DEVICE_HIDDEN_PROBE.json`, `device_probe_gate.json`).
- **G2:** the CPU driver lines exit as expected (`G2_driver.json`, `G2_shim.json`).
- **G5:** the mutant re-verification passed (`G5_mutants.json`).
- **G4:** the control replay was in progress, with launcher fixes confined to scratch (`LEDGER.md`). It never
  completed.
- **D3 (modes may only demote):** whether it was finished is not recorded.
- **Unscored:** G1, G3, G6, and the predictions P121 and P122.

**Verdict:** INCOMPLETE. No gate verdict is drawn from this round.

## The contract change (lead decision, 2026-11-01)

**Why:**
- The GN1 loop has run REV1 to REV5 and amendments A1 to A7. Finding counts were 11, 11, 6, 4 and 5.
- A6's one open defect was an edit of the receipts' mode fields after a run.
- The edited-receipt controls that the threat model required are also what the content filter stopped in this round.

Under the project's review-loop cost guard, the lead narrows the contract, the same way as TX3-F2g's A9.

**The new contract, for A8 and for GN1-REV6:**
- **The scorer's job:** to give the right verdict on every root that an honest run of the staged script can produce.
  That includes:
  - wrong programs, PATH or `FORGE_GPU_*` environment;
  - a hidden or missing GPU;
  - interruptions, timeouts, and missing or truncated receipts left by an abnormal ending.
- **The bar:** it must never accept a root unless every staged line ran the sealed programs on the device and passed.
- **No longer the scorer's job:** detecting deliberate edits to well-formed receipts after the run.
  - **What protects that instead:** the lead's GPU wrapper writes a sha256 manifest of the run root as soon as the
    run ends, checks it, scores those bytes, and commits the manifest with the verdict.
- **Status of the open findings:** A6's mode-field promotion (`a6/D0_MODE_PROMOTION.json`) is closed by that manifest
  step. It is not waived.
- **What carries forward:**
  - D2's positive device evidence. It is a real run-condition check: a CPU-only run must never be accepted.
  - D3's demote-only modes.

## Prior art

No new algorithm. The device evidence uses the CUDA runtime's documented version and device-property queries (NVIDIA
CUDA Runtime API, 12.6). The post-run manifest is the ordinary sha256 manifest (GNU coreutils `sha256sum -c`), with
git history for tamper evidence.
