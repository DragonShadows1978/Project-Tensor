# APA-DK-STE5 ledger

- Read local HOUSE_RULES mirror, AGENTS.md, immutable STE5/STE4 mechanism and REV1 findings before execution. No git, subagents or GPU.
- Created early FINAL_REPORT.md. No gate registered by seat; acceptance and signal thresholds come unchanged from orders/APA_DK_STE5.md.
- Prior art: Bengio, Leonard and Courville (2013) STE; Higham (2002) numerical error; NIST FIPS 180-4 (2015) hashing; existing STE1/STE4/REV1 harness/reference code. Unverified - lead to check. Harness composition, signal and byte verification are this round's work; no new kernel algorithm.

- CPU attempt 1 RED: 13 failed, 21 passed. Exact error: `ModuleNotFoundError: No module named 'numpy'` in `-I` subprocess before reaching tensor_cuda extension. Retained receipts cpu_attempt1.log/xml. No engine acceptance inferred.
- INPUT_PINS_AMENDMENT_001.json: pass existing NumPy package parent explicitly to isolated subprocesses; preserve isolation and all rejection assertions. Bind original plus amendment digest into captures/timing/verify metadata. Thresholds unchanged; no GPU data. Prior art: Python isolated interpreter / explicit import paths (Python 3.4, 2014), existing append-only STE4 provenance; unverified - lead to check. Harness-only repair, no kernel change.

- CPU attempt 2 PASS: 34 tests, required negatives and real F4 subprocesses; receipts/S5_CPU.json, cpu.xml/log. CUDA_VISIBLE_DEVICES empty, one-thread BLAS, TMPDIR under STE5 scratch. No GPU.
- CPU real GRAPA reference preflight PASS in 44.25 s, selective and refine-all, 134250496 visible pairs. Evidence is CPU arithmetic only.
- stage5 --check and bash -n PASS; verify5 without captures refuses with rc2, PASS false, missing base governor rows. S5-G2 pending lead, not ACCEPT.
- Manifest disclosure six sections and F4 package hash refreshed; full diff in MANIFEST.diff, exact disclosure in DISCLOSURE.json. BUNDLE actual .so af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760; manifest 5adcaa54901bb80f7a6b8fa24da511f176b481c3e8d12c7c2d4dd68e7c7488d4.
- FINAL_REPORT.md includes exact staged commands, F4 diff, complete disclosure, BUNDLE and exhaustive files/symlinks inventory. All scratch and initial RED receipts retained; no deletion/disposal, no DELETE_ME moves. F1 GRAPA-shape GPU correctness remains not claimed fixed; F5 is outside scope; F3 uses GRAPA own smoke.
