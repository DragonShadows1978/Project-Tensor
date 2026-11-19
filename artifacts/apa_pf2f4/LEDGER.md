# APA-PF2-F4 ledger

- Read the local House Rules mirror, Project-Tensor AGENTS.md, the on-disk order, docs/APA.md §8, dispatcher source and MAINCHK3 receipts. Both PF2 repeats recorded `AttributeError: 'FakeSpo2' object has no attribute 'packed_prefill'` in the two target tests. The lead’s interaction outcomes agree with the receipts.
- L01. Python 3.12.3 and pytest 9.0.2. Gate acceptance and stop conditions are recorded in the immutable REGISTRATION.md and its SHA-256 receipt.
- Prior art. Meszaros, xUnit Test Patterns (2007), recording test doubles, inherited from P1. DeMillo, Lipton and Sayward (1978), mutation testing, applied to the two ordered scratch failures. MAINCHK3 (Project-Tensor, 2026) supplies the suite, phase folding, per-ID comparison and Build F3 pins. PF2-F4 supplies the two route expectations and scratch mutation placements. No new product algorithm. Unverified — lead to check “Meszaros xUnit Test Patterns Test Spy 2007” and “DeMillo Lipton Sayward Hints on Test Data Selection 1978”.

## Deviations

- The initial registration command hit a shell quoting error before it wrote a valid registration. The partial draft is retained as REGISTRATION_DRAFT_FAILED.md. The complete registration was pinned before any gate. No gate script or threshold was repaired.

## Execution receipts

- Before edits, SCOPE_BEFORE.json recorded 32,540 paths. PREFILL_GREP.txt and CENSUS.json identified the two refusal tests. The live Python package and both test trees matched the archived PF2 input before edits.
- Only the two named test source files were patched. TESTS.diff and REPORT.md give exact source and assertion changes. The helper overlays those two files on the archived read-only PF2 tree and imports MAINCHK3's common and suite scripts by path. COPY_DEPS.json records the rebindings and pins.
- G1 passed. G3 then killed the restored refusal and the throwing fake in their call phases. G2 passed against both PF2 repeats. The 23 GiB limit was selected by the pre-registered headroom rail. G1, G2 and both G3 command receipts record the actual CPU process limits and logs.
- G4 passed. The before/after manifests, exact source-range comparison and read-only pin checks all agreed. FILES_WRITTEN.txt and SCRATCH_WRITTEN.txt list the outputs and retained scratch.
- Differential-testing prior art: McKeeman (1998), used through the MAINCHK3 per-ID comparison; unverified — lead to check “McKeeman differential testing 1998”. SHA-256 scope checks use the MAINCHK3 method and NIST FIPS 180-4 (2015).
- Decision: ACCEPT under the registered four-gate rule.

### Initial registration shell transcript

```text
24: line 17: warning: here-document at line 6 delimited by end-of-file (wanted `EOF')
/bin/bash: line 19: MAINCHK3: command not found
/bin/bash: line 20: EOF: command not found
/bin/bash: line 21: /REGISTRATION.sha256: Read-only file system
/bin/bash: line 22: /LEDGER.md: Read-only file system
/bin/bash: line 29: /PREFILL_GREP.txt: Read-only file system
/bin/bash: -c: line 30: unexpected EOF while looking for matching `''
```
