# APA-PF2-F3 implementation ledger

- Read the mirrored house rules, Project-Tensor AGENTS.md, immutable
  order, PF2-REV2 report and docs/APA.md §8. The test and both raw tc
  kernels exercise the specified mechanism. The reviewed F01/F02 logs
  show the old pytest failure and tracked-artifact import failure.
  REV2's norm witness establishes the -Inf coverage hole. Its F04
  source trace is reproduced by the four-load census.
- Registered G1–G6 before running a gate. Local F2 arm has a manifest
  but no module. The sibling F2 build receipt and module match the
  registered hashes. Read only, copied for the isolated F2 stage.
- Prior art. IEEE 754-2008 quiet NaN and Project-Tensor PF2-F1/F2 and
  PF2-REV2 (2026), reuse of the finite guard, bitwise comparison and
  negative code dot. The extension applies these to tc loads and covers
  both prefill guard branches against full and partial mutations.
  External wording unverified — lead to check "IEEE 754-2008 6.2 quiet
  NaN multiplication". Mutation analysis uses DeMillo et al. (1978).
- Build F3 PASS, artifacts/apa_pf2f3/build_F3/BUILD.json. Module 1036ec930a1e15d6c21213d977be22fd7c3846814d783f8fea40949ebb8f2347.
- Build pf3m01_prefill_norm_revert PASS, artifacts/apa_pf2f3/build_pf3m01_prefill_norm_revert/BUILD.json. Module 6d856546adf37b27f4ff82d457243f1f8ffc926aab9c33ec7f5944db8b5d9c1f.
- Build pf3m02_prefill_norm_pinf_only PASS, artifacts/apa_pf2f3/build_pf3m02_prefill_norm_pinf_only/BUILD.json. Module 3d550c8bd4c4f5d29fd5ef4db7f928df3ecd6f24ff31587b5be4ebb07212f850.
- Build pf3m03_tc_norm_revert PASS, artifacts/apa_pf2f3/build_pf3m03_tc_norm_revert/BUILD.json. Module 905d451556ae3b7e18bdc560a9cb8a9a369cfad8f9c13a0928503aa6e87dfbcb.
- Deviation. First archive provenance check assumed the archive HEAD was
  1cb328d22 and raised AssertionError for 66359102d5f01ac1f6ba853131a9aebabbe99722.
  The second archive checked that docs and all three original edit targets
  match 1cb328d22. Kept both archives. No gate criteria changed.
- G4 CPU passed. Archived candidate overlay, CUDA visibility unset,
  77 collected, 1 index pass, 76 CUDA-unavailable skips. CPU_ARCHIVE.json
  and CPU_PYTEST.{json,log,xml} preserve provenance and test results.
- F2, F3, and the three staged mutant packages imported under the CPU
  bounds. Each mutant rebuilt only kernels.cu.o and relinked.
- Script repair 1 of 3. The host guard extractor matched two x multiplies
  and raised "ValueError: too many values to unpack (expected 1)".
  Anchor the extraction directly after the kn[jk] read. Retain the
  partial host source in scratch and use a fresh directory for retry.
- G3 CPU passed the 360 extracted guard bit checks (six sites across four
  arms and 15 representative bit patterns). Covers both tc hook load
  indices. This is host classification evidence, with device kills
  pending the staged lines.
- G5 SASS passed all comparisons. F2 to F3 matched 803 Function blocks
  and changed exactly the ten registered tc blocks. Each prefill mutant
  changed exactly 32 prefill blocks; the tc mutant changed exactly ten.
  No symbols added or removed. See SASS_COMPARISON.json.
- The source diff is CHANGES.diff. NORM_FIXTURE.diff compares the copied
  non-finite harness against F2. Shipped tests affected are the host
  index test, all 16 sink cases and 44 partition cases (module selection),
  and all 16 norm cases (module selection plus -Inf and row checks).
- Staged 69 GPU launchers in LINES.json, with the full handoff in
  STAGED_LINES.md. All 69 dry invocations tested their SHA prechecks,
  returned 78 and reported would_start true and zero lease attempts.
  See DRY_RUNS.json and gpu/*_dry.{start.json,exit.json,log}. No real
  GPU line was invoked by this seat.
- G6 passed. The before/after audit reports exactly the three registered
  product changes, only authorized additions and zero missing paths.
  The dispatcher log is outside the seat's write inventory. The seat
  used read-only git archive for G4 and made zero GPU lease attempts.
  Registration SHA-256 still verifies. No paths deleted by the seat.
