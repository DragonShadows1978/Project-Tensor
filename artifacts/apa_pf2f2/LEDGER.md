# APA-PF2-F2 implementation ledger

- Read the ForgeRealm house-rule mirror, Project-Tensor AGENTS, docs/APA.md §8
  and the PF2-F1 order. House Rules §3 requires a pre-build scope-mismatch stop
  when the ordered path misses the documented mechanism.
- Tool preflight only. `cmake --version` 3.28.3, `nvcc --version` 12.0.140
  (PATH) and 12.6.85 (F1 compiler), `pytest --version` 9.0.2,
  `python3 --version` 3.12.3. Free space at stop 110,122,455,040 bytes.
- Source inspection traced the public single-row packed path and REV1 norm
  witnesses through `r2` to `apa_spq1_core`. The latter routes G=4 and G=8
  to `spq1::bulk_shared` and other G values to `spq1::bulk`. The grouped
  kernel reads `scales[key]` through scalar `n` at `packed.cuh:149`.
  The ordered `norms[]` census cannot see that load. At either D, the grouped
  TQ4 score multiplies by this raw norm at line 178.
- Inspected the lead’s PF2-F1 GPU receipt. The 16 norm cases include eight
  G=4 cases. All 16 failed `raw_inf_decode_row_sets`; decode had zero NaN
  elements for raw +Inf and all NaNs for the NaN control. The lead exit was 1
  and the readback log SHA matched the exit receipt. Local `packed.cuh`,
  `launcher.cuh` and `packed_prefill.cuh` SHA-256 values matched F1’s build
  source map, and its build and staged module digests agreed.
- **RED: SCOPE MISMATCH.** The authorized load covers the non-grouped kernel
  only. Stopped before edits, registration, build, CPU tests or GPU staging.
  The proposed amendment would also permit a guard after the grouped load at
  line 149 and update SASS and mutant expectations for both consumers.
- Prior art for that proposed amendment is IEEE 754-2008 quiet-NaN propagation
  and Project-Tensor PF2-F1 (2026), which normalizes raw non-finite prefill
  norms before multiplying. Reuse PF2-F1’s finite check and quiet-NaN sentinel
  at decode’s two load sites. External clause lookup unverified — lead to
  check `IEEE 754-2008 6.2 quiet NaN multiplication`. No product code was
  written, so there is no new code-site annotation.
- Scope check by execution record. The only writes were the three Markdown
  files in `artifacts/apa_pf2f2/`. No git command, deletion, lease or GPU
  command was executed by this seat.

## r2, Amendment 1

- Source inspection before edits. DISPATCH_TRACE.json and OPERAND_CENSUS.txt
  account for all 13 census hits and both reachable loads. The inherited
  16-case failure and F1 module/receipt SHAs were verified again. On the raw
  path the bad score bypasses exact refinement when it is NaN, remains in
  softmax, and reaches the zero-only denominator guard in the merge.
- Registered G1–G5 in REGISTRATION.md and pinned the two orders and dispatch
  trace before gates. Version preflight returned CMake 3.28.3, nvcc 12.0.140
  on PATH, F1 nvcc 12.6.85, pytest 9.0.2 and Python 3.12.3.
- Prior art. IEEE 754-2008 quiet-NaN propagation and Project-Tensor PF2-F1
  (2026), reusing the prefill finite check and sentinel at both ordered decode
  loads. External citation unverified — lead to check IEEE 754-2008 6.2 quiet
  NaN multiplication. Test grids and isolated incremental reverts follow
  PF2-F1/P1-F3 (2026).
- Build F2 PASS, artifacts/apa_pf2f2/build_F2/BUILD.json. Module aefe2513a4d33d7310421f65e6619f0706de2889410836bda42193053f8ec97a.
- Build pf2m01_bulk_norm_finite_only PASS, artifacts/apa_pf2f2/build_pf2m01_bulk_norm_finite_only/BUILD.json. Module fd0c033ec1d924048d764bfb70c3a8087d78f62b7f433af76a4e3969830123a6.
- Build pf2m02_shared_norm_finite_only PASS, artifacts/apa_pf2f2/build_pf2m02_shared_norm_finite_only/BUILD.json. Module 8a2fe35fbafc1613770f3fa599486e1b60b4003a55147bb681b13021a2608e5b.
- L01 dry-run preflight returned 78 with would_start true. Hashed all 47
  local F1 build inputs against the lead's receipt and captured 19,448
  worktree file stats. The wrapper continues to append its own r2 log.
- Applied precisely the two load replacements in CHANGES.diff. No product
  file besides packed.cuh differs from the 47 pinned F1 build inputs.
- CPU Stage/import and HOST_GUARD.json. All four staged modules imported
  with the expected SHA. Across F2 and both reverts all 90 host bit-pattern
  checks agreed with each load expression, including signed zero, subnormal,
  ±Inf and NaN controls. This is host arithmetic evidence.
- G4 CPU SASS_COMPARISON.json. 813 paired sm_89 symbols. F1 to F2 changed
  12, all expected. F2 to the bulk revert changed only four bulk symbols.
  F2 to the grouped revert changed only eight bulk_shared symbols. No added
  or removed symbols in any comparison. Prior art: PF2-F1 normalized
  Function-block comparison and NVIDIA cuobjdump.
- PYTEST_CPU.log returned 1 passed, 76 skipped (CUDA hidden). G2 fixture
  readback matched 16/16 inherited lead G2_norm cases. All 62 final GPU
  commands dry-ran without taking a lease. DRY_RUNS.json records their
  exit 78, would_start true and log hashes.
- Repair 1 of 3 in AMENDMENT_EXECUTION_001.md. Corrected the G4 FLT_MAX r02
  TensorCUDA-to-NumPy adapter. Re-pinned and repeated all dry runs. The old
  pins and dry receipts were moved within artifacts and remain readable.
  An optional device-hidden TensorCUDA host-construction probe returned
  `RuntimeError: CUDA error at from_host: no CUDA-capable device is detected`.
  The corrected adapter was checked by source inspection. Its GPU gate
  remains staged. A separate initial torch metadata probe had a quoting
  SyntaxError. The rerun in FIXTURE_CENSUS.json records torch 2.11.0.
- G5 SCOPE.json. Zero missing files and zero writes outside the writable
  set. No git command, user-file deletion or admitted GPU run. The worktree
  lead log changed through the launcher, not through a seat file edit.
