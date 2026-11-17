G1 PASS
G2 PASS
G3 PASS
G4 PASS

ACCEPT under the registered ENG3-F5 gates. Test-only evidence; lead MAINCHK1 r2
is the ordered merged-main verification and was not run by this seat.

- **G1 — unit-test runs:** fresh-tree and worktree each 1 passed, no skips.
  Both compile their own shim into pytest tmp_path, preload that exact output,
  and report host_calls=0, control_calls=4. Fresh-tree Python/loader candidate
  paths contain no old artifacts/eng3/tools/cudart_count.so. Test operands were
  copied without compiled artifacts; no gate runner was copied. Evidence:
  G1.json, G1_fresh/RESULT.json, G1_worktree/RESULT.json, *_COMPILES.json,
  logs/eng3f5_G1_fresh.log and logs/eng3f5_G1_worktree.log.
- **G2 — suite run / per-ID equality:** both full test files run on CPU;
  313 IDs, 267 passed / 46 skipped, zero outcome or collection differences
  against F4 host/G1_all/RESULT.json. All 268 IDs from F4 host/G2_host/RESULT.json
  also match (267 passed / 1 skipped). F4's 45 GPU-only host rows receive the
  same runner marks; the inherited device-index row skips for unavailable CUDA.
  No test is omitted from the full selection. Only mechanism exception:
  test_g1_cpu_no_cuda_calls compiles shipped source instead of using an
  untracked binary. Evidence: G2.json (complete per-ID maps), G2_cpu/RESULT.json
  (phases and records), logs/eng3f5_G2_cpu.log. No logs/timings compared.
- **G3 — expected negative unit-test run:** scratch-only source mutant removes
  all 32 total/per-function increment pairs; forwarding and test bytes stay
  intact. The test compiles that source and loads its generated shim. Pytest
  exits 1; exactly the named test fails at `assert res["control_calls"] > 0`,
  with host_calls=0 and control_calls=0 (`assert 0 > 0`). Evidence: G3.json,
  G3_MUTANT.json, mutant.diff, G3_mutant/RESULT.json, G3_mutant_COMPILES.json,
  logs/eng3f5_G3_mutant.log. This failure is the registered power-gate success.
- **G4 — file-state/command audit:** no path outside the writable set changed;
  no removals, protected product/other-test/source changes or git-admin hash
  differences. Bytes outside SHIM and the named function are identical.
  No product rebuild, git commands/writes, deletes, moves, subagents or GPU.
  Evidence: SCOPE_BEFORE.json, SCOPE_AFTER.json, G4.json, source.diff, LEDGER.md.
  Audit covers worktree metadata, protected content and this seat's git-admin
  hashes; it is not a system-wide write audit.

Build H was checked before each run and the imported module path asserted:
`tensor_cuda/build-eng3f3/source_H/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`
SHA256 `ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43`.
All runs use CUDA_VISIBLE_DEVICES='', ENG3_EXPECT_CUDA=0, pytest cache disabled,
unique --basetemp and TMPDIR under /tmp/eng3f5_run. Module not rebuilt.

## Diff summary

- `tensor_cuda/tests/test_eng3_host.py`: only SHIM and
  test_g1_cpu_no_cuda_calls changed. SHIM names HERE/support/cudart_count.c;
  the function takes tmp_path, selects cc/gcc/clang, skips only if none exists,
  compiles with a 60-second timeout, fails with compiler stdout/stderr on
  compile error, then preloads its tmp .so. Child and counter assertions intact.
- `tensor_cuda/tests/support/cudart_count.c`: new 188-line byte copy of the
  tracked ENG3 source; original untouched. No product file changed.
- Two new F5 evidence helpers: scripts/eng3f5_run.py and scripts/eng3f5_scope.py.
  Registration, ledger, diff, receipts/report and logs are in the writable set.

No full shim compile command was found in ENG3's text receipts after targeted
cudart_count/shared/general-regs searches. The C source comment mentions
-mgeneral-regs-only, without a recorded command. Used the order's fallback:

`/usr/bin/cc -shared -fPIC -O2 -o <pytest tmp_path>/cudart_count.so <support>/cudart_count.c -ldl`

Regular shim builds all produced SHA256
`9aa551fddce52c6d4fa20949b1e435185de8beb874f5c7fa2dc67972474f6c93`;
the scratch mutant produced
`0f8f216ed39546731efcfce1aabf71938e843a2bc2c09746a6cfeca79ecd64f2`.
Compile argv/output paths and generated hashes are recorded, not inferred.

## Source SHA256s

- `artifacts/eng3/tools/cudart_count.c`:
  `095bab8f7e2b1907c437914fe4eb8ebfaf76bf49afd4372e084b488e603f5135`
- `tensor_cuda/tests/support/cudart_count.c`:
  `095bab8f7e2b1907c437914fe4eb8ebfaf76bf49afd4372e084b488e603f5135`

Byte equality checked, not just filename identity.

## Deviations

- Initial broad receipt reads/searches returned oversized, truncated output;
  repeated bounded, targeted reads. No gate/result depended on truncated text.
- An rg inventory included the not-yet-created support directory and exited 2;
  no gate ran and no file changed.
- Initial --version-only calls did not set TMPDIR. Repeated with scratch TMPDIR
  before registration; no temporary output was observed from the first calls.
- Own-script repairs: 0 of 3. No gate failures except the registered G3 negative.
- Existing dispatcher transcript logs/eng3f5_r1.log grew within the allowed set;
  it is listed separately from the four gate logs. No files moved or deleted.

## Prior art

Pytest tmp_path/hooks (Krekel et al., 2004+), Python subprocess (PSF, 2003+),
local ENG3 interposer and F4 per-ID harness (2026), glibc LD_PRELOAD/dlsym
interposition (year unverified), SHA-256 (NIST, 2001), and mutation testing
(DeMillo, Lipton and Sayward, 1978). Taken: fixture/process APIs, existing
interposition and assertions, per-ID equality and counter-silencing mutation.
Ours: F5's test-local compilation wiring and receipt assembly. No new numerical
algorithm. External citations unverified — lead to check: pytest tmp_path,
Python subprocess, glibc LD_PRELOAD, NIST SHA-256, DeMillo Lipton Sayward 1978.
Prior-art annotations are at the code site, in LEDGER.md, and here.

## Every file written

FILES_WRITTEN.txt enumerates every repository file created/modified (including
this report and both inventory files), every retained scratch regular file,
and pytest-created scratch symlinks. FILES_WRITTEN.json is its structured form.
No scratch was deleted. Repository paths are relative to this worktree;
scratch entries are absolute. Existing dispatcher transcript is explicitly
labeled. Two scratch test-tree copies contain the shipped Python/C operands;
the mutant differs only at the 32 registered increment pairs.

All retained scratch is under `/tmp/eng3f5_run/`: fresh/ and mutant/ test trees,
and four pytest basetemp directories containing their generated counter shims.
The complete exact file inventory follows (no directory-only omissions).

```text
artifacts/eng3f5/FILES_WRITTEN.json
artifacts/eng3f5/FILES_WRITTEN.txt
artifacts/eng3f5/G1.json
artifacts/eng3f5/G1_fresh/RESULT.json
artifacts/eng3f5/G1_fresh/junit.xml
artifacts/eng3f5/G1_fresh_COMMAND.json
artifacts/eng3f5/G1_fresh_COMPILES.json
artifacts/eng3f5/G1_worktree/RESULT.json
artifacts/eng3f5/G1_worktree/junit.xml
artifacts/eng3f5/G1_worktree_COMMAND.json
artifacts/eng3f5/G1_worktree_COMPILES.json
artifacts/eng3f5/G2.json
artifacts/eng3f5/G2_cpu/RESULT.json
artifacts/eng3f5/G2_cpu/junit.xml
artifacts/eng3f5/G2_cpu_COMMAND.json
artifacts/eng3f5/G2_cpu_COMPILES.json
artifacts/eng3f5/G3.json
artifacts/eng3f5/G3_MUTANT.json
artifacts/eng3f5/G3_mutant/RESULT.json
artifacts/eng3f5/G3_mutant/junit.xml
artifacts/eng3f5/G3_mutant_COMMAND.json
artifacts/eng3f5/G3_mutant_COMPILES.json
artifacts/eng3f5/G4.json
artifacts/eng3f5/L01.txt
artifacts/eng3f5/LEDGER.md
artifacts/eng3f5/MODULE.json
artifacts/eng3f5/REGISTRATION.md
artifacts/eng3f5/REGISTRATION.sha256
artifacts/eng3f5/REPORT.md
artifacts/eng3f5/SCOPE_AFTER.json
artifacts/eng3f5/SCOPE_BEFORE.json
artifacts/eng3f5/fresh_copies.json
artifacts/eng3f5/mutant.diff
artifacts/eng3f5/mutant_copies.json
artifacts/eng3f5/source.diff
logs/eng3f5_G1_fresh.log
logs/eng3f5_G1_worktree.log
logs/eng3f5_G2_cpu.log
logs/eng3f5_G3_mutant.log
logs/eng3f5_r1.log
scripts/eng3f5_run.py
scripts/eng3f5_scope.py
tensor_cuda/tests/support/cudart_count.c
tensor_cuda/tests/test_eng3_host.py

Scratch regular files (retained):
/tmp/eng3f5_run/fresh/tensor_cuda/tests/apa_sp1_1_reference.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/apa_sp1_reference.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/bench_paint_bake.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/bench_paint_inpaint.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/bench_paint_raster.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/bench_w8a16_k0b.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/bench_wo1a_sdpa.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/support/cudart_count.c
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_int4_sdpa_noncausal.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_ld1_pdwl.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_ld2_pdwl.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf1_quant.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_1_definition.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_1_gaps.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_1_sentinel.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_1_short.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_2_n1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_2_n2.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_2a_blocked.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_2b_blocked.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_2c_longkeys.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_2c_nits.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_mask.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_mf2_rule.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_om135_imm.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_phase6.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_pidv.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_selective.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_selective_int4.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_selective_splitk.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp1_1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp1_harness.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp1_host.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp1_reference.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp2.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp3.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp3_a4.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp3_a5.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp3_a6.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g_a2.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g_a3.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g_a4.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g_a5.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g_a6.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_sp4g_a7.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_spd1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_tiled.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apa_value_dim.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apamq_df1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_apamq_sb1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_argmax_last_axis.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_attention_phase5.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_checkpoint.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_conv_phase3b.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_dda_raycast.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_dlpack.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_eng1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_eng2.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_eng3_f3.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_eng3_host.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_engine.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_export_rows.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_ext_phase7.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_gated_delta_step.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_groupnorm_gn1.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_hy3d_engine_ops.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_int4_linear.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_int4_symmetric.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_integration.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_intn_linear.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_mxfp4_linear.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_nn_phase3.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_ops_phase2.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_paint_bake.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_paint_inpaint.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_paint_raster.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_phase9.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_qtile_attention.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_quantization_math.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_rnn_phase5b.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_rope_apply.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_selector_accuracy.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_splice_rows.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_terrain_render.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_w8a16_matmul.py
/tmp/eng3f5_run/fresh/tensor_cuda/tests/test_write_rows.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/apa_sp1_1_reference.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/apa_sp1_reference.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/bench_paint_bake.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/bench_paint_inpaint.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/bench_paint_raster.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/bench_w8a16_k0b.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/bench_wo1a_sdpa.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/support/cudart_count.c
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_int4_sdpa_noncausal.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_ld1_pdwl.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_ld2_pdwl.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf1_quant.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_1_definition.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_1_gaps.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_1_sentinel.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_1_short.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_2_n1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_2_n2.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_2a_blocked.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_2b_blocked.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_2c_longkeys.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_2c_nits.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_mask.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_mf2_rule.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_om135_imm.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_phase6.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_pidv.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_selective.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_selective_int4.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_selective_splitk.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp1_1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp1_harness.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp1_host.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp1_reference.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp2.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp3.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp3_a4.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp3_a5.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp3_a6.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g_a2.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g_a3.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g_a4.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g_a5.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g_a6.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_sp4g_a7.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_spd1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_tiled.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apa_value_dim.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apamq_df1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_apamq_sb1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_argmax_last_axis.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_attention_phase5.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_checkpoint.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_conv_phase3b.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_dda_raycast.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_dlpack.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_eng1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_eng2.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_eng3_f3.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_eng3_host.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_engine.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_export_rows.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_ext_phase7.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_gated_delta_step.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_groupnorm_gn1.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_hy3d_engine_ops.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_int4_linear.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_int4_symmetric.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_integration.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_intn_linear.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_mxfp4_linear.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_nn_phase3.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_ops_phase2.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_paint_bake.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_paint_inpaint.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_paint_raster.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_phase9.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_qtile_attention.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_quantization_math.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_rnn_phase5b.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_rope_apply.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_selector_accuracy.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_splice_rows.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_terrain_render.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_w8a16_matmul.py
/tmp/eng3f5_run/mutant/tensor_cuda/tests/test_write_rows.py
/tmp/eng3f5_run/pytest_G1_fresh/test_g1_cpu_no_cuda_calls0/cudart_count.so
/tmp/eng3f5_run/pytest_G1_worktree/test_g1_cpu_no_cuda_calls0/cudart_count.so
/tmp/eng3f5_run/pytest_G2_cpu/test_g1_cpu_no_cuda_calls0/cudart_count.so
/tmp/eng3f5_run/pytest_G3_mutant/test_g1_cpu_no_cuda_calls0/cudart_count.so

Scratch symlinks (pytest-created):
/tmp/eng3f5_run/pytest_G1_fresh/test_g1_cpu_no_cuda_callscurrent
/tmp/eng3f5_run/pytest_G1_worktree/test_g1_cpu_no_cuda_callscurrent
/tmp/eng3f5_run/pytest_G2_cpu/test_g1_cpu_no_cuda_callscurrent
/tmp/eng3f5_run/pytest_G3_mutant/test_g1_cpu_no_cuda_callscurrent
```
