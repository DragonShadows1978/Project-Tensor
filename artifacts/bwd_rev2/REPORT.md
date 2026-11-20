Filed by the lead from the reviewer's final message (the seat's write was refused). Text verbatim.

# BWD-REV2: blind cross-model review of BWD-R2 (Claude Opus 5.5; the author was Codex)

**Verdict: no BLOCKER and no MAJOR findings; 3 MINOR and 8 NIT.**
- **The three VJPs are correct.** They match independent references at every boundary I tried: every scale from 1 to 7, every legal reflect pad for T from 1 to 9, and adversarial instance-norm rows.
- **The tracked path matches the design.** Selection is right in all 324 operand configurations I tried. The untracked path makes the same calls as production in all 228 untracked configurations.
- **The MINOR findings are about tests and docs, not the product:**
  - the finite-difference gate (G4) never runs candidate code;
  - the Kokoro and BP2 cells pass `None` affine where Voice passes identity tensors;
  - the public docs do not describe the tracked path.

**findings.json sha256:**
- **v1:** `e8c3827c688fd04d5e1057013b2b3bd2e6bf901b378b068be43b383be51eff3b`, sealed 2026-11-18T21:37:53-05:00. The mtime, sha and `date` line were written by one command into `findings_v1.seal`.
- **final:** `553098afa2d66b4c87636ab0a9a39f69ec1727ae63e4e5529ac2511be42010f4`, sealed 2026-11-18T21:39:09-05:00 in `findings_final.seal`.

**How I ran things:**
- Every command ran with `CUDA_VISIBLE_DEVICES` empty, under `nice -n 19` and a `timeout`, single-threaded, as `python3 -B -P`. Versions: Python 3.12.3, git 2.43.0 (read-only use).
- I used scratch copies of BWD-R2's pinned packages in `/tmp/bwd_rev2_pkgs/{candidate,prod}`. The core .so is `4a092149…`, the same as r1, and the tc_signal .so is `9e4dc289…`. The candidate `.py` files are byte-identical to `facb2169` (checked with `cmp`).
- Core TensorCUDA ops are CUDA-only on this host: `slice` raised `CUDA error at slice: no CUDA-capable device is detected`. So my checks run the real candidate `.py` source against a NumPy fp32 stand-in for the core (`checks/bwd_rev2_fakecore.py`). It copies TensorCUDA's reduction orders as I read them in `kernels.cu`/`ops.cpp`: serial `reduce_kernel`, 256-thread `reduce_lastdim_kernel` plus a tree, and `mean = sum*fp32(1/n)`. Index and algebra results from it are exact. Precision numbers from it are an emulation, not the GPU.

## Findings (verbatim from findings.json, final)

**BWDR2R-F01 (MINOR)**: G4 (finite differences) never runs candidate code. Site: `tc_signal/tests/test_bwd_r2_fd.py` 60 (refs = oracle_call(...)); 81, 90.
Argument: check_cell compares Voice's fp64 oracle VJP (bwd_r2_support/oracle.py) with central differences of the same oracle's fp64 forward. No tc_signal or tensor_cuda call is made. The reported 'FD: 120 probes, worst 0.0017 of the bound' (reviewer recount: tiny 46 coordinates + Kokoro 74 projections = 120; worst 0.00169 at Kokoro reflection) validates the answer key, not the candidate VJPs. Candidate correctness rests on G3 alone (candidate vs the FD-validated key), which is sound transitively, but the gate cannot fail for a wrong candidate (L16: the cheapest pass). Evidence: plain run with fixtures (checks/plain_run_with_fixtures.log) passes these 8 tests on a host with no CUDA device and CUDA_VISIBLE_DEVICES empty.
Fix: Label G4 as answer-key validation in the report and acceptance text. If candidate-level FD is wanted, add a GPU line that projects the candidate gradient on the seeded directions and compares with the fp64 FD of the oracle forward.
Post-v1: Unchanged after reading the receipts. BWD-R2's REGISTRATION says G4 'uses verbatim Voice fp64 operators', and its LEDGER calls G4 'the registered fallback smoke'. So the seat labelled it honestly. REPORT.md's gate table still lists 'G4 PASS ... 120 probes', and the review order lists it among the gates that passed, which is the misreading this finding is about.

**BWDR2R-F02 (MINOR)**: Kokoro and BP2 cells do not use the production caller's operand form for instance norm and AdaIN. Site: `tc_signal/tests/bwd_r2_support/cases.py` 100-105 (gpu_args omit_affine).
Argument: gpu_args drops scale and bias when they are identity, so the Kokoro cell and all 12 BP2 cells call instance_norm1d(x, None, None). Voice's caller passes identity tensors: /mnt/ForgeRealm/Project-Voice/engine/kokoro_tc/model.py:751 `self.ts.instance_norm1d(x, self.f32(p['scale']), self.f32(p['bias']), eps=p['eps'])`. On the tracked path that is the 3-slot hook (x, scale, bias) and `h = dy * scale.reshape(...)`. That branch is exercised only at the BP1 small cells (at most [2, 8, 25]). With identity scale the product is exact (x1.0), so no numeric defect is predicted, but no Kokoro-geometry receipt runs the caller's form (house lesson: operand parity with the production caller). The round order's premise 'Kokoro uses scale=None and bias=None' contradicts model.py:751.
Fix: Run the Kokoro and BP2 instance-norm and AdaIN cells with the identity tensors passed (as model.py:751 does), in addition to or instead of the None spelling.
Post-v1: Unchanged. BWD-R2's LEDGER: 'BP1 Kokoro and BP2 use identity affine'. The seat spelled them None. The PROD forward guard (scripts/bwd_r2_gpu.py forward_guard, gpu_real_g2-forward_worker.json) uses the same gpu_args, so the None spelling is also what the Kokoro forward-byte receipt covers.

**BWDR2R-F03 (NIT; MINOR at v1)**: Gate tolerances load from a receipts folder without checking their seal. Site: `tc_signal/tests/bwd_r2_support/cases.py` 22-26 (TOLERANCES, tolerances()).
Argument: tolerances() reads artifacts/bwd_r2/TOLERANCES_R2.json through a repo-relative path and never compares it with the sealed sha256 (0e99da33...e2 in TOLERANCES_R2.sha256). A later edit to that JSON silently loosens or tightens every G3/G4 bound, and the test package depends on a receipts directory outside tc_signal/. The values themselves match DESIGN section 6 (nearest/reflection 2e-7/2e-6; instance norm 5e-5/3e-4; matmul max(3e-6, 1.5e-5*S)/1e-4; broadcast max(2e-6, 8e-6*S)/5e-5; FD 1e-7 + 1e-5*|v|).
Fix: Assert the sealed sha256 in tolerances(), or move the constants with their seal into tests/bwd_r2_support/.
Post-v1: Downgraded from MINOR to NIT. The receipts show the seal predates every comparison: TOLERANCES_R2.json and .sha256 mtime 20:16:37.89; the first comparison, logs/bwd_r2_g4_fd.log, ends 20:26:28; GPU lines ran 21:20-21:21. The GPU launcher pins the file's sha256 (GPU_SOURCES.json lists 0e99da33...; scripts/bwd_r2_gpu.py:25 checks the TOLERANCES_R2 seal). The residual is plain pytest runs, which still load the JSON unchecked.

**BWDR2R-F04 (MINOR)**: The public docstrings and README do not describe the new tracked-path behaviour. Site: `tc_signal/tc_signal/_small.py; tc_signal/tc_signal/_norms.py; tc_signal/README.md` _small.py:3, 37, 82; _norms.py:50-52.
Argument: When a TensorCUDA operand requires grad and grad mode is on, the three ops now: return a new TensorCUDA tensor that does not come from the module allocator; raise ValueError for out=; raise TypeError for any non-TensorCUDA operand; and (for instance norm) raise ValueError for a trainable scale or bias. The docstrings still say 'Returns `out`' and 'the output comes from the module allocator unless `out` is given'. The README has no gradient section. Orchestrator rule 5: docs follow the code.
Fix: Add a short 'Gradients (BWD-R2)' paragraph to each docstring and to the README: the selection rule, the refusals, and that the tracked output bypasses set_allocator.

**BWDR2R-F05 (NIT)**: Nothing checks that the instance-norm VJP reuses the forward's statistics scheme. Site: `tc_signal/tc_signal/_norms.py` 70.
Argument: The VJP replays the forward's scheme: shift by x[...,0], mean of the shifted row, centred biased variance, then rstd = 1/sqrt(var+eps). It does so with different reduction trees than src/norms.cu (core reduce_lastdim and mul by fp32(1/T), not block_tree_sum / Lf and fmaf). So it is a replay, not a bitwise reuse, which DESIGN section 5 allows. Reviewer mutant B7 (shift by the last sample instead of the first) survives every ported BWD-R2 test (checks/boundary_mutants.json). The variants are numerically equivalent in the tested regimes, so the 'reuse' property is unenforced, not violated.
Fix: Accept and record it as unenforced, or share the forward's per-row (mean, rstd) from the kernel if exact reuse is ever required.

**BWDR2R-F06 (NIT)**: GPU tests skip only when CUDA_VISIBLE_DEVICES is exactly the empty string. Site: `tc_signal/tests/bwd_r2_support/cases.py` 29-31 (require_gpu).
Argument: On a host without a usable GPU, these tests error instead of skipping when the variable is unset or set to -1. The fixture-absent skip, which BWD-R1 required, is correct.
Fix: Also skip when no CUDA device is usable (a device-count probe).

**BWDR2R-F07 (NIT)**: Version skew with a core that lacks attach_vjp fails after the forward has run. Site: `tc_signal/tc_signal/__init__.py` 59-73 (_recording).
Argument: Production tensor_cuda (3138cf54 package) has Tensor.requires_grad, is_grad_enabled and mark_modified, but no attach_vjp. Pairing the new tc_signal with that core turns a previously silent detached result into an AttributeError, raised only after the forward kernel has run, for any caller that passes a requires_grad tensor in grad mode. Both ship from one repository, so this is low risk.
Fix: In _recording, check hasattr(tc, 'attach_vjp') and raise a clear error before any work, or document the minimum core.

**BWDR2R-F08 (NIT)**: The instance-norm VJP keeps every intermediate alive until it returns. Site: `tc_signal/tc_signal/_norms.py` 64-76 (vjp).
Argument: The locals shifted, centered, z and h, plus the expression temporaries, keep about 7 to 8 full-size fp32 tensors alive at peak. That is about 0.2 GB at the generator's [1,128,48000] AdaIN rows (DESIGN section 5 memory model: 24.6 MB per tensor). This is inside the design's 0.4 to 0.7 GB backward-temp allowance, but it adds to it.
Fix: Release shifted after centered, and so on (del), or fuse in a later round.

**BWDR2R-F09 (NIT)**: The instance-norm closure captures graph-bearing x and scale, not detached copies. Site: `tc_signal/tc_signal/_norms.py` 64-76.
Argument: DESIGN section 3 says 'A closure keeps only detached saved tensors or small immutable metadata'. The closure keeps x (which may carry grad history) and scale. This is harmless here: the reviewer's closure census shows every captured tensor is a listed hook input, so it is revision-checked; the attached node already owns those Variables; there is no cycle; and the callback runs with grad off. It still departs from the design text.
Fix: Amend the design text to 'listed inputs or detached saved tensors', or capture detached aliases that share revision.

**BWDR2R-F10 (NIT)**: Ops without a VJP still silently drop the gradient for operands that require grad. Site: `tc_signal/tc_signal/__init__.py; _norms.py; _lstm.py; _small.py` conv1d, conv_transpose1d, layer_norm, group_norm, lstm, rint.
Argument: This behaviour predates BWD-R2 and is outside its scope (the convolutions are round 3). A training caller that routes through one of these ops gets no gradient upstream and no error. A production training path should refuse instead.
Fix: Register a successor: refuse requires_grad operands under grad mode in ops without a VJP, or document it, before round 4 integration.

**BWDR2R-F11 (NIT)**: The AdaIN dB check uses the matmul class, which is looser than the design's broadcast class. Site: `tc_signal/tests/test_bwd_r2_numeric.py` kind selection in test_gpu_voice_vjps.
Argument: dB is an add-unbroadcast sum. DESIGN section 6 places that kind of sum in the 'broadcast' class (max(2e-6, 8e-6*S)/5e-5). The round order put 'the AdaIN Gemm parts under the matmul class', so the test follows the order. For N=1 (every cell except small2), dB = dh with no reduction, so the difference matters only at small2.
Fix: None required. Optionally use the broadcast class for dB.

## Answers

### 1. VJP correctness

**Read:** the diff, the forward index maps in `src/small_ops.cu:55-81`, `src/norms.cu:93-143`, core `ops.cpp` (slice, sum, mean, reshape, flip, cat) and `kernels.cu` (`reduce_impl`, `reduce_to`, `slice_nd`, `cat_nd`).
**Ran:** `checks/bwd_rev2_vjp_check.py`; results in `checks/vjp_candidate.json`.

**Nearest upsample: correct for every integer scale.**
- `reshape([count,s,1]).sum([1])` lands on the serial `reduce_kernel`, so repeats are summed in ascending order.
- Scale 1 returns `dy` itself. That alias is safe because `reduce_to` clones it on accumulation (`kernels.cu:1364-1365`).
- **Result:** bitwise equal to an independent fp32 ascending sum in 63/63 cells (ranks 1 to 3, T in {1,2,5}, s = 1 to 7), and exact on integer seeds.

**Reflection: correct for every legal pad.**
- **Result:** bitwise equal in 285/285 cells to a reference written from DESIGN §5's direct, left, right formulas, in that order.
- **Coverage:** every (pb, pe) in [0, T−1]² for T = 1 to 9, so pb = 0, pe = 0 and pads of T−1 are all included.
- **Terms:** at most 3 per input (measured maximum 3). Exact against `np.add.at` on integer seeds.
- **Slices:** all are in bounds. My stand-in slice raises on out-of-range, and core `slice_nd` itself has no bounds check.

**Instance norm: correct, including the covariance term.**
- **Formula:** `dx = rstd·(h − mean h − z·mean(hz))` is the exact derivative for biased variance with eps.
- **Scheme:** it re-runs the forward's statistics recipe (shift by x[…,0], shifted mean, centred biased variance, `1/sqrt(var+eps)`) with core reduction trees. That is a replay, not a bitwise reuse (F05).
- **13 adversarial cells, all within the class bound 5e-5/3e-4; worst ratio 0.0158** (eps 1e-12, tiny variance):
  - T = 1 gives dx exactly 0;
  - T = 2;
  - constant rows;
  - large offset, and a first-sample outlier;
  - tiny and huge (1e15) variance;
  - eps = 0 on non-constant rows;
  - T = 25,440.
- **Singular case:** eps = 0 on a constant row gives all NaN, which is the design's classification.
- **Voice fixtures (emulated core):**
  - BP1, 4/4 pass: worst 0.0030 against the replay and 0.0028 against Voice's key.
  - BP2, 6/6 pass: worst 0.0192 against the replay and 0.0193 against the key, both at c514 (the near-constant row).

**Affine and eps: handled.**
- Frozen scale and bias are hook slots, with `h = dy·scale`. Bias is listed (so it is revision-checked) and unused, which is correct.
- `eps` is captured after validation.
- A trainable affine is refused before any work (10 configurations in the path check).

### 2. The tracked path and the hook contract

**Ran:** `checks/bwd_rev2_paths_check.py` on the candidate and on prod (prod as a control), then `bwd_rev2_paths_compare.py`; outputs in `checks/paths_*.json` and `paths_compare.log`.

**Selection matches the design in 324/324 configurations, 0 mismatches.**
- **Matrix:** grad mode on/off; x as a requires-grad, frozen or foreign tensor; scale and bias as None, frozen, trainable or foreign; out as None, TensorCUDA or foreign.
- **Outcomes:** 6 tracked, 64 `out=` refusals, 16 foreign refusals, 10 trainable-affine refusals, 228 untracked.
- **Control:** prod shows the expected mismatches, which shows the harness can tell the paths apart.

**Refusals fire only on the tracked path.**
- The foreign refusal covers any non-TensorCUDA operand, including frozen ones. That is broader than the order's wording ("active operand"), but it is required: hook slots must be TensorCUDA tensors (`bindings.cpp:2667`).
- No producer's `__dlpack__` was ever called (export count 0).

**The hook contract holds.**
- **Order:** in every tracked case the events run plan → `empty` → write → `mark_modified(value)` → `attach_vjp(value, …)`.
- **No cycle:** no closure holds the result or `value`. The captured variables are {count, s, shape}, {length, pb, pe, tc} and {e, scale, x}.
- **Revision checks:** every captured tensor is a listed hook input, so core revision-checks it before and after the callback (`autograd.cpp check_revisions`).

**Forward bytes are unchanged.**
- 228/228 untracked configurations make the same call sequence as production 3138cf54, including `out=` and foreign DLPack producers.
- `git diff d4285023 facb2169 -- tensor_cuda tc_signal/src tc_signal/include tc_signal/CMakeLists.txt` is empty, and tc_signal is unchanged between 3138cf54 and d4285023.
- After v1, the PROD forward guard (`gpu_real_g2-forward_worker.json`) showed tracked, untracked and no-grad output byte-identical to PROD at the four Kokoro geometries.

### 3. Determinism and precision

**Fixed order, no atomics.**
- The reductions used are `reduce_lastdim_kernel` (a strided loop over 256 threads, then a halving tree) and the serial `reduce_kernel`. Everything else is copies and elementwise ops.
- Gradients accumulate through `reduce_to` plus `ew_binary`.
- The only atomic I saw in core is `embed_bwd_kernel`, which this round does not use.
- After v1, G5's receipt shows equal digests across 10 reruns and 3 processes.

**Tolerances are DESIGN §6's classes.**
- `row_sum` is an extra check the design does not require.
- The seal (mtime 20:16:37) precedes the first comparison (20:26:28).
- The GPU launcher pins the tolerance file's sha; a plain pytest run does not (F03).

### 4. Tests and gates

**Reach census (68 test IDs, recounted):**

| Test | Count | What it reaches |
|---|---|---|
| `test_host_tiny_fd`, `test_host_kokoro_fd` | 8 | Voice's oracle only (F01) |
| `test_gpu_voice_vjps` | 28 | All three VJPs, the hook, and core matmul/slice/broadcast for AdaIN |
| `exact_integer`, `nearest_ascending_order`, `reflection_direct_left_right_order` | 2 + 1 + 1 | s = 2, 4 and 1; reflect (3,3) at T = 4; (0,0); order witnesses |
| `norm_constant_and_large_offset`, `norm_singleton_and_singularity` | 1 + 2 | Constant, large-offset, T = 1, eps = 0 |
| `host_out_refusal`, `host_cpu_planner_boundary`, `host_foreign_operand_refusal`, `host_trainable_affine_refusal` | 3 + 3 + 3 + 2 | `_recording` and the native planner; the `out=` cases also pin the grad-mode and requires-grad checks |
| `gpu_paths_and_interop`, `gpu_saved_input_revision`, saved affine/output revision, `gpu_foreign_operand_refusal`, `gpu_forward_special_bit_patterns` | 3 + 3 + 1 + 1 + 2 | Tracked path, revisions, DLPack import |
| `test_gpu_grad_bytes` | 4 | Determinism |

No test reaches a VJP closure on a host, because core ops need CUDA.

**Would a boundary-only error pass? No.** `checks/bwd_rev2_boundary_mutants.py` ported the GPU tests, with their registered criteria, to the NumPy stand-in and applied 8 reviewer mutants.
- **Killed (7):**
  - right reflection skipped at pe = 1;
  - left reflection skipped at pb = 1;
  - right reflection not flipped;
  - wrong scale-1 shortcut;
  - left and right reflections added in swapped order;
  - last upsample repeat dropped;
  - eps dropped from the VJP.
- **Survived (1):** B7, the numerically equivalent shift choice (F05).

**Mutants (read after v1):** r2m01 to r2m05 are all killed, each with `MUTATED LINE EXECUTED` and an fp64-replay or refusal assertion.
- r2m05 detaches γ in the test composition. That is valid as a test-sensitivity check, since the product has no AdaIN code of its own.
- The riskiest edits not in the registered set (right reflection, add order, scale 1, eps) are covered by existing tests, per my mutants above.

**Plain run: yes.** Fresh tree, candidate packages only on `PYTHONPATH`:
- **No fixture roots:** 15 passed, 53 skipped. 36 skips give the fixture-root reason and 17 give "CUDA disabled".
- **Fixture roots set:** 19 passed, 49 skipped.
- **Scope grep** for `voice_wt|WorkTree/|spec_from_file_location|subprocess|Project-Voice|/mnt/` exits 1 (no matches).
- **Oracle copy:** `oracle.py` matches Voice's ops.py `fb3fca07…` in 11/11 functions (`checks/oracle_verbatim.log`).

### 5. The AdaIN check

**Yes, it tests d/ds against Voice's key.**
- **Same route as Voice:** `compose_adain` uses the same ops as Voice's `adain`/`gemm_style` (`model.py:727-728,746-752`).
- **Comparisons:** dS is compared against both the oracle replay and the stored `vjp_s`.
- **Key check** (`checks/bwd_rev2_adain_ds.py`, all 10 AdaIN fixtures): replay dS equals `vjp_s` exactly and is non-constant (max |dS| 5.95 to 243).
- **Sensitivity:** dropping the γ path exceeds the bound by 2.9e3 to 1.3e4×.
- **Caveat:** the affine operand form differs from Voice's call (F02).

### 6. Anything else

F07, F08, F04 and F10. Separately, core `slice_nd` has no bounds check; that predates BWD-R2, and the candidate's slices are in range.

## Deviations
- Before v1 I read `artifacts/bwd_r2/TOLERANCES_R2.json` and its `.sha256` (the tests load them), the `packages_r2` listing, and the `artifacts/bwd_r2` file names. No report, ledger or receipt content was read before v1.
- My first host probe failed (`RuntimeError: CUDA error at slice: no CUDA-capable device is detected`); that is why the checks use the NumPy stand-in.
- tc_signal's atexit hook prints a no-device `synchronize()` error after each pytest run; my own scripts unregister it.
- Scratch left in place (no deletes): `/tmp/bwd_rev2_pkgs/`, `/tmp/bwd_rev2_mut/` (9 package copies), `/tmp/bwd_rev2_plain/`, `/tmp/bwd_rev2_findings_v1.py`, `/tmp/bwd_rev2_findings_final.py`.

## Prior art
- **Reference adjoints:**
  - ONNX Resize nearest (asymmetric, floor) and Pad reflect index maps (ONNX, 2017–2018);
  - PyTorch `upsample_nearest1d` and `reflection_pad1d` backward (Paszke et al., 2019);
  - Ulyanov, Vedaldi & Lempitsky (2016), instance normalization;
  - Ioffe & Szegedy (2015), the normalization chain rule.

  I re-derived these; the fp32-order references written from DESIGN §5 are mine.
- **Boundary mutants:** mutation testing per DeMillo, Lipton & Sayward (1978). The 8 boundary mutants are mine.
- **Emulated reduction orders:** these are TensorCUDA's own orders (Harris 2007 tree reduction, as the design cites).
- **Path check, call-sequence comparison and NumPy core stand-in:** no prior art known to me.
- Unverified, lead to check: "ONNX Resize asymmetric floor Pad reflect", "PyTorch reflection_pad1d_backward", "Ulyanov instance normalization 2016", "DeMillo Lipton Sayward 1978 mutation", "Harris optimizing parallel reduction CUDA 2007".

## Files
Everything is in `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_rev2/artifacts/bwd_rev2/`:
- `findings.json`, `findings_v1.json`, `findings_v1.seal`, `findings_final.seal`
- `checks/`:
  - scripts: `bwd_rev2_fakecore.py`, `bwd_rev2_paths_check.py`, `bwd_rev2_paths_compare.py`, `bwd_rev2_vjp_check.py`, `bwd_rev2_boundary_mutants.py`, `bwd_rev2_oracle_verbatim.py`, `bwd_rev2_adain_ds.py`
  - outputs: `paths_candidate.json`/`.log`, `paths_prod.json`/`.log`, `paths_compare.log`, `vjp_candidate.json`/`.log`, `boundary_mutants.json`/`.log`, `oracle_verbatim.log`, `adain_ds.log`, `plain_run_no_fixtures.log`, `plain_run_with_fixtures.log`
- `REPORT.md` was refused by my tools; its content is this message.
