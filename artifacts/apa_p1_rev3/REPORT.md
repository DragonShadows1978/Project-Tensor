# APA-P1-REV3 narrow review

RED: R1 and R3 are incomplete; retained D12 violates non-finite containment. GT PASS: 86/86 IDs, zero skips. CPU/source reasoning only; one GPU reproduction staged, not run.

Reviewed diff: 16 files, +904/-76; product 14 files, +270/-48. All 38 hunks read. Baseline 710cc0d4..71c77cb7 tensor_cuda diff empty. Branch apa-p1rev3. Order sha256 b525b3dc65a5b2ef5cd1e8cefcf180c970fb41f56963eaa490e1c2d06930c446.

## Blind seal and provenance

Start record: `SHIM-RUN 20261116T104839Z-1783116`

Seal: `75d9aae9ba1cb06b9f848ba13837a3cd8c13e7c76416fb8511f412ad3ab12eba  findings.blind.v1.json; DATE 2026-11-16T10:53:53.017101+00:00`

Blind mtime_ns: `1794826433016905294`

First F2 report read: ledger label014 at `2026-11-16T10:54:04.270945+00:00`; command date `2026-11-16T10:54:04,283884891+00:00`. [receipt](receipts/014_first_F2_report_read.log)

Final findings sha256: `63e856a7580d7346b90985e91a511441171ed8436debe5f6948b8d93b88cf6e5`

[LEDGER.jsonl](LEDGER.jsonl) records commands, exits, dates and log shas. [INSTRUMENTS.log](INSTRUMENTS.log) contains required versions/help (git help exit129 expected). No relevant memory notes were used.

## Findings

P1R3-F01 — MINOR — apa_packed.py:446. R1 validates a snapshot, then lookup rereads the caller-controlled mapping. A public hand-built table with a stateful dict supplies pinned validation values but returns e_q=0; attention reaches the recording backend with delta=0 at epsilon=1. Loaded pinned tables are safe. Fix: return the validated snapshot or canonical pinned value. Confirmed on CPU.

Prior art: TOCTOU / CWE-367, unverified -- lead to check: CWE-367 check use snapshot; SHA-256 NIST FIPS180-4 (2015). No new technique.

Evidence: BLIND_CHECKS.json:margin_pin_bypass, receipts/012_blind_checks_repair2.log, apa_packed.py:463-473,440-446,488-499.

P1R3-F02 — MAJOR — packed.cuh:108. D12 can hide a dense non-finite row: at D64, finite k[1,0]=1e20 and q=-ones yield norm=Inf, code 8 and packed bulk=-Inf, while dense reconstruction has mixed-sign infinities and a NaN dot. Beside zero keys, packed skips the bad key. This is retained behavior, not a new F-to-F2 finite-result change. Fix: carry a NaN norm on norm overflow; the lead must resolve LF1’s finite-operand constraint. Source/CPU arithmetic supports this; GPU confirmation is pending.

Prior art: IEEE 754-2008 overflow/NaN propagation; TurboQuant (Zandieh et al. 2025) inherited norm/rotation. External literature unverified -- lead to check: TurboQuant normalized key norm overflow.

Evidence: BLIND_CHECKS.json:arithmetic finite_squared_norm_overflow, receipts/012_blind_checks_repair2.log, packed.cuh:94-108,198-201, quant.py:122-127, D12_SOFTMAX.json, receipts/031_dense_nonfinite_arithmetic.log, tensor_cuda/src/apa_sp1_1.cuh:73-102, repro/r01_finite_norm_overflow.py, receipts/026_repro01_cpu_prefix_repair1.log.

P1R3-F03 — MINOR — eb2b.cuh:159; launcher.cuh:30-34,132-134. R3’s census overstates coverage: EB2b evaluates B*H in signed int before checking its range, and attention extents above INT_MAX still raise generic RuntimeError instead of operand-naming ValueError. The later rotate guard catches oversized B*H under ordinary execution; no out-of-bounds access was demonstrated. Fix: widen/check the multiplication first and implement R3’s extent error contract.

Prior art: ISO C++17 (2017) signed integer range/overflow; ordinary checked multiplication, no prior art known to me for the exact repair.

Evidence: BLIND_CHECKS.json:unguarded_eb2b_product, receipts/007_index_entries.log, eb2b.cuh:157-159, launcher.cuh:30-34, artifacts/apa_p1f2/INDEX_CENSUS.json, receipts/023_ledger_tables_bindings.log, receipts/029_dense_and_dependencies.log, packed_launcher.cuh:26-28, eb2b.cuh:179.

## Fix completeness

| REV2 / ruling | Disposition | Evidence and boundary |
|---|---|---|
| P1R2-F01 / R1 | INCOMPLETE | Common pinned-load, wrong sha, wrong entries and subclass cases fixed; stateful hand-built mapping still bypasses use pin (P1R3-F01). Read/run: apa_packed.py:408-473,476-516; receipts012; GT R1 tests |
| P1R2-F02 / R2 | COMPLETE | The prescribed non-finite-input fix is complete across the one shared TQ4 prepare/pack quantizer; the universal containment claim still fails on finite norm overflow (P1R3-F02). Read/run: packed.cuh:101-112,15-22; CENSUS_AUDIT.json; BLIND_CHECKS.json |
| P1R2-F04 / R3 | INCOMPLETE | Capacity and main helper bounds fixed; signed multiplication in EB2b and attention out-of-range error contract remain (P1R3-F03). Read/run: eb2b.cuh:158-159,179; packed_launcher.cuh:21-70; launcher.cuh:30-34,132-146; GT R3 tests |
| P1R2-F05 / R4 | COMPLETE | All six raw helpers refuse non-positive leading extents on the host; EB3 timeline G=3,5,6,7,8 refuses before lock/arena/launch; G2/G4 remains accepted. Read/run: packed_launcher.cuh:23-75,93-118; eb2b.cuh:106-134; eb3.cuh:939-960; GT source-order test |
| P1R2-F06 / R5 | COMPLETE | Comments correctly restrict identity to finite intermediate scores and denominator; overflow, NaN/+Inf sinks and Inf queries are named. No merge arithmetic changed in this diff. Read/run: merge.cuh:23-42; eb3.cuh:782-800; eb4.cuh:645-658; review.diff |
| P1R2-F07 / R6 | COMPLETE | Header has the one current 16-argument declaration; direct tc binding replaces the bridge. Native syntax probe passes after and fails before as intended. NIT closed. Read/run: ops.h:328; eb2b.cuh:146-149; bindings.cpp:2803-2817; GT R6 tests (including syntax-only nvcc probe) |
| P1R2-F08 / R7 | COMPLETE | Active expectation path/hash recorded beside original; paired renames separated from additions/removals. Synthetic tests pass. Registered G6 remains FAIL for unregistered path-derived names. NIT closed. Read/run: scripts/apa_p1f1_sass.py:47-176; GT R7 tests; SASS_RECOUNT.json |
| P1R2-F09 / R8 | COMPLETE | Keyword-only capacity forwarded for packed; dense/fallback rejects non-None capacity before device work; None default unchanged. Read/run: apa_sp2.py:61-111; __init__.py:1632-1680; GT R8 tests |

## Questions 2 to 8

2. No demonstrated new finite-result change, valid-input refusal, off-by-one bound or broken binding. P1R3-F01 is an incomplete new guard; F02/F03 are retained defects/contract gaps. The +127 eb2b_values bound is needed for its existing int launch numerator (total+127), not merely its largest thread index. S+2047 exactly bounds the existing int partition expression. R4s exception changes are authorized; R3s positive out-of-int exception contract is incomplete. Evidence: all 38 hunks in HUNK_COVERAGE.json; eb2b.cuh:113,118-119; launcher.cuh:101-104,143; GT86; SASS3 changed prepare bodies.

3. One shared TQ4 key quantizer: prepare owns norm/rotation and pack_tq copies norm into scales; EB2b/EB3/EB4/SPQ4/SPQ6 read it. All owned packed quantizers carry non-finite inputs as NaN. For k0=+/-Inf, old norm Inf plus code0 gives coefficient -1.6977524757 and old score -Inf; F2 stored norm/score are NaN; dense has 33 +Inf and 31 -Inf components and NaN dot. NaN-input classification arithmetic is unchanged. Universal containment is false because D12 is outside that input-only test. Raw caller-supplied codes/norms are a caller contract, not another owned quantizer. Evidence: BLIND_CHECKS.json, CENSUS_AUDIT.json (10 TQ4 rows, five matching source files); packed.cuh:15-22,101-112; readers/callers in receipts005-007.

4. Yes: a public constructed stateful mapping returns pins during validation, then zero during lookup; the recording attention backend receives e_q=0 and delta=0. Real load() cannot override the digest and its immutable mapping is safe. Evidence: P1R3-F01; BLIND_CHECKS.json; receipts012; apa_packed.py:440-446,463-473,488-499.

5. The 16-row index census matches its source hashes but is not complete as a correctness claim: EB2b B*H is evaluated in signed int before its check. Its later rotate guard mitigates ordinary execution before allocation. Native capacity checks and prepare/pack/rotation product checks precede allocation/launch; other main guards use safe int64 products or division. Positive attention extents beyond int range still return the wrong R3 exception contract. Evidence: CENSUS_AUDIT.json, BLIND_CHECKS.json; eb2b.cuh:158-159,179; packed_launcher.cuh:26-28; launcher.cuh:132-146.

6. MAJOR: a finite row k0=1e20 squares to Inf in fp32. Norm Inf gives unit0 and code8; q=-ones yields coefficient -0.0803470984 and packed score -Inf. Dense reconstruction has 31 +Inf and 33 -Inf coordinates, hence NaN dot. Packed softmax weights [1,0,1] and denominator2 are finite; dense NaN accumulator times its guarded zero remains NaN. Thus it can hide a dense non-finite row. This does not demonstrate a new F-to-F2 finite result change: both retain the same overflow path. No claim of general finite-output parity with dense is made. Evidence: BLIND_CHECKS.json; D12_SOFTMAX.json; quant.py:122-127; packed.cuh:94-112,198-201; apa_sp1_1.cuh:73-102; staged r01, CPU prefix PASS.

7. Own recount: F781, F2 781, paired781, equal778, changed3, added0, removed0. Lead: 781/build, changed3, added0, removed0. Only prepare<float>, prepare<__half>, prepare<__nv_bfloat16> differ. Raw names pair521 (518 equal/3 changed) with260 only per side; independent path-length/hash map pairs all260. Exact body bytes untouched. G6 as registered remains FAIL. Evidence: SASS_RECOUNT.json; receipts020-021; sha-verified streaming inputs.

8. No further production defect detected within these 16 files/38 hunks. Ship disposition remains RED for the retained non-finite violation and incomplete guards. No GPU, model quality, speed or promotion conclusion was measured. Evidence: HUNK_COVERAGE.json, CLAIM_CHECKS.json, GT and source/census audit.

## Census and receipt claim audit

Recomputed beside F2: TQ4 rows 10 (F2 10); index rows 16 (F2 16); spo2 definitions 15 including 2 hooks-only (F2 15), raw helpers 6 (F2 6); boundary rows 47 (F2 47), of which 44 have f2_ok=true and 3 are source-only/null. All 16 source hashes in each of the three source censuses match live code. Per-row dispositions: [CENSUS_AUDIT.json](CENSUS_AUDIT.json).

F2’s claims that all eight findings are fixed and packed never hides dense non-finite rows are too broad. Its D12 acknowledges an unchanged path but does not test it. Its finite GPU counts remain bounded authored receipt claims. [CLAIM_CHECKS.json](CLAIM_CHECKS.json) gives each claim disposition. No quoted lead-ledger entry or path exists in the supplied REPORT; none was found in the named local ledgers searched. A path clarification was requested. The lead SASS receipt itself was read after the independent count.

## Step T / GT

Only tensor_cuda/tests/test_apa_p1d_policy.py changed: +38/-22. The helper loads the real pinned table through public load(). Forwarding expectations read e_q from the pinned JSON. The loader test keeps all malformed synthetic registration/entry cases and checks R1 refusal before any recorded allocation/device work, plus the pinned positive controls and lookup-domain refusals. No assertions removed to suppress a tested property; no skips.

| File | IDs | Before | After |
|---|---:|---|---|
| test_apa_p1d_policy.py |40|32 PASS /8 FAIL|40 PASS|
| test_apa_p1f1_cpu.py |15|15 PASS|15 PASS|
| test_apa_p1f2_cpu.py |31|31 PASS|31 PASS|

Exactly these eight IDs change FAILED -> PASSED; every other one of the 86 per-ID outcomes remains PASSED:

- `tensor_cuda/tests/test_apa_p1d_policy.py::test_stale_generation_is_refused`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_packed_table_on_dense_kq_is_refused`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_packed_route_refusals_before_launch`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_packed_route_forwards_delta_causal_sinks_and_r2[64]`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_packed_route_forwards_delta_causal_sinks_and_r2[128]`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_packed_loader_checks_its_own_registration_quantizer_and_entries`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_routes_are_reported_for_quantize_and_attention`: FAILED -> PASSED.
- `tensor_cuda/tests/test_apa_p1d_policy.py::test_prefill_on_packed_keys_is_refused_naming_dense`: FAILED -> PASSED.

Collection is identical, with zero collection errors. Before agrees per ID with F2’s CPU_G3_COMPARISON and S4 receipts. The other 689 IDs of the 744-ID inherited comparison were not rerun; all other test files and product sources remain byte-identical, and no other test file imports this helper (receipt029). [T_RESULTS.json](T_RESULTS.json) lists all 86 IDs and their recorded/before/after outcomes; [T.diff](T.diff) is the exact edit. Test options match F2: -q --tb=short -ra -p no:cacheprovider --basetemp under scratch, plugin autoload disabled, module first on PYTHONPATH. Recorded commands/outputs: receipts017 and019. The inherited R6 test runs syntax-only probes under scratch; no engine build occurred.

## SASS recount

| Count | Own F / F2 recount | Lead |
|---|---|---|
| Kernels per build |781 /781|781 /781|
| Identical bodies |778|778 inferred from781-3|
| Changed |3|3|
| Added / removed |0 /0|0 /0|

Raw-name counts: 521 paired, 518 identical, 3 changed, 260 only per side. Own bijective mapping normalizes nvcc path hashes, path-length fields and _cu_ suffix fields in SYMBOLS ONLY; body bytes are not changed. Exactly 260 name pairs become matched; only prepare<float>, prepare<__half>, prepare<__nv_bfloat16> differ. [SASS_RECOUNT.json](SASS_RECOUNT.json) includes every pair and body hash. Registered G6 FAIL is preserved.

## Builds and verified dump hashes

- F: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/arms/F/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` sha256 `5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243`.
- F2: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/arms/F2/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` sha256 `6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579`.
- F.sass: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/sass/F.sass`; 236887417 bytes; sha256 `b36e77cae62835462f547f30e13f2d358ce896d9736bff7745c9cdc43af9e77a` (matches HELD_BACK.txt, checked before streaming use).
- F2.sass: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/sass/F2.sass`; 236924021 bytes; sha256 `3b7efd2a7b6608a3f482374bcecf7273b3586ee5444a96166d2bf2eb6df1de3b` (matches HELD_BACK.txt, checked before streaming use).

## Staged reproduction

- Path: `artifacts/apa_p1_rev3/repro/r01_finite_norm_overflow.py`
- sha256: `afce010a8678490475d7571e01de6018a7a5507812039a26e242783a05801150`
- CPU prefix: PASS, `receipts/026_repro01_cpu_prefix_repair1.log`, receipt sha256 `8ca1e36f131905b8b7e2936ee538f24d195cc02bde9f56242cec7d62e8a8d165`; real imports and both public table signatures exercised, stopped before tc.tensor.
- Confirm: finite control valid, dense has a non-finite row missing from default packed output.
- Refute for this instance: finite control valid, dense non-finite row retained in packed on this instance. Dense all-finite is inconclusive; failed finite control or exception is ERROR.
- Lead run, unchanged from this worktree root: `forge-gpu-run --min 30 --max 600 --label p1rev3-01 -- timeout 560s python3 artifacts/apa_p1_rev3/repro/r01_finite_norm_overflow.py`.

## Scope, deviations and limits

Only the authorized policy test and artifacts/apa_p1_rev3 were written in the worktree. SHA census covered116 named source/test files; only the policy file changed; read-only git diff agrees. No GPU, builds, git writes, deletes or subagents. Scratch is /tmp/p1_rev3_r1. The new Python basenames do not shadow standard-library modules.

[DEVIATIONS.md](DEVIATIONS.md): initial rule-read used TMPDIR=/tmp and omitted nice/thread variables; corrected thereafter. blind_checks.py had two repairs (wrong delta assertion then query sign for the central code); staged r01 had one repair (missing held-back dense table, now sha-verified read-only production path). Report assembly had a shell-quoting write slip, repaired using a literal heredoc. Failed receipts are retained. F03 was narrowed after tracing the downstream rotate guard. The optional lead-ledger path is unavailable. No L35 or time ceiling rail was reached.

Would delete: nothing in the worktree. Keep all receipts, including failed attempts, for review. Optional scratch cleanup by the lead: /tmp/p1_rev3_r1; nothing deleted here.

## Prior art

- IEEE754-2008/2019 propagation: inherited arithmetic used to construct Inf and norm-overflow cases; no new numerical algorithm.
- TurboQuant, Zandieh et al. (2025): reused live rotation/codebook functions unchanged; no new quantizer. Literature unverified -- lead to check: TurboQuant normalized key norm overflow.
- SHA-256, NIST FIPS180-4 (2015): immutable pins and receipt hashes, inherited.
- ISO C++17 (2017): signed overflow and range rules; checked multiplication proposed. Exact census/repair: no prior art known to me.
- TOCTOU/CWE-367 (MITRE, year unverified -- lead to check: CWE-367 check use immutable snapshot): use the validated value rather than rereading mutable input.
- Meszaros, xUnit Test Patterns (2007): inherited recording fakes for CPU policy tests. Only fixtures/assertions were adapted.
- NVIDIA cuobjdump Function blocks and c++filt: independent streaming body hashes; symbol canonicalization for this recount: no prior art known to me.
- Milakov and Gimelshein (2018), Dao et al. FlashAttention (2022): inherited online-softmax reasoning only, no merge change proposed beyond the stated norm sentinel.

## Python files written

- `artifacts/apa_p1_rev3/blind_checks.py`
- `artifacts/apa_p1_rev3/finalize_review.py`
- `artifacts/apa_p1_rev3/policy.before.py`
- `artifacts/apa_p1_rev3/recount_sass.py`
- `artifacts/apa_p1_rev3/repro/r01_finite_norm_overflow.py`
- `artifacts/apa_p1_rev3/run_T.py`
- `artifacts/apa_p1_rev3/run_review.py`
- `tensor_cuda/tests/test_apa_p1d_policy.py`

All parse successfully; none shares a standard-library module basename. policy.before.py is the unmodified reference copy.
