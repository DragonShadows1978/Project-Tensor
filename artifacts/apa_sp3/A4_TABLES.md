## A4 full-context reference — model perplexity

T: pinned HF MiniCPM3 snapshot, bf16 weights, full SDPA attention. Flash is preferred only when eligible on actual Q/K=96, V=64 tensors; otherwise efficient is forced. Math fallback is disabled. Actual backend is UNRUN until a T receipt exists.
1024 uses six independent windows / 3072 total targets. Long rows use prefix 0 / 512 targets. Only required LM-head rows are projected in new T and 32K engine cells. Attention is never chunked in these PPL/reference cells.

| S | Arm | Status | PPL | Engine minus T | SDPA backend |
|---:|---|---|---:|---:|---|
| 1024 | T | PASS | 7.610716 | — | ['SDPBackend.EFFICIENT_ATTENTION'] |
| 1024 | A | PASS | 8.655828 | 1.045111 | — |
| 1024 | B | PASS | 8.778802 | 1.168086 | — |
| 1024 | C | PASS | 8.661116 | 1.0504 | — |
| 1024 | D | PASS | 8.655352 | 1.044635 | — |
| 8192 | T | PASS | 8.475996 | — | ['SDPBackend.EFFICIENT_ATTENTION'] |
| 8192 | A | RED | — | — | — |
| 8192 | B | PASS | 10.403578 | 1.927582 | — |
| 8192 | C | PASS | 10.042419 | 1.566423 | — |
| 8192 | D | UNRUN | — | — | — |
| 32768 | T | PASS | 6.603129 | — | ['SDPBackend.EFFICIENT_ATTENTION'] |
| 32768 | A | UNRUN | — | — | — |
| 32768 | B | UNRUN | — | — | — |
| 32768 | C | UNRUN | — | — | — |
| 32768 | D | RED | — | — | — |

INT4 engine versus bf16 T gaps are observations, not RED parity failures. D@32768 requires T with identical targets; the 0.005 D/A gate remains confined to existing engine controls. D@32768 adds a layer-0 first-128-query refine-all check; no full 62-layer 32K diagnostic claim.

## A4 ceiling grid — kernel sweep / memory shape

| Arm | S | Status | Fit | Outcome | Peak resident MiB estimate |
|---|---:|---|---|---|---:|
| A | 4096 | PASS | True | FIT | 5925.15625 |
| A | 8192 | UNRUN | — | — | — |
| A | 16384 | PASS | False | OOM | 3966.5 |
| A | 24576 | PASS | False | OOM | 4493.5625 |
| A | 32768 | UNRUN | — | — | — |
| B | 4096 | PASS | True | FIT | 4924.25 |
| B | 8192 | PASS | True | FIT | 4436.59375 |
| B | 16384 | PASS | True | FIT | 5959.40625 |
| B | 24576 | RED | — | — | — |
| B | 32768 | RED | — | — | — |
| C | 4096 | PASS | True | FIT | 3675.1875 |
| C | 8192 | PASS | True | FIT | 4436.59375 |
| C | 16384 | PASS | True | FIT | 5959.40625 |
| C | 24576 | PASS | True | FIT | 7482.21875 |
| C | 32768 | RED | — | — | — |

Measured grid summary: `{"A": {"grid_complete": false, "max_successful_grid_S": 4096}, "B": {"grid_complete": false, "max_successful_grid_S": 16384}, "C": {"grid_complete": false, "max_successful_grid_S": 24576}}`.
A timeout is unknown fit, never an OOM or successful prefill. Every grid point is independent. Largest successful S is only a grid result, not an extrapolated capacity or model-quality finding.

## A4 capture split and immutable receipt handling

8192: [0,16), [16,32), [32,48), [48,62), at most 188s planning estimate each. 32768: 62 single-layer ranges, 188s each under the registered quadratic extrapolation. These are unmeasured estimates; timeout rails remain authoritative.
Ranges restore predecessor hidden activations; each layer sees the full token prefix. Row-block diagnostic replay is checked bitwise against native output. The unchanged margin ids depend on the original capture id, now an aggregation cell. New/changed receipts are in jobs_a4; legacy RED/PASS receipts are preserved.
Aggregation rehashes each layer manifest and checks exact stat identity of every array since its completed range SHA256. It pins all 62 manifests as one set. Margin workers rehash their input arrays. 32768 captures are B/C bulk4; no new 32768 margin/E calibration cells were authorized.

Fingerprint compatibility is governed by amendment_006_fingerprint.json: per-kind import closures and exact reviewed source transitions. Unknown closure changes reject reuse. Source eligibility and current runtime/build prerequisites are reported separately in a4_receipt_audit.json.

