# ENG3-F3 r2 registration — immutable, before gates

Start: SHIM-RUN 20261117T080246Z-3662740, logs/eng3f3_r2.log.
Plan: orders/ENG3_F3_REV3_FIXES.md plus lead Amendment 1 in
orders/ENG3_F3_R2_RESUME.md. Both were present before launch; no git used.
Mechanism read: original F3 order, Amendment 1, REV3 findings.json, r1
G2_HOST_COMPARE.json, shipped tests and runners. This work exercises exactly
the amended early-query error path and the F01 pinning docstrings.

Only test edits: test_f2_cpu_overflow_message's 24 bare-cuda rows get a real
libcudart.so.12 cudaGetDevice query. On query failure they require RuntimeError
naming cudaGetDevice and the hook; on query success the original OverflowError
shape/dtype assertions execute. CPU rows retain their original path and checks.
No skips, stubs, product changes, or rebuilds. F01 citations go in both new
zero-extent test docstrings, citing ENG3R3-F01 and A1.2.

G1_r2: CUDA_VISIBLE_DEVICES empty, pinned H, same cpu-or-moved-test selection
as r1. Require pytest exit 0, all 267 CPU IDs pass, moved GPU test skips for
no device. All 24 amended rows must execute the query-failure branch and
assert RuntimeError/cudaGetDevice, never skip.

G2_r2 host: same amended test tree and harness on G and H. Census and
collection-error sets must agree with r1. Only inherited outcome exceptions
are the exact 24 IDs in r1 inherited_unapproved_differences; list every ID
and both actual exception outcomes/messages. On CPU expect amended tests to
fail G with OverflowError and pass H with RuntimeError. G1's H expectations
are never changed to accommodate G. Every other inherited ID must match.
The 14 new/moved IDs and six F04 message differences stay separately listed.
Record r1's original G/H outcomes too. Compare IDs/outcomes only, never logs
or timings. Require no missing/pending/error IDs, unexpected differences,
or amended-row skips.

G2_r2 engine: rerun the frozen F2 CPU suite form, reference G versus H,
with inherited census/options and collection-error file sets. Require
per-ID identity, nonempty census, no writes by tests. Inherited suite failures
and collection errors remain explicit; wrapper exit 0 is not a clean pytest
suite. Fresh r2 receipt and temporary paths; no r1 overwrite.

G3 PASS carried only from r1: three kills, with original mutant modules,
test receipts and hashes checked unchanged. No new mutation or kill claimed.
G4: before/after cwd metadata census; hash all r1 artifacts and product/order
files; test diff confined to A1.1/A1.2. No git commands/writes, subagents, GPU,
deletes, or product edits. New filenames use _r2 suffix. r1 receipts remain
byte-identical. Active shim logs excluded from stable hash sealing.

GPU restaging: new _r2 copies of inherited g3-g1/g3-g2 launchers/helpers,
explicit environment, G/H module SHA256 checks BEFORE lease, real host CPU
prefix on each arm stopping before device allocation. Each real line uses
forge-gpu-run --min 30 --max 540. Add all 48 overflow-message rows and moved
GPU-index test inside each lease after the inherited split suite; require all
pass and identical G/H, exactly the 24 bare-cuda rows taking query-success
OverflowError branches. No CPU-only no-device-query test on GPU. Real
forge/inner/launcher expected exits 0; dry forge/launcher expected 78, inner
null, no run/no lease. Seat invokes only FORGE_GPU_DRY_RUN=1. Lead receives
exact commands and finish receipts recording exits and GPU log SHA256.

Tools: pytest --version first = 9.0.2; python3 --version first = 3.12.3.
G sha256 5fcd179b98b0749af0d75414d35ae0fc8519bf200364c30c743904a90fc1a711.
H sha256 ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43.
Both verified against existing build receipts before this registration.
Commands foreground under timeout, Python -B -P, no cacheprovider,
TMPDIR/basetemp fresh under /tmp/eng3f3_*_r2. Any running exec session is
resumed with write_stdin yield_time_ms 300000. No status between those waits.
Maximum 3 repairs of own scripts, all reported, fresh count for r2.
All other F3 checklist answers unchanged; A1.1–A1.3 are lead amendments.

Prior art: public tensor_empty/P1T-FIX4 (local 2026) and CUDA Runtime API
cudaGetDevice, taken: real current-device query/error branching; ours: the
amended assertion placement, not a new device-selection algorithm. NumPy
nominal-size refusal/P1T-FIX2 (2026), taken: existing F01 ruling, no new byte
rule. ENG3-F2/F3 harness (2026), pytest hooks (2004+), SHA-256 (NIST 2001),
taken: per-ID comparisons, phase folding and pinning; ours: r2 receipts and
explicit GPU overflow parity. External literature unverified — lead to check:
CUDA Runtime API cudaGetDevice, NumPy array is too big, pytest hook API.
