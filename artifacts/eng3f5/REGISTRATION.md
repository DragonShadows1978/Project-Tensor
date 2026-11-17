# ENG3-F5 registration (immutable; before gates)

Order/plan: orders/ENG3_F5_SHIM_SELF_CONTAINED.md, unchanged.
Rules: /mnt/ForgeRealm/HOUSE_RULES.md mirror, Project-Tensor/AGENTS.md,
and worktree AGENTS.md. No git commands, subagents, background waits,
deletes, product edits/rebuilds, or GPU. Every shell command under timeout.
Tool versions checked before use (L01.txt); TMPDIR /tmp/eng3f5_run.
Module: Build H, SHA256 ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43,
verified before registration and before each test run; pin the real imported module.
Mechanism: MAINCHK1 REPORT.md lines 703,870 name the failed ID and absent
artifacts/eng3/tools/cudart_count.so. ENG3 REPORT.md line 17 describes the
32-import counter and positive control. The ordered compile/preload path
exercises this exact mechanism. No product or GPU coverage claimed.

Implementation: change only SHIM and test_g1_cpu_no_cuda_calls; source byte
copy artifacts/eng3/tools/cudart_count.c to tensor_cuda/tests/support/.
No full compile command found by cudart_count/shared/general-regs searches
in artifacts/eng3 text receipts (LEDGER.md, REPORT.md, logs and scripts).
Source comment mentions -mgeneral-regs-only but is not a recorded command.
Use the order's exact fallback flags: cc -shared -fPIC -O2 -o
<pytest tmp_path>/cudart_count.so <support>/cudart_count.c -ldl.
Compiler executable selection: first available cc, gcc, clang. Only absent
compiler permits pytest.skip; compiler failure must fail with compiler output.
Keep child and both counter assertions unchanged. Add local prior-art comment.

G1 PASS requires BOTH fresh-tree and worktree named-test runs to pass,
zero skips, host_calls=0 and control_calls>0, with shim compilation in their
unique pytest tmp dirs. Fresh scratch receives shipped test tree/source only
(no copied gate harness or compiled artifacts). Run from fresh cwd with clean
Python/loader paths: Build H package, standard Python/site directories, CUDA
runtime library path; no prior artifacts/eng3/tools/cudart_count.so candidate.
Record compile argv and generated shim hash. No old binary is provisioned.

G2 PASS requires full two-file CPU run identical per ID to F4
artifacts/eng3f4/host/G1_all/RESULT.json: 313 IDs, 267 pass/46 skip.
Reuse F4's runner-only skipping of 45 GPU-only host rows, with CVD empty,
ENG3_EXPECT_CUDA=0; existing device-index unavailable-runtime skip retained.
Also compare all 268 F4 host/G2_host/RESULT.json IDs against the same run:
267 pass/1 skip. No outcomes waived, no IDs omitted, no logs/timing compared.
Only listed mechanism exception: no_cuda_calls now compiles shipped C source
instead of requiring an untracked .so. Capture per-phase outcomes and records.

G3 PASS requires scratch-only counter mutant after passing G1/G2 baseline.
Fresh copy of shipped test tree has identical test bytes, support C source
mutated only to remove all 32 total/per-function increment pairs (64 atomic
increments). Test compiles this source itself, preloads the resulting tmp .so.
Require pytest exit 1, exactly the named test failed at control_calls>0,
control_calls=0 and host_calls=0. Record source diff/hash and compiled shim hash.
Never mutate the working source or supply a precompiled mutant.

G4 PASS requires before/after worktree metadata census, hashes of product
source/package/Build H/tests, unchanged old tool C and order, identical seat
.git admin hashes; no removals, no changed paths outside the ordered writable
set; byte comparison outside SHIM and the one function; new C byte identity.
All new files and retained /tmp/eng3f5_* scratch listed. No git calls/writes,
no GPU use; direct command audit supplements file-state evidence.
Test cache disabled, unique --basetemp and TMPDIR under /tmp/eng3f5_*;
helpers run python3 -B -P. No standard-library helper names.

ACCEPT iff G1-G4 pass. Expected negative pytest failure only for G3.
Stop/report any unexpected gate failure; never weaken tests. At most three
own-script repairs, recorded separately before reruns. Deadline two hours.
Registration amendments are separate files, never edits to this file.

## Prior art
Pytest tmp_path/hooks (Krekel et al., 2004+), Python subprocess (PSF, 2003+),
ENG3 LD_PRELOAD/dlsym shim (local 2026; glibc ld.so interposition, year
unverified), F4 per-ID receipt harness (local 2026), SHA-256 (NIST, 2001),
mutation testing (DeMillo, Lipton and Sayward, 1978).
Taken: temp fixture, compiler subprocess, existing interposer/assertions,
per-ID equality and counter-silencing negative control. Ours: F5 test-local
build wiring and receipt assembly. External literature unverified -- lead
to check: pytest tmp_path; Python subprocess; glibc LD_PRELOAD; mutation
DeMillo Lipton Sayward 1978; NIST SHA-256. No new numerical algorithm.
