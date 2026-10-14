# Engine build log: what `_tensor_cuda` is installed in production

The compiled engine `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` is gitignored, so git
history alone does not show which binary consumers load. Every install is recorded here.

## 2026-10-13 ~19:15 EDT: installed d6b1b7ad (main 3a678e0 sources, P1-T DLPack) and the new tc_signal add-on 30aee4e6

- **Installed, core:** sha256 `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24`.
  - It was copied, not rebuilt, from fork worktree `Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8`, built
    with `./build.sh 89` from e4cf1d6.
  - Main 3a678e0's `tensor_cuda/` and `tc_signal/` are identical to e4cf1d6 (`git diff --quiet e4cf1d6 3a678e0 -- tensor_cuda tc_signal`).
  - It is the binary behind the P1-T FIX8 lead verification and the round-9 review.
- **Installed, add-on (new):** `tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so`, sha256
  `30aee4e67fbe80320863cf9d0d4b7748dca46c0db7a3689f70bcc9d9e8dc782b`. It is gitignored, and it does not link
  `_tensor_cuda`.
- **Replaced:** sha256 `d8fda9a8879b2c0de1b659653e8899aa6fbb89da36af03d1339e2a1c9b011e31` (the 2026-10-09 install).
- **Backup:** `tensor_cuda/engine_backups/_tensor_cuda.d8fda9a8.installed-20261009.so`.
- **Method:** as on 10-09. `cp -p` the old binary to the backup, `cp -p` the new one to `<so>.d6b1b7ad.part`, then
  `mv -T`. The add-on went the same way, from `<so>.30aee4e6.part`. No process had the production binary mapped.
- **Linkage:** CUDA 12.6 (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`).
- **Verification**, with the production packages and the installed binaries. The PYTHONPATH was
  `tensor_cuda/tensor_cuda:tensor_cuda:tc_signal` of this repo, and the import paths were confirmed. The test tree was
  wt_fix8, which equals main for these paths.
  - Full suite: 37 failed, 3040 passed, 47 skipped, 15 errors. FAILED/ERROR equals the 52 baseline ids, with 0 new.
  - Voice runner: PASS 249/249. Project-Voice `git status` clean.
  - Receipts: `Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8/artifacts/p1t_fix8/lead/prod_install/`.
  - Driver: 595.99.02.
- **Approval:** David, 2026-10-13. He answered the question "After the merge, should I also install the new engine .so
  into production?" with: "I meant to select yes to the second one". The merge itself went ahead under his review
  override; see `docs/P1T_PROMOTION_RECORD.md`.
- **Known issues in this build:** V9-01 and V9-02, both MINOR and diagnostics-only, from P1-T review round 9.
- **Rollback:** `mv -T` the backup back over `<so>`, after copying it, so that a backup is kept. Then remove the add-on
  binary by moving it to DELETE_ME with `delete-me`. Pre-P1-T consumers never import `tc_signal`.

## 2026-10-09 ~20:13 EDT: installed d8fda9a8 (main a28e28e sources, mask-fix kernels)

- **Installed:** sha256 `d8fda9a8879b2c0de1b659653e8899aa6fbb89da36af03d1339e2a1c9b011e31`, 10,048,000 bytes.
  - It was copied, not rebuilt, from the fork's pinned main build
    `Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/artifacts/apa_mf_main/build/`, built 2026-10-08 11:17.
  - Its `build_manifest.json` lists 19 engine source sha256s. The APA-REGT1 seat checked all 19 against a28e28e:
    0 mismatches (`artifacts/apa_regt1/main_engine_manifest_check.log`).
  - It is the binary behind every APA test and review receipt since 2026-10-08, including the MF chain, LD1–LD3,
    OM107W/116/135/178/179 and REGT1.
- **Replaced:** sha256 `b6022d3dcd5b3b7662f039d6d372f86339b92ec3303b2f073577d10d880d1e2f`, built 2026-09-14 07:02.
  - It predates the mask-fix merge a28e28e. So the kernel half of that merge was never compiled into production:
    - 93810ef: −inf/+inf reduction identities;
    - 3299218: noncausal fused online-softmax guards;
    - 7bdc81d: fully-blocked noncausal rows return 0.
  - Reported by the GRM lead, 2026-10-09 (SessionMail `20261009-181500_GraftRepositoryMemory_tensor-so-predates-mask-fix.md`).
  - Per those commits, only rows containing −inf or values beyond the old ±3.4e38 sentinels change. Finite rows are
    byte-identical.
- **Backup:** `tensor_cuda/engine_backups/_tensor_cuda.b6022d3d.built-20260914.so`. It is gitignored, and it is not
  importable under that name.
- **Method:**
  1. `cp -p` the old binary to the backup path.
  2. `cp -p` the new binary to `<so>.d8fda9a8.part` in the package directory.
  3. `mv -T` it over `<so>`. That is an atomic rename, so a process with the old binary mapped keeps its old inode.
     No process had it mapped at install time (checked via `/proc/*/maps`).
- **Linkage:** both binaries link CUDA 12.6 (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`).
- **Verification:**
  - The full suite ran with the production package and its installed binary. The PYTHONPATH was
    `tensor_cuda/tensor_cuda:tensor_cuda`, from a fork worktree whose tree equals production main 2f2dd93 (tree
    1b0e989).
  - Result: 37 failed, 1956 passed, 46 skipped, 15 errors. The FAILED/ERROR set equals the 52 baseline ids, with 0 new.
    The 31 APA-REGT1 regression tests pass.
  - Receipt: `Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_regt1/logs/apa_regt1_prod_engine_suite_20261009_2017.log`.
  - Driver: 595.99.02 (native).
- **Approval:** David, 2026-10-09, "Yes, yes, and yes", answering the question "Should I install the verified engine
  build (`d8fda9a8`) into production?"
- **Rollback:** `mv -T` the backup back over `<so>`, after copying it, so that a backup is kept.
