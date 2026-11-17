# ENG3-F4: fix ENG3-REV4's test findings, then close (Codex; tests only; GPU lines staged)

Follows: ENG3_F3_R2_RESUME.md
ID reuse: ENG3-F4 is the fourth fix round of the same ENG3 change, as F1 to F3 were

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3` (branch
`eng3`, at the commit carrying this order; ENG3-F3 is `c42d0880`). Edits and CPU runs are AUTHORIZED, limited to:
- `tensor_cuda/tests/test_eng3_f3.py`;
- in `tensor_cuda/tests/test_eng3_host.py`, only the 24 bare-cuda rows of `test_f2_cpu_overflow_message` that F3's
  Amendment 1 edited;
- new `scripts/eng3f4_*.py`, `artifacts/eng3f4/` (new), `logs/eng3f4_*`;
- scratch under `/tmp/eng3f4_*`. Point `TMPDIR` there.

**Read-only:**
- every product file (`bindings.cpp`, `kernels.cu` and the rest): no rebuild, and Build H (`ea723385…`) is the module;
- the earlier artifacts;
- `tensor_fork_apa_wt_eng3rev3` (the reviews: `artifacts/eng3_rev4/findings.json` and its receipts);
- every other worktree;
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why and rulings

ENG3-REV4 (the Claude reviewer, narrow, over F3's Codex-written diff) found the product code correct, with one MINOR
and three NITs, all in tests. Real findings are fixed. The NITs are fixed once and closed. The lead ruled:
- **ENG3R4-F01, fix.** `test_f3_cpu_no_device_query` asserts `CUDA_VISIBLE_DEVICES == ""` instead of creating that
  condition, so its 6 rows fail on any normal GPU host. Run its body in a child process with `CUDA_VISIBLE_DEVICES=""`,
  using the file's own `run_child`. It must then pass on a GPU host and on a host with no device.
- **ENG3R4-F02 (NIT), fix.** `test_f1_gpu_device_index` asserts the exact label: `t.device == f"cuda:{current}"`, for
  both `"cuda"` and `"cuda:<current>"`.
- **ENG3R4-F03 (NIT), fix.**
  - The 24 rows take their expected branch from a registered variable, `ENG3_EXPECT_CUDA`: `1` on a device host, `0`
    on a host with no device. They fail when the ctypes probe disagrees.
  - When the variable is unset, the rows follow the probe, as now, but record which branch ran in the test's output.
- **ENG3R4-F04 (NIT), fix.** Remove the unused child mode `overflow` from `test_eng3_f3.py` (lines of code; no file is
  deleted).

Nothing else changes.

## Gates

- **G1, CPU** (`CUDA_VISIBLE_DEVICES` empty, Build H, `ENG3_EXPECT_CUDA=0`):
  - all of `test_eng3_f3.py` and `test_eng3_host.py` pass, except the GPU-only skips;
  - the 6 no-device rows also pass with `CUDA_VISIBLE_DEVICES` unset in the parent (the child hides the device).
- **G2, regression:** the CPU suite per test ID on Build H with r3's tests against F3 r2's receipts. Identical, except
  the IDs this round touches, each listed.
- **G3, the GPU lane, staged for the lead:**
  - one line runs all of `test_eng3_f3.py` plus the 24 rows with `ENG3_EXPECT_CUDA=1` on Build H. Expected: all pass,
    including the 6 no-device rows (now via the child), and the exact-label test;
  - the F3 r2 `g3-g1` and `g3-g2` comparison forms are re-staged with `ENG3_EXPECT_CUDA=1`, against Build G.

  Each line runs `forge-gpu-run --min 30 --max 540`, through a launcher that sets its full environment and checks the
  module sha before the lease.
- **G4, power:** with `ENG3_EXPECT_CUDA=1` on a no-device host, the 24 rows must fail. That shows the registered branch
  binds. Record it as a CPU receipt.
- **G5, scope:** writes only in the writable set; no product change; no git writes; nothing deleted; no GPU.

**Registered acceptance:** ACCEPT when G1, G2, G4 and G5 pass and the lead's G3 lines pass. These are test fixes, so the
lead's GPU runs verify them. No further review. ENG3 is then promoted.

## Mechanism & scope

- **Measures:** whether REV4's test findings are fixed, so that the shipped tests pass on both GPU hosts and no-device
  hosts and bind their expected branch.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3rev3/artifacts/eng3_rev4/findings.json`,
  with this order's rulings.
- **Exercises:** the two test files; CPU runs on Build H; the staged GPU lane.
- **Does not cover:** product code (unchanged); HOST-COMPUTE-HMM; promotion.
- **Operand census:**
  - the 6 no-device rows;
  - the exact-label test;
  - the 24 rows;
  - the unused child mode.
- **Baseline isolation:** Build H is the module; F3 r2's test receipts are the reference.
- **Required evidence:**
  - the diff (tests only);
  - G1, G2 and G4 receipts;
  - the staged lines with expected exits;
  - `REPORT.md`.
- **Forbidden configuration:** any product file change; any other test edit; a rebuild; GPU use; git writes; deletes.

## Seat

Codex gpt-6.1-sol, high effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable.
- No git writes, no subagents, no background waits; every command under a `timeout`.
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`.
- **Ceiling:** 3 hours. Your final message is the report; also write `artifacts/eng3f4/REPORT.md`.
- **Prior art:** follow `AGENTS.md`.

## Registration checklist (order-lint)

- **L01:** pytest and python3, each run with `--version` first. Build H's module is checked against its sha.
- **L02:** staged lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L09:** G2 compares per test ID. Logs and timings are never compared.
- **L10:** n/a. No identity on stages.
- **L12:** inherited: F3 r2's per-ID identity, with the listed exceptions.
- **L14:** n/a. No gain gate.
- **L16:** the cheapest pass would skip the 6 rows on GPU hosts. Excluded: G3 requires them to run and pass on the GPU
  host, through the child.
- **L19:** n/a. No array identity.
- **L32:** n/a.
- **L35:** your registrations may fix cases. They may not narrow a gate.
- **L38:** only the listed tests are edited.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng3f4_*`.
- **L41:** G4's binding check must execute the registered branch, shown by the recorded branch line.
- **L45:** the lead records each staged line's exits and log sha256.
- **L47:** expected outcomes come from this order.
- **L48:** staged lines run as written, from the worktree root.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty, except G1's unset-parent check, which hides the device
  in the child.
- **L65:** no deletes.
- **L69:** one reference per class: CPU test outcomes come from F3 r2's receipts (G2); GPU comparisons come from
  Build G (G3's re-staged forms).
- **L79:** staged lines' dry runs stop before the lease. Each launcher states its full environment.
- **L82:** `grep -n 'def test_f3_cpu_no_device_query\|def test_f1_gpu_device_index\|"overflow"' tensor_cuda/tests/test_eng3_f3.py`
  names the edit sites, and the 24 rows are those listed in F3 r2's `G2_HOST_COMPARE_r2.json`.
- **L99:** the lead's readings:
  - REV4's findings list F01 at `test_eng3_f3.py:157`, F02 at `:121`, F03 at `test_eng3_host.py:653-667`, and F04 at
    `test_eng3_f3.py:33-55`;
  - Build H's sha is `ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43`.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** n/a.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L111:** n/a.

## Done

Your final message gives, verbatim:
- G1, G2, G4 and G5, each PASS or FAIL;
- the staged GPU lines exactly as the lead should run them, with their expected exits;
- the diff summary;
- the deviations;
- every file written.
