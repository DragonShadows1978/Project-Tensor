# ENG3-F2: fix ENG3-REV2's two diagnostic findings, then close (Codex; CPU round with a build; GPU lines staged)

Follows: ENG3_F1_REV1_FIXES.md

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3`
(branch `eng3`, at the commit carrying this order; ENG3-F1 is `3597cf84`). Edits, CPU runs and engine builds are
AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`, only the byte-bound check that ENG3-F1 added to the NDArray allocating constructor;
- `tensor_cuda/src/bindings.cpp`, only the device-string refusal messages in the private `_core_*` hooks;
- `tensor_cuda/tests/test_eng3_host.py`, only new tests for the two messages;
- new `scripts/eng3f2_*.py`, `artifacts/eng3f2/` (new), the build directory `tensor_cuda/build-eng3f2/`, `logs/eng3f2_*`,
  and scratch under `/tmp/eng3f2_*` (point `TMPDIR` there).

**Read-only:** everything else here, including `artifacts/eng3/`, `artifacts/eng3f1/` and the earlier build folders;
`tensor_fork_apa_wt_eng3rev2` (the review); every other worktree; production. Never walk `/`, `/mnt` or `/mnt/Shared`
recursively.

## Why and rulings

ENG3-REV2 (receipts `16db930c` in `tensor_fork_apa_wt_eng3rev2`) found that ENG3-F1's allocation safety fixes hold, and
raised three MINOR findings. The lead ruled:
- **ENG3R2-F01, fix:** the byte-overflow error names the shape and the dtype, as F1's R1 asked.
- **ENG3R2-F02, fix:** a device-string refusal names the full offending string. Escape non-printable bytes, including
  NUL, as `\xNN`, so the Python message is never truncated.
- **ENG3R2-F03, closed with no change:** zero-padded device indices behave as the public `tensor_empty` parser does
  (accepted up to nine digits, refused at ten). The hooks stay consistent with the public rule.

Nothing else changes. No numeric or device behaviour changes: only error message text.

## Gates

- **G1, CPU** (`CUDA_VISIBLE_DEVICES` empty, Build G, this round's build):
  - every overflow refusal's message contains the shape and the dtype name, for all six dtypes and both creation hooks;
  - every NUL-containing device string's message contains the full escaped string;
  - every other test in `test_eng3_host.py` passes as on Build F.
- **G2, regression:** the CPU suite per test ID on Build G against Build F (ENG3-F1's `build-eng3f1/source_F` module,
  `642d5ab7…`) is identical. GPU lines are staged for the lead: ENG3-F1's `g3-g1` and `g3-g2` forms, on Build G
  against Build F, with expected exits.
- **G3, mutant:** `f2m01_message_unchanged`. The F01 message is reverted to F1's. It must fail G1's first test.
- **G4, scope:** writes only in the writable set; no git writes; nothing deleted; no GPU.

**Registered acceptance:** ACCEPT when G1, G3 and G4 pass, and G2 passes on CPU and in the lead's GPU lines. No further
review: these are diagnostic fixes, closed by this round.

## Mechanism & scope

- **Measures:** whether the two error messages carry the information F1's rulings asked for, with nothing else changed.
- **Mechanism doc:** `orders/ENG3_F1_REV1_FIXES.md` (R1 and R2), with ENG3-REV2's `findings.json`.
- **Exercises:** the two message sites; Build G and one mutant build; CPU tests; staged GPU lines.
- **Does not cover:** the device-index parser itself (F03, closed); anything else in ENG3; promotion.
- **Operand census:** the byte-bound check; the four hooks' refusal messages.
- **Baseline isolation:** Build F is the reference.
- **Required evidence:** the diff; Build G and mutant receipts with module sha256s; G1 and G2 CPU results; the staged
  lines with expected exits; `REPORT.md`.
- **Forbidden configuration:** any change other than the two message texts and their tests; GPU use; git writes;
  deletes.

## Seat

Codex gpt-6.1-sol, high effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable. No git writes, no subagents, no background waits, every command under a `timeout` (builds under 60 minutes).
**No deletes**: never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call; leave and list.
Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`. Builds use at most `-j4` under `nice -n 19`; stop if
`MemAvailable` falls below 12 GB or `/mnt/ForgeRealm` below 40 GB free. Your final message is the report; also write
`artifacts/eng3f2/REPORT.md`. Prior art: follow `AGENTS.md`.

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, pytest and python3, each run with `--version` first.
- **L02:** staged GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L09:** G2 compares per test ID; logs and timings are never compared.
- **L10:** n/a. Only message text changes; G2 holds outcomes to Build F.
- **L12:** inherited: ENG3-F1's G3 per-ID identity, unchanged.
- **L14:** n/a. No gain gate.
- **L16:** the cheapest pass would change one hook's message only. Excluded: G1 covers all six dtypes, both creation
  hooks and every device-taking hook.
- **L19:** n/a. No array identity is compared; G1 compares message strings.
- **L32:** n/a. "Overflow" is a byte count.
- **L35:** your registrations may fix cases. They may not narrow a gate.
- **L38:** only new tests are added to `test_eng3_host.py`.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng3f2_*`.
- **L41:** the mutant's killing test must execute the mutated line.
- **L45:** the lead records each staged line's exits and log sha256.
- **L47:** the mutant's expected failure comes from this order.
- **L48:** staged lines run as written, from the worktree root, with the environment they state.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L65:** no deletes.
- **L67:** builds go to `tensor_cuda/build-eng3f2/`; stop below 40 GB free.
- **L69:** one reference: test outcomes -> Build F (G2).
- **L79:** staged lines' dry runs stop before the lease. Each staged line states its full environment in its own
  launcher (REV2's reproduction needed four variables its line did not show).
- **L82:** `grep -n 'invalid device' tensor_cuda/src/bindings.cpp` -> 1 hit (`bindings.cpp:2662`, the shared refusal
  helper the hooks call), writable. `grep -n 'byte size overflows' tensor_cuda/src/kernels.cu` -> 1 hit (`kernels.cu:551`,
  the NDArray constructor's check), writable. `numel_of`'s own overflow message (`kernels.cu:86`) is read-only: it
  names the element count, not the byte count, and REV2 did not raise it.
- **L99:** the lead's readings: REV2's `findings.json` lists F01 (`kernels.cu:551`), F02 (`bindings.cpp:2662`) and F03;
  the lead's GPU run confirmed F03 (`lead_eng3rev2_gpu_repro.json`).
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** n/a.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L111:** n/a.

## Done

Your final message gives, verbatim: G1, G3 and G4, each PASS or FAIL; G2's CPU result; the staged GPU lines exactly as
the lead should run them (each with a launcher that sets its environment) and their expected exits; the diff summary;
module sha256s; the deviations; and every file written.
