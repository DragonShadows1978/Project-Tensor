# APA-P1-REV2: blind cross-model red team of the packed default (P1-D) and the REV1 fixes (P1-F1), before promotion

YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1rev2/artifacts/apa_p1_rev2/`
(branch `apa-p1rev2`, at the commit carrying this order). Edits there, CPU runs, and scratch under `/tmp/p1_rev2_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory for every command.

**Read-only:**
- everything else in this worktree;
- every other worktree, including `tensor_fork_apa_wt_p1d` (it holds the builds and receipts named below);
- the production repository.

Read-only git is allowed. **Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively. Search inside this
worktree, or inside a named folder of another worktree, by path.

## What you review

The engine's selective-precision attention (APA-SP) keeps every key. It scores all keys from a compressed copy and
recomputes the top-scoring ones at full precision. P1 ported the lab's packed form of that compressed copy (4-bit key
codes, the EB2b/EB3/EB4 kernels, the SPQ1/SPQ4/SPQ6 entries) onto production's engine. A blind Claude review (P1-REV1)
found twelve defects, P1R-F01 to P1R-F12. Then two rounds followed:
- **P1-D** (`f01201cf`) made packed storage the default representation of SP keys. It added `apa_packed.py` (the
  `PackedSpKeys` class and the packed margin table), a `storage` argument on `quantize_sp2_keys`, a storage policy and
  a CMake default `TC_APA_PACKED=ON`.
- **P1-F1** (`710cc0d4`) fixed the twelve REV1 findings.

**The diff:** `git diff e2fb85a1 710cc0d4 -- tensor_cuda`, 17 files, +1,927/−92. `e2fb85a1` is the P1 port merged
with production main `2478a4d`, and is the base. Everything in this diff becomes production's SP path when it is
promoted.

The P1 port itself (`git diff 0e1dea3 daef0291 -- tensor_cuda`) was reviewed by REV1. Read it wherever the new code
calls into it, but your findings are about the diff above and how it interacts with the port.

**This review is blind.**
1. Form your own view from the code first.
2. Write `findings.json` (first version).
3. Only then read the seats' reports, the lead's rulings and `orders/APA_P1_REV2_KNOWN_ITEMS.md`, to check their
   claims. A finding already on the lead's list still counts if you find it independently; say so.

## Builds and dumps (read-only, in `tensor_fork_apa_wt_p1d`)

| Name | Path under `tensor_fork_apa_wt_p1d/artifacts/` | sha256 |
|---|---|---|
| Build B (P1-D, before F1) | `apa_p1d/build_B/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` | `8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e` |
| Build F (final, ships) | `apa_p1f1/build_F_r2/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` | `5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243` |
| Build FH (test hooks on) | `apa_p1f1/build_FH/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` | `19b4a1783e981302522e79654c278e0e3c462606c2b3687c11876395756be2b6` |
| SASS of B | `apa_p1f1/sass/B.sass` (236 MB; read it streaming) | in `SASS_COMPARISON_F_r2.json` |
| SASS of F | `apa_p1f1/sass/F_r2.sass` (236 MB; read it streaming) | in `SASS_COMPARISON_F_r2.json` |

Check each sha before you use the file.

## Questions (answer each, with receipts)

1. **The default switch (P1-D).**
   - Trace every public route into SP with `storage=None`: which representation is built, which kernel runs, and which
     margin table applies.
   - Is the packed margin table's pin (`apa_packed.py`, the registered sha check) enforced before any use?
   - What does a caller who used the dense API before P1-D get now (arguments, return types, errors)?
   - The storage policy (`TC_APA_SP_STORAGE`), the refusals (for example prefill on packed keys), and `PackedSpKeys`'
     identity, capacity and append semantics: are they sound?
   - Is there any input on which the packed default skips a key that the dense rule, applied to the same bulk scores
     and margins, would refine?
2. **Boundary census.** For every entry the diff adds or changes, list the preconditions its pointer and index
   arithmetic relies on: device, dtype, rank, extents, contiguity, alignment for vector loads, visible rows versus
   capacity, the packed tuple's shapes, GQA ratio, empty shapes, and 32-bit overflow. Classify each as guarded (where),
   impossible (why) or unguarded (what happens).
   - Is the alignment census (`apa_p1f1/ALIGNMENT_CENSUS.json`) complete? Check it against every vector-width load in
     `tensor_cuda/src/apa_spo/` (`uint4`, `uint2`, `float4`, `float2`, `int4`, `int2`, `reinterpret_cast` to a vector
     type).
   - Does every refusal happen before the first launch?
3. **Non-finite values.** Do NaN and ±Inf in q, k or v, and zero-norm keys, propagate on every entry the way the dense
   SP route's do? Name every entry and case where the packed path returns a finite row while dense returns a
   non-finite one, or the reverse.
4. **Tickets, hooks and streams.**
   - Are the R5 ticket reset and the per-device ticket ownership correct after a normal call, a failed launch and a
     timeout?
   - Are all test hooks (`TC_APA_P1_TEST_HOOKS`, the forced-timeout mode, the ticket accessors) absent from Build F?
     Show this from its exported and device symbols, not only from the source.
   - What happens on a non-default or per-thread default stream?
5. **Linkage and loading.**
   - F1 reaches its new EB2b signature through global-namespace bridges (`apa_p1f1_eb2b_run`,
     `apa_p1f1_merge_tickets`), declared at block scope in `bindings.cpp`. Is that sound (ODR, linkage, name clashes)?
   - What remains declared in `ops.h` without a definition?
   - `__init__.py` probes the second initializer `PyInit__tensor_cuda_spo2` with `RTLD_NOLOAD` and installs stubs on a
     non-packed build. Is the loader sound on a packed build, a non-packed build and a failed load?
6. **Code generation.**
   - Verify from the two dumps which kernels differ between B and F. Map only nvcc's per-build anonymous-namespace
     suffix (`_cu_<8 hex>` with an optional `_<n>`).
   - F keeps six `tc::spq6::bulk_mma_tq` instantiations that F1's order expected to be removed. Can any path in the
     default build reach them?
7. **Determinism and numerics.** Is the output bitwise run-to-run on one device, given the merges and atomics? Did F1's
   guard change (`den > 0` to `den != 0`) change any finite result?
8. **Tests and evidence.** Do `test_apa_p1d_policy.py`, `test_apa_p1f1_cpu.py` and F1's GPU lines (receipts under
   `apa_p1f1/gpu/`) exercise the risky cases from questions 1 to 7? Name each gap.
9. **REV1 fix completeness** (after your blind pass). For each of P1R-F01 to P1R-F12
   (`tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_rev1/findings.json`), is the fix complete on every site the finding
   names, and on any site it missed?
10. **Anything else** that production inherits from this diff.

## Method

- **Blind first:** the code, then the claims.
- **CPU only.** Your sandbox has no GPU, and the engine cannot build a tensor without one. CPU checks are reading,
  arithmetic, symbol and SASS inspection (`nm`, `cuobjdump`, `c++filt`), constructed host-side cases, and the CPU test
  files. Run them with `CUDA_VISIBLE_DEVICES` empty, under `nice -n 19`, single-threaded, and with
  `PYTHONDONTWRITEBYTECODE=1`. Give every command a `timeout` of at most 1,200 s.
  - To run the CPU test files against Build F, put `tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/arms/F` first on
    `PYTHONPATH`. Pass `-p no:cacheprovider` so nothing is written into another tree.
- **GPU reproductions are staged, not run.** For a suspected fault that needs the GPU, write at most 8 scripts under
  `artifacts/apa_p1_rev2/repro/`, named `r<nn>_<short>.py`. Each script:
  - checks the module sha before importing and refuses on a mismatch;
  - loads the module by path, the way `scripts/apa_p1f1_gpu.py` does;
  - prints one JSON line with its verdict;
  - states, in its docstring, the outcome that confirms the finding and the outcome that refutes it.

  Compile each with `python3 -m py_compile` (with caches redirected to TMPDIR). After your round, the lead runs each
  once through `forge-gpu-run --min 30 --max 600 --label p1rev2-<nn> -- timeout 560s python3 <script>` and files the
  results. A finding must stand on a precise argument from the code; a staged reproduction supports it and does not
  replace it.
- **A finding** has:
  - an ID (`P1R2-Fnn`);
  - a severity: BLOCKER, MAJOR, MINOR or NIT;
  - the file and line;
  - the argument or the staged reproduction;
  - a suggested fix.

  A question checked and found sound is reported as checked, with what you ran or read.
- **NITs follow the 11-03 rule:** they are fixed once and closed without another review round.
- Write `artifacts/apa_p1_rev2/findings.json` early and keep it current. Also write `artifacts/apa_p1_rev2/REPORT.md`.
- **Ceiling:** 8 hours of wall time. If you cannot finish, report what was checked and what was not.

## Mechanism & scope

- **Measures:** whether P1-D's packed default and P1-F1's fixes are correct and safe to ship as production's SP path:
  - the default switch;
  - preconditions and alignment;
  - non-finite handling;
  - tickets, hooks and streams;
  - linkage and loading;
  - code generation;
  - determinism;
  - test coverage;
  - REV1 fix completeness.
- **Mechanism doc:**
  - `orders/APA_P1D_PACKED_DEFAULT.md` and `orders/APA_P1F1_REV1_FIXES.md` (in this worktree);
  - `artifacts/apa_p1d/REPORT.md`, and `REPORT.md` under `tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/`.
- **Exercises:** reading the 17-file diff and the port code it calls; CPU tests; symbol and SASS inspection; at most 8
  staged GPU reproductions.
- **Does not cover:**
  - model quality (P1-Q, separately);
  - speed;
  - memory residency (MEM1);
  - DS1's pitched cache;
  - multi-stream support, which is out of contract (question 4 asks only what happens);
  - the P1 port's own diff, except where the new code calls into it.
- **Operand census:**
  - every hunk of the 17-file diff;
  - every entry it adds or changes;
  - every vector-width load in `tensor_cuda/src/apa_spo/`;
  - both ticket buffers;
  - the three modules above.
- **Baseline isolation:** `e2fb85a1` is the baseline. Every entry that existed there must behave as before, except
  where F1's rulings changed it (`orders/APA_P1F1_REV1_FIXES.md`, R1 to R12).
- **Required evidence:**
  - `findings.json` and `REPORT.md`;
  - each check's command and output;
  - each staged reproduction, with its compile check.
- **Forbidden configuration:**
  - edits outside the writable target;
  - builds;
  - any GPU use;
  - git writes;
  - deleting anything yourself.

## Seat

Codex gpt-6.1-sol, xhigh effort. Claude seats wrote P1-D and P1-F1, and REV1 was a Claude review, so this is the
cross-model review.

**Rules:**
- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- No git writes (read-only git is allowed). No subagents. No background waits. Run everything synchronously, each
  command under an explicit timeout.
- **No deletes.** Never run `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Claude's
  delete guard does not cover your seat, so this rule is yours to keep. Anything that should go is listed in your report
  for the lead.
- Never kill a process you did not start.
- **Prior art:** follow the directive in `AGENTS.md`. Every technique you cite or propose carries its source, or
  "no prior art known to me".

## Registration checklist (order-lint)

- **L01:** the instruments are `git diff` and `git show` (read-only), python3 with NumPy, pytest, `nm`, `cuobjdump` and
  `c++filt`. Run each with `--version` (or `--help`) first, and record the outputs in `artifacts/apa_p1_rev2/INSTRUMENTS.log`.
- **L03:** n/a. No share or coverage bar is gated. Question 8 asks for gaps, and does not gate a share.
- **L09:** this order registers no identity gate. When you check a bitwise claim from receipts, use `np.array_equal`
  on same-width unsigned views, after dtype and shape checks, and exclude logs and timing.
- **L10:** n/a. No identity is held or gated here. The quantizers in scope are reviewed, not re-gated. F1's G1 held
  finite identity to Build B on the deterministic paths. The review may report a quantizer finding but registers no
  "unchanged stage" gate.
- **L44:** every count you report (kernels that differ, census rows, symbols, test IDs) is recomputed by you from the
  raw source: the dumps, the code, `nm` output or the pytest log. Each count is printed beside the receipt's stored
  count. A receipt summary is never taken on trust. You build no validator, so no refusal test is owed.
- **L11:** n/a. No program-unchanged claim is gated. Question 6 asks for the SASS difference as a finding, not a gate.
- **L12:** n/a. The review inherits no gate and changes no predicate.
- **L14:** n/a. There is no gain gate.
- **L16:** the cheapest passing review is "checked, no findings" without running anything. That is excluded:
  - each answer names what was run, or the `file:line` read;
  - question 2 requires the full precondition table;
  - question 6 requires your own count from the dumps.
- **L18:** P1-D made packed storage the shipped default. That is the owner's requirement since 11-02 (APA memory back
  to the two-pass level), as recorded in the lead's ledger. The review judges whether the default is safe, not whether
  it should be the default.
- **L19:** any array you compare is checked nonzero and non-constant first, at full width.
- **L25:** you run no bitwise comparison on a device. Receipts you check were made on one device, from one build per
  side; say which.
- **L32:** the bounded buffers in scope are the ticket buffers, the workspace arenas and `PackedSpKeys`' capacity.
  Questions 1, 2 and 4 ask what happens at growth and capacity. No fill gate is registered. Each finding names its
  buffer.
- **L45:** you run no GPU line. Each staged reproduction is run by the lead, and recorded with `forge-gpu-run`'s exit,
  the inner `timeout` exit and the log's sha256. Exit 124 is a timeout, never a pass.
- **L48:** staged reproductions run from this worktree's root. They pass `py_compile` here first. The lead runs them
  unchanged.
- **L55:** for question 1's last item and question 3, show the semantics from the code on one constructed instance per
  criterion (delta, carry, tie, causal, sink, NaN, +Inf, −Inf, zero norm) before relying on any receipt.
- **L56:**
  - (a) A slip in running your own tools goes under Deviations: re-run and continue.
  - (b) An exception in your own scripts is repaired at most 3 times, each recorded.

  A review has no registered data gate.
- **L58:** every command runs with `CUDA_VISIBLE_DEVICES` empty. You use no GPU.
- **L60:** question 2 is the boundary census. For each refusal, state the baseline (`e2fb85a1`) behaviour where the
  entry existed there.
- **L65:** no deletes, as stated.
- **L79:** n/a. You stage no line that the lead rehearses on the CPU. Staged reproductions are checked with
  `py_compile` only.
- **L82:** n/a. The writable target is one new folder. The review reads, and does not edit, product code.
- **L99:** the review's premises are the lead's own readings, run before sealing:
  - `git diff --stat e2fb85a1 710cc0d4 -- tensor_cuda` gave 17 files, +1,927/−92;
  - `git log --oneline e2fb85a1..710cc0d4` listed `710cc0d4`, `e504c165`, `f01201cf` and `3524edca`;
  - the lead recomputed the three module shas above;
  - REV1's `findings.json` lists P1R-F01 to P1R-F12.

  Re-run the first two first. If either differs, report it before reviewing.
- **L103:** the modules are named by path and sha256 in the table above. Check the sha before each use, and refuse on a
  mismatch.

## Done

Your final message gives, verbatim:
- every finding;
- every question's answer, with what you ran or read;
- each staged reproduction's path, sha256 and stated confirm/refute outcomes;
- `findings.json`'s sha256;
- what you would delete, if anything (the lead does it);
- a "Prior art" heading.
