# ENG2-REV3: narrow cross-model review of ENG2-A3 (ten test rows retired; consumer census) before ENG2's promotion

YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng2/rev3/` (branch `eng2`,
at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/eng2_rev3_*` are AUTHORIZED.
Point `TMPDIR` at the scratch directory for every command.

**Read-only:**
- everything else in this worktree;
- ENG1's worktree, `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1`, for its G5
  run root;
- the three consumer repositories named below, through `git -C <repo> grep/show/ls-files` and Python's `ast` on file
  contents only;
- every other worktree and the production repository.

Read-only git is allowed. **Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter
`/mnt/Shared/ForgeRealm_Link`. Search inside the named paths only.

## What you review

A Claude seat (Opus 5.5) wrote ENG2-A3 under `orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md`. Receipts are commit
`10738fe3`, under `artifacts/eng2/a3/`. The seat's final message is `artifacts/eng2/a3/SEAT_FINAL_MESSAGE.md`. You
(Codex) reviewed ENG2 twice before (REV1, REV2). This round covers A3 only.

## Questions (answer each, with receipts)

1. **The edit.** Does `git diff e2f166b8 10738fe3 -- tensor_cuda` touch only `tests/test_eng1.py`, at the three dtype
   lists and one comment? Do both arms lose exactly the ten L38-pinned IDs and no other? Re-collect `test_eng1.py`'s
   IDs yourself at `e2f166b8` and at `10738fe3` (`git show` into scratch) and diff them.
2. **Mutation coverage.** Independently of the seat's script, for each of the 31 mutants (ENG1:
   `artifacts/eng1/mutants/mutants.json`; ENG2: its mutant registry under `artifacts/eng2/`), check that no registered
   killer pattern matches only removed IDs, and that a remaining killer failed on that mutant's G5 line. Read the G5
   receipts from the run roots (ENG1: `…/tensor_fork_apa_wt_eng1/artifacts/eng1/tool/gpu_runs/g5_1114`; ENG2:
   `artifacts/eng2/tool/gpu_runs/g5_1114`) and verify each file you use against its run's `MANIFEST.sha256`.
3. **The consumer census.** Is "0 newly refused call sites" right?
   - With your own method, find calls into `tensor_cuda`'s changed entries (`tc.tensor`, `zeros`, `ones`, `randn`,
     `rand`, `nn.parameter`, `functional._causal_mask`) that pass an unsupported dtype string, or a non-string, in
     tracked files of `/mnt/ForgeRealm/Project-Voice`, `/mnt/ForgeRealm/GRAPA-Native-LLM` and
     `/mnt/ForgeRealm/GraftRepository`.
   - Spot-check at least 8 of the 48 "unresolved, changed entry" sites in `artifacts/eng2/a3/CONSUMER_CENSUS.json`,
     covering every repository, and say whether the seat's reading holds.
   - Check the claim that the "unchanged" entries (`.astype`, `tc.empty`, the affine helpers) behave identically on
     both arms, against `artifacts/eng2/L60_dtype_census_baseline.json` and `_candidate.json`.
4. **Seal and staging.** Does `artifacts/eng2/tool/staging_seal.json` (sha256 `b424bdcf…`) verify against the arms,
   and does the dry-run receipt `gpu_runs/dry_4` show 21 of 21 labels as registered? Re-run the seal verification.

## Method

- **CPU only:** `CUDA_VISIBLE_DEVICES` empty, `nice -n 10`, one thread, at most 2 CPU-busy processes at a time (the
  machine is loaded), every command under 10 minutes with `timeout`. Any process over 1 GB runs in its own scope
  (`systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue -p CPUWeight=50 ...`) and starts only when
  `free -g` shows at least 12 GB available; if that command is unavailable, keep under 1 GB.
- **A finding** has an ID (`E2R3-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction or
  a precise argument, and a suggested fix. A question checked and found sound is reported as checked, with what you
  ran.
- **NITs follow the 11-03 rule:** fixed once and closed, with no further review round.
- Write `artifacts/eng2/rev3/REPORT.md` and `artifacts/eng2/rev3/findings.json` early.

## Mechanism & scope

- **Measures:** whether A3's test edit keeps every mutant killed and every suite consistent, and whether ENG2's
  refusal reaches any consumer call site.
- **Mechanism doc:** `orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md`.
- **Exercises:** the A3 diff; ID collection on both commits; the 31 mutants' killer sets against G5 receipts; an
  independent consumer call-site search; the seal.
- **Does not cover:**
  - GPU execution (the lead reruns G5 under the new seal);
  - ENG2's product code (reviewed in REV1 and REV2; unchanged);
  - the C++ dtype defects;
  - promotion.
- **Operand census:** the 10 removed IDs; 31 mutants; the three consumer repositories' tracked Python files.
- **Baseline isolation:** `e2f166b8` (G5-reviewed) is the before; `10738fe3` is the after.
- **Required evidence:** `REPORT.md`, `findings.json`, each check's commands and outputs.
- **Forbidden configuration:** edits outside the writable target; GPU use; git writes; deletes; writes into the
  consumer repositories.

## Seat

Codex gpt-6.1-sol, reasoning xhigh. A3 was written by Claude, so this is the cross-model review.

**Rules:**
- No git writes (read-only git is allowed). No subagents. No background waits. Run every check in the foreground.
- **No deletes.** You are a Codex seat, and the Claude delete guard does not cover you: never run `rm`, `rmdir`,
  `unlink`, `git clean`, `find -delete` or an inline delete call.
- Never kill a process you did not start. Other seats and lead jobs are running on this machine.

## Registration checklist (order-lint)

- **L01:** the instruments are python3 with NumPy and pytest (versions in `artifacts/eng2/preflight_L01.txt`), the
  copied tool, and read-only git. Run each with `--version` first.
- **L06:** n/a for gates. The 12 GB figure is the machine's standing memory rule, and "at least 8" spot checks is a
  review sample size, not a data threshold.
- **L09:** ID comparisons are exact, by normalized test ID; logs and timing are excluded.
- **L16:** the cheapest passing review is "checked, no findings" without running anything. Excluded: questions 1, 2
  and 3 each require an independent run (your own ID collection, your own killer check, your own call-site search),
  with commands and outputs.
- **L30:** every G5 receipt you read is verified against its run's manifest first.
- **L41:** the review runs no registered gate; it reads recorded G5 kills and does not re-execute mutants.
- **L56:** (a) A slip in running your own tools goes under Deviations: re-run, and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each ledgered. A review has no registered data gate.
- **L60:** question 3 is the consumer boundary census: every changed entry, every repository, unresolved sites
  listed, not dropped.
- **L64:** reads are the paths named above, all permitted. The code you write may manage its own files inside the
  writable set.
- **L65:** no deletes, as stated.
- **L82:** n/a. The writable target is one new folder; no product code is edited.
- **L03:** n/a. "Mutation coverage" here is per mutant (each keeps a killer), not a share; there is no coverage bar.
- **L21:** the lead commits `REPORT.md` and `findings.json` by name (force-adding ignored JSON); write nothing else
  that a gate depends on.
- **L42:** the run is reading and small CPU collections: two ID collections of one test file, one killer check over 31
  mutants, and one AST pass over three repositories (A3's pass took minutes). No sizing beyond the 10-minute command
  cap is needed.
- **L67:** no lines with per-line receipts are executed; keep scratch under `/tmp/eng2_rev3_*` and write at most
  200 MB into the target.
- **L90:** roles (product, script, test, receipt) are counted per repository in your census; treatment is the same for
  every role: report the site.

## Done

Your final message gives, verbatim:
- every finding;
- every question's answer, with what you ran;
- `findings.json`'s sha256;
- a "Prior art" heading for any technique you cite or propose.
