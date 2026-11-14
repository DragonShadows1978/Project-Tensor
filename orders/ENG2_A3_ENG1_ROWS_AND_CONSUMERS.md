# ENG2-A3: retire test_eng1's ten float64 rows, and census consumer call sites, before ENG2's promotion

**Dispatched 2026-11-14** on branch `eng2`, at the commit carrying this order (after G5's receipts, `e2f166b8`).
`orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md` and its amendments stand unchanged except for what this order adds.

## Why

The owner approved promoting ENG1 and ENG2 on the condition that they pass all the testing, mutation included. ENG1 is
merged to production main (`d7887e5`). ENG2 passed G5 (21 of 21 lines, 15 of 15 mutants killed), but merged as it
stands it would leave ten `test_eng1.py` IDs failing on main by design: the L38 pins in `artifacts/eng2/L38_PINS.json`.
Those rows asked for float64, which the engine does not have, and ran as float32 through the silent fallback that ENG2
removes (`test_eng1.py:180`, `:187`, `:266`). A suite with ten known failures does not meet the condition.

Second, ENG2 turns silent fallbacks into TypeErrors. Every consumer that imports `tensor_cuda` from production loads
that change the moment ENG2 merges. A consumer that passes an unsupported dtype string or a non-string dtype (for
example a NumPy dtype object) works today, by accident, and would raise after the merge. Nobody has counted those
call sites.

## Tasks

1. **R1, retire the float64 rows.** In `tensor_cuda/tests/test_eng1.py`, remove `"float64"` from the dtype lists
   that produce the ten pinned IDs (`E2_DTYPES` at `:180`, its float64 mapping at `:187`, and the parametrize at
   `:266`), with a one-line comment that the engine has no float64 and that `test_eng2.py` tests the refusal. Change
   nothing else in that file.
   - **Mutation coverage:** for each of ENG1's 16 mutants (`artifacts/eng1/mutants/mutants.json`, killer patterns
     with `*`) and each of ENG2's 15, list the registered killer patterns that match any removed ID. Show that every
     mutant still has at least one registered killer that is not removed, and that, in the recorded G5 receipts
     (ENG1 `artifacts/eng1/lead/g5_1114/`, ENG2 `artifacts/eng2/lead/g5_1114/`), one of those remaining killers
     failed on that mutant's line. If any mutant would lose its only killer, stop and report it: that is a RED.
   - **Tool:** `L38_PINS.json` becomes an empty `expected_changes` (the pins no longer exist). Update the regnorm and
     regdtype ID censuses (`artifacts/eng2/tool/ids_*`, `collect_*`) by re-collecting, not by hand-editing. The
     baseline arm runs the edited test file too, so both arms lose the same ten IDs.
2. **R2, consumer census (read-only).** For the three production consumers, census every call into `tensor_cuda`
   that passes a dtype, on tracked files of their main checkouts:
   - `/mnt/ForgeRealm/Project-Voice`, `/mnt/ForgeRealm/GRAPA-Native-LLM`, `/mnt/ForgeRealm/GraftRepository`;
   - list files with `git -C <repo> grep -l` for imports of `tensor_cuda`; exclude `WorkTree/`; parse each with
     Python's `ast` and find calls to the refusal entries in the census (`tc.tensor`, the `_factory` makers,
     `.astype`, `_causal_mask`, and any other entry `artifacts/eng2/L60_dtype_census_candidate.json` lists), whether
     reached as `tc.<name>`, `tensor_cuda.<name>`, or an alias.
   - **Classify each dtype argument:** supported string literal; unsupported string literal; non-string (a NumPy dtype
     object, `None`, a type); or not a literal (a name or expression: trace it within the file if you can, else mark
     it "unresolved").
   - **Report** every call site ENG2 would newly refuse, and every unresolved one, with `file:line`. Mark each as
     live code (package, engine, scripts) or receipts (under `artifacts/`).
   - This is reading only. Never write into those repositories.
3. **R3, re-run and re-seal.**
   - The CPU suite: `test_eng1.py` and `test_eng2.py` on CPU, candidate and baseline, as A-G2 ran them. The candidate
     passes every CPU ID, and the baseline fails exactly the registered F1 IDs plus the non-string IDs.
   - MF1 and MF2 on CPU, scored through `evaluate()`.
   - Re-seal the staging. The device-mode dry run passes on every root label (governor 78), and on every dependent
     label (its dependency refusal, then its argv to governor 78). The audit route passes.

## Gates

- **A3-G1:** R1 done; the mutation-coverage table shows every one of the 31 mutants keeps a registered, non-removed
  killer that failed on its G5 line; `L38_PINS.json` is empty; the censuses are re-collected.
- **A3-G2:** R2's census table, with its counts per repository and class, and the full list of newly refused and
  unresolved sites.
- **A3-G3:** R3's suite, mutants, seal, dry run and audit pass.
- **Scope:** sha256 maps show that only these changed: `test_eng1.py` (the three dtype lists and one comment), the
  tool's pin and census files, the seal and staging files that re-sealing rewrites, and new files under
  `artifacts/eng2/a3/`.
- **Acceptance:** ACCEPT when A3-G1 and A3-G3 pass. A3-G2 is a census, not a pass/fail gate: the lead acts on what it
  finds before the merge.

## Mechanism & scope

- **Measures:** whether retiring ten mislabelled test rows keeps every mutant killed and every suite green, and which
  consumer call sites ENG2's refusal would change.
- **Mechanism doc:** `orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md`.
- **Exercises:** `test_eng1.py`'s dtype lists; the GPU-line tool's pins and censuses; the CPU suite; MF1/MF2 on CPU;
  staging; read-only AST parsing of three consumer repositories.
- **Does not cover:**
  - GPU execution (the lead reruns the affected G5 lines afterwards);
  - any product code (`__init__.py`, `nn.py`, `functional.py`, C++);
  - fixing consumers (the lead mails their leads);
  - promotion.
- **Operand census:** the 10 pinned IDs; ENG1's 16 and ENG2's 15 mutants with their killer patterns; the three
  consumer repositories' tracked Python files that import `tensor_cuda`.
- **Baseline isolation:** `bc02d3a1` stays the baseline arm. ENG1's receipts and G5 receipts are read-only.
- **Required evidence:** the diff; the mutation-coverage table; the census table and lists; A3-G1 to A3-G3 receipts;
  the new seal.
- **Forbidden configuration:** changing product code, tolerances, expected exits or bounds; editing receipts in place;
  GPU execution; git writes; deletes; writes outside the target.

## Where to work

**Seat:** Claude Opus 5.5, high effort (Agent tool). ENG2 was written by Claude; a narrow Codex review follows.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2`
(branch `eng2`). Edits and CPU runs are AUTHORIZED only in:
- `tensor_cuda/tests/test_eng1.py`: the three dtype lists and one comment, as R1 says;
- `artifacts/eng2/L38_PINS.json`, and `artifacts/eng2/tool/`: the census, collect, seal and staging files that
  re-collecting and re-sealing rewrite, and `gpu_runs/dry_3`, `gpu_runs/audit_3` (new);
- everything under `artifacts/eng2/a3/` (new);
- scratch under `/tmp/eng2_a3_*`. Point `TMPDIR` there.

**Read-only:** every other file in this worktree; every other worktree; the production repository; and the three
consumer repositories (reading only, through `git grep`, `git show` and Python's `ast` on file contents).

**Rules:**
- No git writes (read-only `git grep`/`git show`/`git ls-files` on the consumer repositories is allowed). No subagents.
  No background waits. Every command stays under 10 minutes, with `timeout`.
- **No GPU.** Dry runs only, with `FORGE_GPU_DRY_RUN=1`.
- **No deletes.** Never run `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call.
- **CPU and memory:** at most 2 CPU-busy processes at a time (the machine is loaded), `nice -n 10`, one thread per
  test process. Any process over 1 GB runs in its own scope (`systemd-run --user --scope -p MemoryMax=4G
  -p OOMPolicy=continue -p CPUWeight=50 ...`) and starts only when `free -g` shows at least 12 GB available.
- Never kill a process you did not start. Never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the named
  repositories by path, through git.
- **Your final message is the report.** Do not name any file `REPORT.md`.

## Registration checklist (order-lint)

- **L01:** the instruments are python3 with NumPy and pytest (versions as in `artifacts/eng2/preflight_L01.txt`), the
  copied tool, and read-only git. Run `--version` and one no-op collection first.
- **L09:** the suite comparisons are by normalized test ID and outcome, exactly; logs and timing are excluded.
- **L10:** n/a. No adaptive stage is involved.
- **L16:** the cheapest passing fix deletes the rows and stops. Excluded: the mutation-coverage table must show a
  remaining killer that failed on every mutant's G5 line, and the censuses must be re-collected from a run.
- **L30:** the "killer failed on its G5 line" check reads the recorded G5 receipts and their manifests; quote each
  manifest's sha256 and confirm it against the file.
- **L38:** this order edits an inherited test file on purpose. The ten removed IDs are listed by name; no other ID in
  `test_eng1.py` changes, shown by an ID diff before and after.
- **L41:** MF1 and MF2 are shown executing their hunks on CPU and scored through `evaluate()`.
- **L48:** the staged lines are re-run as written by the dry run; the only addition is `FORGE_GPU_DRY_RUN=1`.
- **L56:** (a) A slip in running your own tools goes under Deviations: re-run, and continue. (b) Before acceptance, a
  failed gate is a harness defect: at most 3 repairs per gate, ledgered.
- **L60:** R2 is the consumer census: every call site, classified, with the unresolved ones listed, not dropped.
- **L64:** the suite and tool read this worktree; R2 reads the three consumer repositories through git, which this
  order permits. The code under test may manage its own files inside the writable set.
- **L65:** no deletes, as stated.
- **L03:** "mutation coverage" here is per mutant (each keeps a killer), not a share of all IDs; there is no coverage
  bar and no coupled budget.
- **L04:** n/a. "Fallback" means the silent dtype fallback ENG2 removed; there is no builder with fallback rungs.
- **L06:** n/a for gates. The one fixed count (12 GB available before a job over 1 GB starts) is the machine's
  standing memory rule, not a threshold on the data.
- **L90:** roles are counted by the census itself: product (package or engine code), script, test, and receipt (under
  `artifacts/`), per repository. Treatment is the same for every role: report the site; the lead decides per site.
  The census states what its file listing skips (`WorkTree/`, untracked files).
- **L82:** `grep -n 'float64' tensor_cuda/tests/test_eng1.py` locates the three lists. Writable: those lines and one
  comment. Read-only: every other line.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- A failed gate is a result.
- **Prior Art Directive:** annotate the census method (Python `ast` call-site analysis), or write "no prior art known
  to me".

## Done

Your final message gives, verbatim:
- the diff;
- the mutation-coverage table;
- the consumer census table and the full lists of newly refused and unresolved call sites;
- A3-G1 to A3-G3 with their receipts;
- the new seal's sha256;
- every deviation;
- a "Prior art" heading.
