# ENG2-A1: fix ENG2-REV1's five findings (CPU round; restage the GPU lines)

**Dispatched 2026-11-14** on branch `eng2`, at the commit carrying this order (after REV1's receipts, `9452ef61`).
`orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md` stands unchanged except for the fixes below. Read it first.

## The findings to fix (`artifacts/eng2/rev1/findings.json`, sha256 `1938c6e0…1a67`; `REPORT.md`)

1. **E2R-F03 (BLOCKER).** `artifacts/eng2/tool/evidence.py:229`. `matches()` strips the test-file prefix from the
   observed ID, but compares it with registered patterns that keep the prefix. No registered killer matches itself,
   for any of the 15 mutants.
   - **Fix:** normalize both sides the same way, keeping literal parameter brackets.
   - **Show:** every registered killer matches itself, and a real CPU MF1/MF2 receipt goes through `evaluate()` and
     scores as killed.
2. **E2R-F04 (BLOCKER).** `artifacts/eng2/tool/score_gpu.py:46-47` keys the `f1bytes` and `p302bits` rows by the raw
   `request.node.nodeid`, which includes the arm path (`test_eng2.py:57`).
   - **Fix:** compare by `evidence.norm_id` on both sides. Refuse duplicate normalized IDs. Check the expected census
     of 35 and 20 rows. Leave the hashes unchanged.
3. **E2R-F01 (MAJOR).** `tensor_cuda/tensor_cuda/__init__.py:52`. A non-string equal to a supported name (for example
   `collections.UserString("float32")`) passes the set and dict membership tests. It is then converted, and
   `randn`/`rand` advance NumPy's RNG.
   - **Fix:** in `tensor` and in `make`, check `isinstance(dtype, str)` before any membership test, raising the
     registered TypeError.
   - **Tests:** add non-string tests: `UserString`, an unhashable object, `None`, and a NumPy dtype object.
4. **E2R-F02 (MAJOR).** `artifacts/eng2/tool/evidence.py:338`. The L38 pin branch accepts any failure whose longrepr
   contains the registered text.
   - **Fix:** require the call-phase terminal exception to be TypeError, with the registered exact message, from
     structured exception data (JUnit or report fields, not a substring of the longrepr). Reject every other
     exception class, and any extra failure.
   - **Tests:** add REV1's constructed negatives (`wrapped_other_reason`, `wrong_type_exact_text`).
5. **E2R-F05 (MINOR).** The census (`artifacts/eng2/dtype_census.py:45`) omits:
   - the private `functional._causal_mask`, which still falls back silently to float32;
   - three `quantization.affine` `scale_dtype`/`out_dtype` entries.

   **Fix:**
   - complete the census with these entries and their baseline outcomes;
   - guard `_causal_mask` against unsupported and non-string dtype arguments, before it builds a tensor;
   - keep the affine helpers' valid NumPy dtype support unchanged, and record their behaviour.

**NITs:** none were raised. Per the owner's 11-03 rule, there is one fix round with a narrow re-review, and no
further round unless that re-review finds a real defect.

## Gates

- **A-G1, REV1's probes.** Re-run REV1's own scripts, unchanged, from `artifacts/eng2/rev1/`, into fresh output paths
  under `artifacts/eng2/a1/`:
  - `probe_edges.py`: the UserString cases are refused; the pin cases are refused; every killer matches itself;
  - `cross_arm_review.py`: identical hashes give "IDs equal True", and no differing rows;
  - `census_review.py`: the new rows are present, and `_causal_mask` refuses.

  If a script cannot run unchanged against the fixed tool, for example because it imports a renamed symbol, copy it
  to `artifacts/eng2/a1/`, change only the import, and ledger the diff.
- **A-G2, the CPU suite.**
  - The candidate passes all of `test_eng2.py`'s CPU IDs, including the new ones.
  - The baseline fails exactly the registered F1 IDs plus the new non-string IDs.
  - Re-run MF1 and MF2 on CPU: each is killed, and the fixed `evaluate()` scores it as killed.
- **A-G3, scope and staging.**
  - Sha256 maps show that only these changed: the product lines named above (`__init__.py`'s `tensor`/`make`, and
    `functional._causal_mask`), the tool files named above, `test_eng2.py` (additions only), and new files under
    `artifacts/eng2/a1/`.
  - Re-seal the staging. The device-mode dry run passes on every root label (governor 78), and on every dependent
    label (its dependency refusal, and its argv to governor 78).
  - The audit route passes.
- **Acceptance:** ACCEPT when A-G1 to A-G3 pass. Then a narrow Codex re-review of these fixes only follows, and then
  the lead's G5.

## Mechanism & scope

- **Measures:** whether the five REV1 findings are fixed without changing anything else.
- **Mechanism doc:**
  - `orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md`;
  - `artifacts/eng2/rev1/REPORT.md`.
- **Exercises:**
  - the refusal entries;
  - `_causal_mask`;
  - the tool's matcher, pin evaluation and cross-arm scoring;
  - REV1's probes;
  - the CPU suite;
  - staging.
- **Does not cover:**
  - the N1 norm code (REV1 found it sound; unchanged);
  - the C++ dtype defects;
  - GPU execution;
  - promotion.
- **Operand census:**
  - the 5 findings;
  - the 15 mutants' killer sets;
  - the 35 `f1bytes` and 20 `p302bits` rows;
  - the census entries.
- **Baseline isolation:** `bc02d3a1` stays the baseline arm. REV1's receipts are read-only.
- **Required evidence:**
  - the diff;
  - A-G1 to A-G3 receipts;
  - the new seal;
  - the dry-run and audit receipts.
- **Forbidden configuration:**
  - changing tolerances, N1 code, expected exits or bounds;
  - editing REV1's receipts or scripts in place;
  - GPU execution;
  - git writes;
  - deleting anything.

## Where to work

**Seat:** Claude Opus 5.5, high effort (Agent tool).

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2`
(branch `eng2`). Edits and CPU runs are AUTHORIZED only in:
- `tensor_cuda/tensor_cuda/__init__.py`: `tensor`, `make` and `_np_dtype`;
- `tensor_cuda/tensor_cuda/functional.py`: `_causal_mask` only;
- `tensor_cuda/tests/test_eng2.py`: additions only;
- `artifacts/eng2/tool/`: `evidence.py` (the matcher and the pin branch), `score_gpu.py` (the cross-arm keys), the
  seal and staging files that re-sealing rewrites, and `gpu_runs/dry_2`, `gpu_runs/audit_2` (new);
- `artifacts/eng2/dtype_census.py` (completion), and everything under `artifacts/eng2/a1/` (new);
- scratch under `/tmp/eng2_a1_*`. Point `TMPDIR` there.

**Read-only:**
- `artifacts/eng2/rev1/`;
- every other file;
- every other worktree.

**Rules:**
- No git, not even read-only. No subagents. No background waits. Every command stays under 10 minutes, with
  `timeout`.
- **No GPU.** Dry runs only, with `FORGE_GPU_DRY_RUN=1`.
- **No deletes.** Never run `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call.
- Never kill a process you did not start. Never walk `/`, `/mnt` or `/mnt/Shared` recursively.
- **Your final message is the report.**

## Registration checklist (order-lint)

- **L01:** the instruments are python3 with NumPy and pytest (the versions in `artifacts/eng2/` L01), REV1's scripts,
  and the copied tool. Run `--version` and one no-op collection first.
- **L10:** n/a. No adaptive stage is involved. The identities are on deterministic CPU paths and recorded hashes.
- **L30:** the cross-arm "hashes unchanged" clause is bound to the recorded rows in REV1's `cross_arm_review.json`
  and the A2-era receipts. Re-derive them by re-running `cross_arm_review.py`. No stored hash is trusted without
  that reproduction.
- **L09:** A-G1's cross-arm check compares rows by normalized ID and hash, exactly. Logs and timing are excluded.
- **L16:** the cheapest passing fix makes the matcher match everything. Excluded: REV1's pin negatives must be refused,
  a constructed survivor (a killer set with no failing ID) must score as a survivor, and duplicate normalized IDs must
  be refused.
- **L38:** the only inherited pins are the 10 `test_eng1.py` float64 IDs, now checked on structured exception data.
  `test_eng2.py` gains IDs but no existing ID changes.
- **L41:** MF1 and MF2 are shown executing their hunks on CPU (the trace), and are scored through the fixed
  `evaluate()`.
- **L48:** the staged lines are re-run as written by the dry run. The only addition is `FORGE_GPU_DRY_RUN=1`.
- **L56:** (a) A slip in running your own tools goes under Deviations: re-run, and continue. (b) Before acceptance, a
  failed gate is a harness defect: at most 3 repairs per gate, ledgered.
- **L60:** the census now covers `_causal_mask` and the affine helpers. For each refusal, give the baseline behaviour.
  No supported dtype or valid NumPy dtype in the affine helpers is refused.
- **L65:** no deletes, as stated.
- **L82:**
  - `grep -n 'def matches\|def evaluate\|regression_reasons' artifacts/eng2/tool/evidence.py` and
    `grep -n 'test_id\|nodeid' artifacts/eng2/tool/score_gpu.py` locate the tool sites. Writable: those functions.
  - `grep -n '_causal_mask' tensor_cuda/tensor_cuda/*.py` locates the product site. Writable: its definition.
    Read-only: its callers.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- A failed gate is a result.
- **Prior Art Directive:** annotate each fix, or write "no prior art known to me".

## Done

Your final message gives, verbatim:
- the diff;
- A-G1 to A-G3 with their receipts;
- the new seal's sha256;
- every deviation;
- a "Prior art" heading.
