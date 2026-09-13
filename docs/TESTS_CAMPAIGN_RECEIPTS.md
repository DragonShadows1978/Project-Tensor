# Campaign-receipt tests (BP-H1, Project-Tensor, 2026-09-13)

This file ports GraftRepository's campaign-receipt mechanism to Project-Tensor
for the BP-KERNEL-1 / BP-KERNEL-2 harness tests on `main`.

**Attribution.** The rules in the next section are **copied, with edits marked,
from `/mnt/ForgeRealm/GraftRepository/docs/TESTS_CAMPAIGN_RECEIPTS.md`**
(GRM-H1 2026-09-11, GRM-H2 2026-09-11, GRM-H3, GRM-D2 2026-09-12). The marker
implementation in `tests/conftest.py` is a port of that repository's
`tests/conftest.py`. Nothing in the mechanism is original to BP-H1; what is
new here is the **binary-bound** class and the H4 provisioning precondition,
both flagged below.

---

## The rules (copied from GraftRepository, attributed above)

A **campaign receipt** is a test that is valid only at the sha its campaign
registered. It either

* asserts `INPUT_SHA_MISMATCH`-class binding against source shas that the
  campaign froze on its own day, or
* reads gitignored campaign artifacts under `artifacts/` that a given tree
  does not carry complete.

Such a test fails on **its own source branch** once the tree moves past its
registration. It is a receipt, not a regression detector, and a tree-wide run
must not report it as a regression of the tree under test.

**A test that pins its own worktree path is always a receipt** (GRM-H1). If a
test, or the script it drives, verifies shas against **absolute paths inside
the worktree it was authored in**, it **cannot pass on any other tree**, by
construction, no matter how the core moves or how the campaign is re-pinned.
That is the strongest form of campaign receipt, and the most durable:
rebinding a campaign's pins retires a sha-bound mark, but never a
worktree-path one.

**An inherited default is not a controlled arm** (GRM-D2). A test that is
silent about a setting reads whatever the tree currently defaults to, so a
default flip moves it without any assertion changing. A campaign receipt must
PIN the arm it was registered under; an arm is a PIN, never an absence. This
is why a completed campaign is **marked** rather than rebound to new defaults
— rebinding would re-label a finished run as having been done under settings
it never saw.

**A receipt whose binding runs at IMPORT time needs a pair** (GRM-H2). If the
binding is evaluated by the module import itself, a function-level mark cannot
reach it: the import raises during collection, pytest reports `Interrupted: 1
error during collection`, and the **whole tree-wide run stops** — the mis-rule
the marker exists to prevent, only worse. `-m campaign_receipt` aborts
identically, so the receipt gate cannot run either. The fix is a pair: the
module skips itself at import via `campaign_receipt_module()`, and a companion
module carries ONE `campaign_receipt`-marked test invoking the same binding
directly, so the receipt stays collectable and reproduced under the gate. The
skip alone would convert a receipt into silence; the pair is what keeps it
honest.

**Mark the function, not the module**, wherever collection can happen, so
healthy coverage in a mixed module keeps running.

**No test assertion may be changed to obtain a mark.** Every marked test still
asserts exactly what its campaign registered; only its default *collection*
changes.

**A failure that is not a receipt must be fixed or reported, never marked.**
Marking a genuine defect files a bug under a label that means "not our
problem".

### Added by BP-H1

**A missing provisionable file is NOT a receipt** (H4 principle). Absence of a
gitignored file that *can* be copied in from the canonical repository is a
**provisioning gap**, not a binding. Provision first, then classify; only what
survives provisioning is a receipt. See "Provisioning" below — this pass moved
one bk2 pin from "apparently failing" to "verifying" purely by provisioning,
and had classification run first, that pin would have been mis-marked.

**binary-bound** is a fourth class, alongside sha-bound / artifact-bound /
worktree-path. A campaign may pin the **engine binary fingerprint** it built —
here `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`.
`*.so` is gitignored, so the file is absent from any fresh tree, and the
canonical build is a *different* build with a different sha. Such a pin cannot
be satisfied by provisioning (unlike an artifact) and is not a source sha
(unlike sha-bound): only rebuilding the campaign's exact binary would satisfy
it. It is therefore its own class, and effectively as durable as a
worktree-path pin.

## The marker

```python
@pytest.mark.campaign_receipt(
    registration='artifacts/bp_kernel_1/registration.json',
    reason='...')
def test_...():
```

Registered in `tests/conftest.py`. Behaviour:

| Invocation | Campaign receipts |
|---|---|
| `pytest tests/` | **SKIPPED**, with a reason naming the registration |
| `pytest -m campaign_receipt tests/` | run (and expected to fail) |
| `pytest --campaign-receipts tests/` | run alongside everything else |

`tests/conftest.py` is **new**: Project-Tensor had no conftest at any level
(checked `./conftest.py`, `tests/conftest.py`, and `pyproject.toml` — no
`[tool.pytest.ini_options]`, no `pytest.ini`, no `setup.cfg`, no `tox.ini`).
GraftRepository's `GRM_*` environment guard was deliberately **not** ported:
nothing in these two modules touches process environment, so porting it would
land an unexercised mechanism.

## Provisioning (done before classification)

Six gitignored pinned files that the campaigns read were absent from this
worktree and were copied in from `/mnt/ForgeRealm/Project-Tensor` at the same
relative paths, before any test was classified:

| File | Result after provisioning |
|---|---|
| `artifacts/bp_kernel_1/inputs.npz` | pin `53c38919…` **verifies** |
| `artifacts/bp_kernel_1/forward_state.npz` | pin `77afc01e…` **verifies** |
| `artifacts/bp_kernel_1/dry_run_final/inputs.npz` | read by the dry-run receipt |
| `artifacts/bp_kernel_1/dry_run_final/forward_state.npz` | read by the dry-run receipt |
| `artifacts/bp_kernel_2/reference.npz` | pin `43f6d06a…` **verifies** |
| `artifacts/bp_kernel_2/census/receipt.json` | read by the census gate |

This is why BP-KERNEL-2's registration audit comes out at **41 of 42 pins
verifying**. Before provisioning it would have looked like several failures;
after it, exactly one pin is unsatisfiable, and that one is the engine binary.
Classifying first would have mis-marked provisioning gaps as receipts.

## Registration audits (the evidence behind the classes)

**BP-KERNEL-1** — `artifacts/bp_kernel_1/registration.json`, 12 pins:

* 4 BAD, absent: `/mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json`,
  `…/grapa/attention.py`, `…/grapa/attention_mla.py`, `…/grapa/model_mla.py`
  — the campaign worktree is pruned (`/mnt/ForgeRealm/wt/` now holds only
  `apamq-fa2` and `pt-bph1`). **worktree-path.**
* 1 BAD, absent: the engine `.so`. **binary-bound.**
* 7 OK, including `inputs.npz` after provisioning.
* `sources`: all three `after_sha256` (`kernels.cu`, `ops.cpp`,
  `bindings.cpp`) BAD; all three `before_sha256` OK. **sha-bound** — main
  carries BP-KERNEL-2's sources now.

**BP-KERNEL-2** — `artifacts/bp_kernel_2/registration.json`, 42 pins:

* **41 OK.** All four `sources` before/after sha pairs OK. The
  `variant_a_region` byte-range sha OK.
* 1 BAD, absent: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`,
  pinned `6fd610a5…`. The canonical build at `/mnt/ForgeRealm/Project-Tensor`
  is `ff330c8a…` — a **different build**, so this is not provisionable.
  **binary-bound.**

## The BP-H1 delta table

Marks sit on the **test function**, never the module, except where the binding
is evaluated by the import (one case, handled by the pair). Every class below
was established by reproducing the failure **in isolation, in its own pytest
process**; no mark rests on another mark.

| Module | Node id | Class | Reason | Registration |
|---|---|---|---|---|
| `tests/test_bp_kernel_1.py` | `test_registration_and_immutability` | worktree-path | `verify_registration()` raises `FileNotFoundError` on `/mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json` before reaching any assertion. Also binary-bound and sha-bound (see audit). | `artifacts/bp_kernel_1/registration.json` |
| `tests/test_bp_kernel_1.py` | `test_source_drift_closed` | worktree-path | Same pin walk dies on the same pruned path, so the `source drift` branch the test exists to prove is never reached. | `artifacts/bp_kernel_1/registration.json` |
| `tests/test_bp_kernel_1.py` | `test_default_regions_unchanged_vs_pinned_main` | sha-bound | BP-KERNEL-1 registered `ops.cpp` as append-only over its baseline; BP-KERNEL-2 **inserted** at byte 438, so `startswith(baseline)` is false. The `kernels.cu` region sha, the `kernels.cu` prefix and the `bindings.cpp` slice all still PASS — only `ops.cpp` moved. | `artifacts/bp_kernel_1/baseline_pins.json` + `artifacts/bp_kernel_1/baseline/ops.cpp` |
| `tests/test_bp_kernel_1.py` | `test_dry_run_subprocess_precedence_and_schema` | worktree-path | The child `scripts/bp_kernel_1.py --run --dry-run` **fails closed** (exit 2) with `FAIL_CLOSED FileNotFoundError: … /mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json`. Failing closed is the script behaving correctly. | `artifacts/bp_kernel_1/registration.json` |
| `tests/test_bp_kernel_2.py` | *(module import)* | worktree-path | `scripts/bp_census_2.py:30` is `GRAPA=Path('/mnt/ForgeRealm/wt/grapa-bp1')` and exec's a file inside it at module scope. Aborts collection of the **entire run**. Guarded by `campaign_receipt_module()`; paired with the module below. | `artifacts/bp_kernel_2/census/registration.json` |
| `tests/test_bp_kernel_2_receipt.py` **(new)** | `test_bp_census_2_import_binding_is_a_campaign_receipt` | worktree-path | The collectable half of the pair: reproduces the import binding directly, so the receipt is not lost to the module skip. | `artifacts/bp_kernel_2/census/registration.json` |
| `tests/test_bp_kernel_2.py` | `test_registration_and_fail_closed` | **binary-bound** | Of 42 pins, exactly one is unsatisfiable: the pt-bk2 engine `.so` (`6fd610a5…`; canonical is `ff330c8a…`). Gitignored and a different build — not provisionable. | `artifacts/bp_kernel_2/registration.json` |
| `tests/test_bp_kernel_2.py` | `test_census_registration_and_fixed_protocol` | worktree-path **and** binary-bound | Reproduced with `GRAPA` redirected to canonical `/mnt/ForgeRealm/GRAPA-Native-LLM`: `c.verify()` **still** raises, now on the absent engine `.so`. Bound twice over; repairing either alone would not free it. | `artifacts/bp_kernel_2/census/registration.json` |
| `tests/test_bp_kernel_2.py` | `test_census_red_gate_timing_only_and_incomplete` | worktree-path (import only) | **Its body PASSES.** Replayed by hand against `bp_census_2` with `GRAPA` redirected: all three `summarize()` calls give the registered verdicts (`TIMING_ONLY_NOT_A_VALID_STEP` / `CONFIRMED` / `INCONCLUSIVE`). Marked solely because the module import cannot resolve. Retires the moment `scripts/bp_census_2.py` is re-pinned repo-relative. | `artifacts/bp_kernel_2/census/registration.json` |
| **3 modules** | **8 marked functions + 1 module-level** | | | |

### Not marked

| Module | Node id | Why not |
|---|---|---|
| `tests/test_bp_kernel_2.py` | `test_schema_rejects_incomplete_or_cpu_claim` | **It PASSES.** The order listed it among the expected failures, but it only ever appeared to fail as collateral of the module-level import abort. Reproduced in isolation with the import stubbed: passes. Marking a passing test is forbidden. |
| `tests/test_bp_kernel_2.py` | `test_variant_a_pinned_bytes_and_default_body`, `test_f_has_key_ownership_and_reuses_c_without_atomics`, and the 18 other bk2 tests | Pass on this tree. |
| `tests/test_bp_kernel_1.py` | the other 18 bk1 tests | Pass on this tree. |

### Corrections to the order's predicted failure list

The order said "verify the current list yourself". Two corrections:

1. **`tests/test_bp_kernel_1.py::test_registration_and_immutability` fails and
   was NOT on the list.** bk1 has **4** failures, not 3.
2. **`tests/test_bp_kernel_2.py::test_schema_rejects_incomplete_or_cpu_claim`
   PASSES and WAS on the list.** It was collateral of the collection abort.

Both were established by isolated reproduction, not inference.

## Failures that are NOT receipts

**None.** Every failure in both modules resolved to a registration binding — a
pruned absolute worktree path, an engine-binary fingerprint, or a source sha a
later campaign moved past. No genuine defect was found, so nothing needed
pinning-without-marking, and no assertion was changed anywhere.

One finding sits close to the line and is recorded rather than marked away:
`scripts/bp_census_2.py:30` hard-codes `/mnt/ForgeRealm/wt/grapa-bp1`, and
under GraftRepository's **GRM-F6 rule** ("scripts and registrations pin
repo-relative paths; a dead-path glob is RED, never zero rows") that is a
latent defect in the script, not merely a receipt: the file it needs,
`scripts/bp_census_1.py`, **does exist** in the canonical
`/mnt/ForgeRealm/GRAPA-Native-LLM`, so a repo-relative or configurable pin
would resolve today. `scripts/` is outside BP-H1's writable target, so this
pass **reports** it instead of repairing it. Repairing it would retire the
module-level receipt, the companion receipt module, and the mark on
`test_census_red_gate_timing_only_and_incomplete` (whose body already passes),
leaving `test_census_registration_and_fixed_protocol` correctly marked
binary-bound. **Lead's call.**

## Gates (BP-H1, `bp-h1` worktree, 2026-09-13)

Every run: `PYTHONPATH=tensor_cuda python3 -m pytest -q -p no:cacheprovider …`

```
$ … tests/test_bp_kernel_1.py tests/test_bp_kernel_2.py
18 passed, 5 skipped in 0.14s

$ … tests/
18 passed, 6 skipped in 0.16s

$ … -m campaign_receipt tests/test_bp_kernel_1.py tests/test_bp_kernel_2.py
4 failed, 1 skipped, 18 deselected in 0.28s

$ … -m campaign_receipt tests/
5 failed, 1 skipped, 18 deselected in 0.29s

$ … --campaign-receipts tests/
5 failed, 18 passed, 1 skipped in 0.35s
```

**0 failed** by default, which was the point of the pass.

Skip arithmetic, `tests/` default run: 6 skipped = 4 bk1 marks + 1 bk2-receipt
mark + 1 module-level bk2 skip. The 3 marks *inside* `test_bp_kernel_2.py` are
not collected at all while its import fails, which is why 6, not 9. They are
kept so that a tree restoring the binding inherits a correctly-classified
module instead of three unexplained failures.

Receipt-gate arithmetic, `tests/`: 5 failed + 1 skipped = the 5 collectable
marks + the module-level skip, matching the default skip count exactly.

### Engine smoke — NOT RUN in this worktree

```
$ … tensor_cuda/tests/test_ops_phase2.py tensor_cuda/tests/test_attention_phase5.py \
      tensor_cuda/tests/test_apa_phase6.py tensor_cuda/tests/test_checkpoint.py
4 errors in 0.31s
ERROR … ModuleNotFoundError: No module named '_tensor_cuda'
```

Cause is **not** the GPU and **not** these tests: the compiled extension
`tensor_cuda/tensor_cuda/_tensor_cuda*.so` is gitignored and absent from this
worktree, so all four modules fail at collection on `import tensor_cuda`.
Building it would mean writing under `tensor_cuda/`, which BP-H1's order
excludes. Reported as NOT-RUN.

These four are **not device-bound**: for reference only, the same command in
the canonical `/mnt/ForgeRealm/Project-Tensor` checkout (read-only, its `.so`
present) gives `17 passed in 1.12s`. That is a reference observation about the
suite, **not** a gate for this branch — this branch's engine smoke is unrun
until a `.so` exists here.

## Prior art

* **pytest custom markers**: registering via `pytest_configure`, deselecting
  in `pytest_collection_modifyitems`, opt-in flag via `pytest_addoption` — the
  canonical recipe from pytest's "Working with custom markers" / "Control
  skipping of tests" (pytest-dev, Holger Krekel et al., 2009–present). Taken
  verbatim as an idiom; nothing about it is ours.
* **The campaign-receipt framing, the marker, `campaign_receipt_module()`, the
  import-time pair, and the rules above**: GraftRepository GRM-H1/H2/H3/D2
  (this house, 2026-09-11/12). Ported; taken essentially unchanged.
* **Ours in BP-H1**: the **binary-bound** class (an engine-`.so` fingerprint
  pin is neither a source sha nor a readable artifact, and unlike an artifact
  it cannot be provisioned), and the **H4 provisioning precondition** as a
  gate on classification rather than prose.
* **Unverified — lead to check.** No network in this seat; the pytest
  documentation citation above is from memory. Search terms: "pytest custom
  marker deselect by default", "pytest addoption opt-in marker",
  "provenance-bound regression test", "test valid only at registration hash".
* Prior art already declared in the modules under test (BP-KERNEL-1's and
  BP-KERNEL-2's own headers — FlashAttention, Dao et al. 2022; NumPy, Harris
  et al. 2020; gprof, Graham/Kessler/McKusick 1982; AdamW, Loshchilov and
  Hutter 2019) is untouched by this pass; BP-H1 changed no algorithm.
