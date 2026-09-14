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

---

# BP-H2 delta — BP-KERNEL-3 / BP-KERNEL-4 on `main` (2026-09-14)

Second pass, same mechanism, applied to the two campaigns that merged to `main`
at `0b8c643` after BP-H1. On canonical `tests/` this tree read **1 failed, 122
passed, 6 skipped**; the single failure is
`tests/test_bp_kernel_4.py::test_registration_pins_and_protocol`.

Every run: `PYTHONPATH=tensor_cuda python3 -m pytest -q -p no:cacheprovider …`

## Provisioning (done before classification — the H4 precondition)

Nine gitignored pinned files were absent from this worktree and were copied in
from `/mnt/ForgeRealm/Project-Tensor` at the same relative paths **before any
test was classified**:

| File | sha256 (16) | Bytes |
|---|---|---|
| `artifacts/bp_kernel_1/inputs.npz` | `53c38919ad77f3f8` | 54,527,148 |
| `artifacts/bp_kernel_1/forward_state.npz` | `77afc01e8a912e3a` | 8,651,488 |
| `artifacts/bp_kernel_1/dry_run_final/inputs.npz` | `9a514b06e0d1eab0` | 1,696 |
| `artifacts/bp_kernel_1/dry_run_final/forward_state.npz` | `a2f9cce6b9e7da78` | 936 |
| `artifacts/bp_kernel_2/reference.npz` | `43f6d06a21090ade` | 134,481,096 |
| `artifacts/bp_kernel_2/census/receipt.json` | `6afd46f78e97fc0e` | 52,883,968 |
| `artifacts/bp_kernel_3/census/receipt.json` | `de9f6adf14b756b4` | 52,885,018 |
| `artifacts/bp_kernel_4/reference.npz` | `b46c7767b4c9c2e1` | 84,411,354 |
| `artifacts/bp_kernel_4/census/receipt.json` | `bb15b7431f4c3293` | 53,094,554 |

**H4 earned its keep again, and this time it was measured.** The provisioning
was removed and the suite re-run to see what provisioning actually buys:

```
$ … tests/          (provisioned files moved aside)
3 failed, 120 passed, 6 skipped in 1.36s
FAILED tests/test_bp_kernel_4.py::test_downstream_red_stops_timing
FAILED tests/test_bp_kernel_4.py::test_receipt_adversarial_schema
FAILED tests/test_bp_kernel_4.py::test_registration_pins_and_protocol

$ … tests/          (files restored)
1 failed, 122 passed, 6 skipped in 2.62s
```

`test_downstream_red_stops_timing` and `test_receipt_adversarial_schema` are
**provisioning gaps, not receipts** — they pass the moment the arrays are
present. Classifying before provisioning would have mis-marked two healthy
tests. After provisioning, exactly one failure remains.

## Registration audit — BP-KERNEL-4

`artifacts/bp_kernel_4/registration.json`, audited on this tree with
provisioning in place:

* `registration.json` self-sha vs `registration.sha256`: **OK**
* protocol drift: **NONE** (all 27 `protocol()` keys match)
* `pins`: **45 / 45 OK**
* `regions` (byte-ranges in `tensor_cuda/src/kernels.cu`): **3 / 3 OK**
* `engine_build_receipt.json`: `rc == 0`, `source_pins` match the registration
* `reference_receipt.json`: registration sha **matches**, `reference.npz` sha
  **matches**
* **1 BAD**: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`,
  pinned `eaec08ed7b849f03…`. **Absent** here (`*.so` is gitignored); the
  canonical build at `/mnt/ForgeRealm/Project-Tensor` is
  `b6022d3dcd5b3b76…` — a **different build** (same 9,442,856 bytes, different
  content). **binary-bound.**

The test's last line, `c.verify()`, binds to the **same** `.so`: with the
kernel registration set aside, `bp_census_4.verify()` still raises
`FileNotFoundError` on that path. So the test is bound twice over by one
fingerprint. There is **no worktree-path binding** here — unlike BP-KERNEL-2,
`bp_census_4.GRAPA` is the canonical `/mnt/ForgeRealm/GRAPA-Native-LLM`, which
resolves.

## Added by BP-H2

**A registration that pins its own test module cannot be marked in place.**
BP-KERNEL-4's registration carries
`pins['tests/test_bp_kernel_4.py'] = 188afdc8…`, and `verify_registration()`
walks that `pins` dict (`scripts/bp_kernel_4.py:60`) **before** it reaches the
engine-binary check (line 72). Adding a `@pytest.mark.campaign_receipt`
decorator and an explanatory comment — changing **no assertion** — was enough
to move the failure from the binary-bound `FileNotFoundError` to:

```
E   ValueError: pin drift: tests/test_bp_kernel_4.py
```

That is the mark **destroying the receipt it exists to preserve**, and silently
re-labelling the binding: a reader of the receipt gate would see "pin drift"
and conclude the campaign's *sources* had moved, when in fact only the marker
had been added. The decorator was reverted and the module restored to its
pinned bytes (`188afdc8…`, verified identical to canonical).

This is the same *shape* as BP-H1's import-time case, with a different remedy.
BP-H1 needed a **pair** because the import aborted collection and the mark
could not be reached at all. Here collection is healthy and only the file's
**bytes** are frozen, so the mark is applied **from outside the pinned file**,
by node id, in `tests/conftest.py` — which is pinned by **no** BP registration
(checked all seven: `bp_kernel_1..4` plus the three census registrations). The
table is `PINNED_MODULE_RECEIPTS`; entries are added as the real
`campaign_receipt` **marker**, not a bare skip, so `-m campaign_receipt`
selects them and `--campaign-receipts` runs them exactly like an in-file mark.

The rule, stated generally: **a receipt in a module its own campaign froze by
sha must be marked from the unpinned conftest, never edited in place.** An
in-file mark on such a module is not a lesser evil — it is a false receipt.

## The BP-H2 delta table

| Node id | Class | Reason | Registration |
|---|---|---|---|
| `tests/test_bp_kernel_4.py::test_registration_pins_and_protocol` | **binary-bound** | Of the whole registration — 45/45 pins, 3/3 regions, protocol, build source pins, reference receipt — exactly one binding is unsatisfiable: the engine `.so` BP-KERNEL-4 built, pinned `eaec08ed…`. Gitignored, so absent here; canonical is `b6022d3d…`, a **different build**. Not provisionable, only rebuildable. `c.verify()` on the test's last line binds to the same `.so`. Marked from `tests/conftest.py` (`PINNED_MODULE_RECEIPTS`) because the registration also pins this module's own sha `188afdc8…` — an in-file decorator turns the receipt into `pin drift`. | `artifacts/bp_kernel_4/registration.json` |
| **1 module** | **1 marked function** (0 in-file edits) | | |

### Not marked

| Module | Node id | Why not |
|---|---|---|
| `tests/test_bp_kernel_3.py` | **all 57** | **They pass.** `57 passed in 0.83s` standalone. Checked for the two ways a pass can be hollow: (a) *provisioning* — an `io.open` trace over the whole bk3 run shows it opens **none** of the nine provisioned files, and bk3 was `57 passed` in the deprovisioned run too; (b) *worktree path* — `test_canonical_paths_and_missing_gate` pins `/mnt/ForgeRealm/GRAPA-Native-LLM`, a **canonical repo**, not a worktree, and it resolves. `test_registration_fail_closed_on_synthetic_pins` builds a synthetic registration under `tmp_path` with a fixture binary, deliberately so the test is not bound to a local GPU binary — it proves the fail-closed *mechanism* and is not campaign evidence. Nothing to mark. |
| `tests/test_bp_kernel_4.py` | `test_downstream_red_stops_timing`, `test_receipt_adversarial_schema` | **Provisioning gaps, not receipts** (H4). They failed only while the pinned arrays were absent and pass once provisioned. Marking them would have been the exact error H4 exists to prevent. |
| `tests/test_bp_kernel_4.py` | the other 6 bk4 tests | Pass on this tree. |

## Failures that are NOT receipts

**None.** The single failure resolved to a registration binding — an engine
binary fingerprint that cannot be satisfied by provisioning. No genuine defect
was found in bk3 or bk4, so nothing needed pinning-without-marking.

**No test assertion was changed.** `tests/test_bp_kernel_4.py` is byte-identical
to its pin (`188afdc8…`) and to the canonical checkout; `tests/test_bp_kernel_3.py`
is untouched. The only edited test file is `tests/conftest.py`.

### Recorded, not marked away

BP-H1 left `scripts/bp_census_2.py:30`'s hard-coded
`/mnt/ForgeRealm/wt/grapa-bp1` as **lead's call**. BP-H2 did not touch it
(`scripts/` is outside this order's writable target) and it remains open. Note
that BP-KERNEL-3 and BP-KERNEL-4 did **not** repeat that mistake:
`bp_census_3.GRAPA` and `bp_census_4.GRAPA` both point at the canonical
`/mnt/ForgeRealm/GRAPA-Native-LLM`, which is why neither campaign contributed a
worktree-path receipt.

A second item for the lead, arising from this pass: **BP-KERNEL-4 pinning its
own test module** (`tests/test_bp_kernel_4.py`) makes that module immutable for
as long as the receipt is to be reproducible — no future hygiene pass, rename
or lint fix can touch it without converting the receipt into `pin drift`.
BP-KERNEL-1/2/3 do not pin their test modules. Whether a campaign *should* pin
its own tests is a registration-design question, not a test-hygiene one;
recorded here, **lead's call**.

## Gates (BP-H2, `bp-h2` worktree, 2026-09-14)

```
$ … tests/
122 passed, 7 skipped in 2.44s

$ … -m campaign_receipt tests/
6 failed, 1 skipped, 122 deselected in 0.51s

$ … --campaign-receipts tests/
6 failed, 122 passed, 1 skipped in 2.75s

$ … tests/test_bp_kernel_3.py
57 passed in 0.83s

$ … -m campaign_receipt tests/test_bp_kernel_4.py
1 failed, 47 deselected in 0.29s
```

**0 failed** by default, which was the point of the pass.

Skip arithmetic, `tests/` default: 7 skipped = BP-H1's 6 (4 bk1 marks + 1
bk2-receipt mark + 1 module-level bk2 skip) **+ 1** new bk4 mark.
Receipt-gate arithmetic: 6 failed = BP-H1's 5 collectable marks **+ 1** bk4,
plus the same 1 module-level skip. Both counts move by exactly one, matching
the one test this pass marked.

The bk4 receipt reproduces its **real** binding under the gate — the engine
`.so`, not `pin drift`:

```
E   FileNotFoundError: [Errno 2] No such file or directory:
    '/mnt/ForgeRealm/wt/pt-bph2/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
```

and its default skip reason names the registration:

```
SKIPPED [1] tests/test_bp_kernel_4.py: campaign receipt
  (registration=artifacts/bp_kernel_4/registration.json): bound to its
  campaign; run with -m campaign_receipt
```

### Engine smoke — NOT RUN in this worktree

Unchanged from BP-H1 and for the same reason: `tensor_cuda/tensor_cuda/*.so` is
gitignored and absent here, so those four modules fail at collection on
`import tensor_cuda`. Building it would mean writing under `tensor_cuda/`,
which this order excludes, and this seat has no GPU. Reported as NOT-RUN.

## Prior art

* **pytest custom markers** — registering via `pytest_configure`, deselecting in
  `pytest_collection_modifyitems`, opt-in flag via `pytest_addoption`, and
  **adding a marker by `item.nodeid` for a test you cannot edit** (the
  mechanism third-party plugins use to mark vendored suites): pytest's
  "Working with custom markers" / "Control skipping of tests" (pytest-dev,
  Holger Krekel et al., 2009–present). Taken as an idiom; nothing about it is
  ours.
* **The campaign-receipt framing, the marker, the four-way class vocabulary
  (including binary-bound), the H4 provisioning precondition, and
  `campaign_receipt_module()`** — BP-H1 (this house, 2026-09-13), itself a port
  of GraftRepository GRM-H1/H2/H3/D2 (2026-09-11/12). Taken unchanged; BP-H2
  added no new class.
* **Ours in BP-H2** — the finding that a campaign registration pinning its
  **own test module's sha** makes in-file marking self-defeating (it converts a
  binary-bound receipt into `pin drift`, silently re-labelling the binding),
  and `PINNED_MODULE_RECEIPTS` in the unpinned `tests/conftest.py` as the
  remedy. Also the *measured* H4 demonstration (deprovision → 3 failed,
  reprovision → 1 failed) rather than an asserted one.
* **Unverified — lead to check.** No network in this seat; the pytest citation
  is from memory. Search terms: "pytest add marker by nodeid conftest",
  "pytest mark test in unmodifiable module", "self-referential test file hash
  pin", "provenance-bound regression test".
* Prior art declared in the modules under test (BP-KERNEL-3's and
  BP-KERNEL-4's own headers — FlashAttention, Dao 2022/2023; NumPy, Harris
  et al. 2020; gprof, Graham/Kessler/McKusick 1982; CUDA events, NVIDIA 2007+;
  SHA256, NIST 2001; pytest, Krekel 2004) is untouched by this pass. BP-H2
  changed no algorithm.
