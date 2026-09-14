"""BP-H1 suite hygiene: campaign-receipt markers for the BP-KERNEL campaigns.

A **campaign receipt** is a test that is valid only at the sha its campaign
registered.  On this tree they bind four ways:

* **sha-bound** -- asserts a source sha (``after_sha256`` / a byte-range pin)
  that the campaign froze on its own day, and that a LATER campaign moved;
* **binary-bound** -- pins the engine ``.so`` fingerprint the campaign built.
  ``*.so`` is gitignored, so the file is absent here, and the canonical build
  in ``/mnt/ForgeRealm/Project-Tensor`` is a DIFFERENT build -- the pin cannot
  be satisfied by provisioning, only by rebuilding the campaign's exact binary;
* **artifact-bound** -- reads gitignored campaign artifacts under
  ``artifacts/`` that a given tree does not carry complete;
* **worktree-path** -- pins an ABSOLUTE path inside the worktree the campaign
  was authored in (here ``/mnt/ForgeRealm/wt/grapa-bp1``, since pruned).  This
  is the strongest and least re-bindable class: re-pinning a campaign retires
  a sha-bound mark but never a worktree-path one.

Such a test fails on **its own source branch** once the tree moves past its
registration.  It is a receipt, not a regression detector, and a tree-wide run
must not report it as a regression of the tree under test.  Default collection
SKIPS them with a reason naming the registration; ``-m campaign_receipt`` or
``--campaign-receipts`` runs them.

**A missing provisionable file is NOT a receipt** (H4 principle).  Before any
test here was classified, the gitignored pinned arrays the campaigns read --
``artifacts/bp_kernel_1/{inputs,forward_state}.npz``,
``artifacts/bp_kernel_1/dry_run_final/{inputs,forward_state}.npz``,
``artifacts/bp_kernel_2/reference.npz`` and
``artifacts/bp_kernel_2/census/receipt.json`` -- were copied in from
``/mnt/ForgeRealm/Project-Tensor`` and their pins re-checked.  All of them now
match the registered sha256.  Absence of a file that CAN be provisioned is a
provisioning gap; only what survives provisioning is a receipt.

**No test assertion was changed to obtain a marker.**  Every marked test still
asserts exactly what its campaign registered; only its default *collection*
changed.

Prior art
---------
* Custom markers registered via ``pytest_configure`` + deselection in
  ``pytest_collection_modifyitems``, and a CLI opt-in flag added in
  ``pytest_addoption`` -- the canonical pytest recipe documented under
  "Working with custom markers" / "Control skipping of tests" (pytest-dev,
  Holger Krekel et al., 2009-present).  Taken verbatim as an idiom; nothing
  about it is ours.
* This file is a PORT of ``/mnt/ForgeRealm/GraftRepository/tests/conftest.py``
  (GRM-H1, 2026-09-11, this house): the ``campaign_receipt`` marker, the
  ``_requested()`` ``-m``-handoff rule and ``campaign_receipt_module()`` are
  taken from it essentially unchanged, and the rules in
  ``docs/TESTS_CAMPAIGN_RECEIPTS.md`` are copied and attributed there.  Ours
  here: the four-way class vocabulary above (GRM's had three; **binary-bound**
  is added for BP-KERNEL-2's engine-``.so`` fingerprint pin, which is neither a
  source sha nor a readable artifact), and the H4 provisioning precondition
  stated as a precondition of classification rather than as prose.  GRM's
  ``GRM_*`` environment guard is deliberately NOT ported: nothing in these two
  modules writes process environment, so porting it would add an unexercised
  mechanism.
* The "receipt valid only at its registration sha" framing is this project's
  own (GRM / BP campaign registrations, 2026).  No external prior art known to
  me for that framing.  Unverified -- lead to check; search terms: "pytest
  custom marker deselect by default", "test valid only at registration hash",
  "provenance-bound regression test".
"""
from __future__ import annotations

import pytest

MARKER = "campaign_receipt"
_OPT = "--campaign-receipts"


def pytest_addoption(parser):
    parser.addoption(
        _OPT, action="store_true", default=False,
        help="run campaign-receipt tests (sha-bound, binary-bound, "
             "artifact-bound or worktree-path-bound to a campaign "
             "registration). They are SKIPPED by default because they fail on "
             "any tree that moved past their registration -- including their "
             "own.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "campaign_receipt(registration=..., reason=...): this test is a "
        "CAMPAIGN RECEIPT, valid only at the sha/binary/paths its campaign "
        "registered. Skipped by default; run with -m campaign_receipt or "
        "--campaign-receipts.")


def _requested(config):
    """True when the run explicitly asked for campaign receipts."""
    if config.getoption(_OPT):
        return True
    expr = config.getoption("-m", default="") or ""
    # A bare "-m campaign_receipt" (or any expression naming it positively)
    # is an explicit request. "not campaign_receipt" is not -- but pytest's
    # own -m evaluation already deselects those, so naming it at all is
    # enough to hand control back to -m.
    return MARKER in expr


def campaign_receipt_module(registration, reason):
    """Declare the WHOLE module a campaign receipt, at IMPORT time.

    A campaign receipt whose binding is evaluated by the module IMPORT -- here
    ``tests/test_bp_kernel_2.py``'s ``import bp_census_2``, where
    ``scripts/bp_census_2.py`` resolves ``/mnt/ForgeRealm/wt/grapa-bp1`` at
    module scope -- cannot be handled by ``pytest_collection_modifyitems``:
    the import raises during COLLECTION, pytest reports "Interrupted: 1 error
    during collection", and the whole tree-wide run stops -- the mis-rule the
    marker exists to prevent, only worse.  ``-m campaign_receipt`` aborts the
    same way, so the receipt gate could not be run either.

    So such a module calls this at the top of its import, guarded by the
    binding it cannot satisfy.  The receipt itself is not lost: the module is
    PAIRED with a sibling ``*_receipt.py`` module carrying ONE
    ``campaign_receipt``-marked test that reproduces the same binding
    directly, so the failure stays collectable, deselected by default and
    REPRODUCED under ``-m campaign_receipt`` / ``--campaign-receipts`` like
    every other receipt.  The skip ALONE would convert a live receipt into
    silence -- the pair is what keeps it honest.  Marks inside the skipped
    module cannot be collected while the import fails; they are kept anyway,
    so that a tree which restores the binding inherits a correctly-classified
    module rather than a wall of unexplained failures.

    Marks still belong on test FUNCTIONS wherever collection can happen --
    this is only for modules whose binding is evaluated by the import itself.

    Prior art: ported from GraftRepository ``tests/conftest.py`` (GRM-H2,
    2026-09-11), where the same shape was derived for
    ``tests/test_grm_c5_grounding.py``. Taken unchanged.
    """
    pytest.skip(
        "campaign receipt (registration=%s): %s; bound to its campaign; the "
        "binding is reproduced by the campaign_receipt-marked test in the "
        "companion *_receipt.py module -- run with -m campaign_receipt"
        % (registration, reason),
        allow_module_level=True)


# --------------------------------------------------------------- BP-H2 -----
# Campaign receipts in a module the campaign PINS BY SHA.
#
# BP-KERNEL-4's registration pins its own test module:
# ``artifacts/bp_kernel_4/registration.json`` carries
# ``pins['tests/test_bp_kernel_4.py'] = 188afdc8...``, and
# ``verify_registration()`` walks that ``pins`` dict (scripts/bp_kernel_4.py:60)
# BEFORE it reaches the engine-binary check (line 72).  So writing ANY byte
# into the module -- including a ``@pytest.mark.campaign_receipt`` decorator
# and a comment, which change no assertion -- makes the test fail with
# ``ValueError: pin drift: tests/test_bp_kernel_4.py`` INSTEAD of the
# binary-bound ``FileNotFoundError`` it exists to record.  Marking in place
# would destroy the very receipt the mark is supposed to preserve, and would
# silently re-label the binding.
#
# This is the same shape as BP-H1's import-time case (``campaign_receipt_module``):
# a binding that a function-level decorator cannot reach.  BP-H1's answer there
# was a PAIR, because the import aborted collection.  Here collection is fine
# and only the file's bytes are frozen, so the answer is simpler and stricter:
# apply the marker from OUTSIDE the pinned file, by node id, from
# ``tests/conftest.py`` -- which is pinned by no BP registration (checked all
# seven: bp_kernel_1..4 and the three census registrations).  The module stays
# byte-identical to its pin, the receipt reproduces its REAL binding under
# ``-m campaign_receipt``, and no assertion is touched.
#
# An entry here is a mark, not a skip: it is added as the real
# ``campaign_receipt`` marker, so ``-m campaign_receipt`` SELECTS it and
# ``--campaign-receipts`` runs it exactly like an in-file mark.  Keep this
# table minimal -- it is only for modules a campaign froze by sha.
#
# Prior art: ``pytest_collection_modifyitems`` adding markers by ``item.nodeid``
# is the canonical pytest idiom for marking tests you cannot edit (pytest-dev,
# "Working with custom markers"; the same mechanism third-party plugins use to
# mark vendored suites) -- taken as an idiom, nothing about it is ours.  The
# campaign-receipt framing, the marker and the class vocabulary are BP-H1's
# (this house, 2026-09-13), itself a port of GraftRepository GRM-H1/H2
# (2026-09-11) -- taken unchanged.  Ours in BP-H2: the observation that a
# registration pinning its OWN test module makes in-file marking
# self-defeating, and marking from the unpinned conftest as the remedy.
# Unverified -- lead to check; no network in this seat.  Search terms:
# "pytest add marker by nodeid conftest", "mark test in unmodifiable module",
# "self-referential test file hash pin".
PINNED_MODULE_RECEIPTS = {
    "tests/test_bp_kernel_4.py::test_registration_pins_and_protocol": dict(
        registration="artifacts/bp_kernel_4/registration.json",
        reason="binary-bound: the engine .so BP-KERNEL-4 built is pinned "
               "eaec08ed... in artifacts/bp_kernel_4/engine_build_receipt.json; "
               "*.so is gitignored so it is absent here, and the canonical "
               "build is b6022d3d... -- a DIFFERENT build, so the pin is not "
               "satisfiable by provisioning, only by rebuilding BP-KERNEL-4's "
               "exact binary. Marked from conftest because the registration "
               "also pins this test module's own sha (188afdc8...), so editing "
               "the file in place would convert the receipt into 'pin drift'."),
}


def _apply_pinned_module_marks(items):
    """Attach campaign_receipt markers to tests in sha-pinned modules."""
    for item in items:
        spec = PINNED_MODULE_RECEIPTS.get(item.nodeid)
        if spec is not None and item.get_closest_marker(MARKER) is None:
            item.add_marker(pytest.mark.campaign_receipt(**spec))


def pytest_collection_modifyitems(config, items):
    _apply_pinned_module_marks(items)
    if _requested(config):
        return
    for item in items:
        mark = item.get_closest_marker(MARKER)
        if mark is None:
            continue
        registration = mark.kwargs.get("registration", "unregistered")
        item.add_marker(pytest.mark.skip(reason=(
            "campaign receipt (registration=%s): bound to its campaign; "
            "run with -m campaign_receipt" % registration)))
