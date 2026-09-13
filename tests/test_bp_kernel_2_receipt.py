"""BP-H1: the collectable half of BP-KERNEL-2's module-level campaign receipt.

``tests/test_bp_kernel_2.py`` evaluates BP-CENSUS-2's binding at IMPORT time
(``import bp_census_2`` -> ``scripts/bp_census_2.py`` line 30 resolves
``/mnt/ForgeRealm/wt/grapa-bp1`` at module scope and exec's a file inside it).
That worktree is pruned, so the import raises during COLLECTION and, unguarded,
aborts the whole run.  The guard there is ``campaign_receipt_module()``, which
skips the module unconditionally -- and a skip ALONE would convert a live
receipt into silence: the three ``campaign_receipt`` marks inside that module
can never be collected while the import fails, so ``-m campaign_receipt``
would report them as nothing at all.

This module is the other half of the pair.  It carries ONE
``campaign_receipt``-marked test that reproduces the SAME import binding
directly, without importing the campaign module, so the receipt stays
collectable, deselected by default, and REPRODUCED under
``-m campaign_receipt`` / ``--campaign-receipts`` like every other receipt.

Prior art: the pair shape (module-level skip + one collectable marked test in
a sibling module) is ported unchanged from GraftRepository GRM-H2, 2026-09-11,
where it was derived for ``tests/test_grm_c5_grounding.py`` /
``tests/test_grm_c5_grounding_receipt.py``.  Nothing about the shape is ours;
the binding it reproduces is BP-CENSUS-2's.  See
``docs/TESTS_CAMPAIGN_RECEIPTS.md``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT/'scripts/bp_census_2.py'
#: The campaign worktree scripts/bp_census_2.py pins at module scope. Pruned.
GRAPA = Path('/mnt/ForgeRealm/wt/grapa-bp1')


@pytest.mark.campaign_receipt(
    registration='artifacts/bp_kernel_2/census/registration.json',
    reason='scripts/bp_census_2.py resolves the pruned worktree '
           '/mnt/ForgeRealm/wt/grapa-bp1 at module scope, so '
           'tests/test_bp_kernel_2.py cannot be imported at all')
def test_bp_census_2_import_binding_is_a_campaign_receipt():
    """The import binding that makes ``test_bp_kernel_2.py`` unimportable.

    Asserts the binding as BP-CENSUS-2 registered it: the module-scope GRAPA
    path is what it is, and importing the script resolves it.  On a tree that
    carries ``grapa-bp1`` this passes; here it raises the same
    ``FileNotFoundError`` the collection abort raised, which is the receipt.
    """
    # The pin is in the source, at module scope -- not behind a function.
    assert "GRAPA=Path('%s')" % GRAPA in CENSUS.read_text(), (
        'scripts/bp_census_2.py no longer pins %s at module scope; if it was '
        're-pinned repo-relative, this receipt and the three marks in '
        'tests/test_bp_kernel_2.py have retired -- re-derive, do not keep '
        'them' % GRAPA)

    # Reproduce the abort itself, as collection would have hit it. The
    # scripts/ path setup mirrors tests/test_bp_kernel_2.py exactly, so the
    # error raised here is the one that aborts that module's import -- a
    # ModuleNotFoundError for bp_kernel_2 would mean this probe, not the
    # campaign, is broken.
    if str(ROOT/'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('bp_census_2_receipt_probe',
                                                  CENSUS)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:                            # pragma: no cover
        pytest.fail('probe setup is wrong, not the campaign binding: %s' % exc)
