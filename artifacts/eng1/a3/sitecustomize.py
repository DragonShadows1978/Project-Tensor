"""A3 CPU-audit routing only (successor of a2/sitecustomize.py); never alters an instrument's evaluation or pin check.
Prior art: Python's site/sitecustomize startup hook and PYTHONPATH search (PSF, Python 2.x+; year unverified -- lead to
check 'python site sitecustomize'); a2/sitecustomize.py (ENG1 A2, 2026), from which this is adapted.
Taken: A2's activation switch, CPU-only asserts and output-root restriction. New in A3: no seal/order rebinding at all
(route (a) makes staging_seal_a3.json and the A3 order evidence.py's own defaults; this module only asserts them),
and the output root is artifacts/eng1/a3/audit. Activated explicitly (ENG1_A3_AUDIT_ROUTING=1) and inherited by
children through PYTHONPATH; G1's device-mode lines never set PYTHONPATH or the switch, so they never load it.
"""
import os
if os.environ.get('ENG1_A3_AUDIT_ROUTING') == '1':
    from pathlib import Path
    import evidence
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('ENG1_AUDIT') == '1'
    assert evidence.SEAL_NAME == 'staging_seal_a3.json', evidence.SEAL_NAME
    assert evidence.ORDER == 'orders/ENG1_A3_DEVICE_SEAL_ROUTING.md', evidence.ORDER

    def a3_guard(runs):
        runs = Path(runs).resolve()
        allowed = evidence.ROOT / 'artifacts/eng1/a3/audit'
        if runs != allowed:
            raise ValueError('A3 audit RUNS must equal ' + str(allowed))
        return runs
    evidence.guard = a3_guard
