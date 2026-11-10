"""A2 audit routing only; never alter an instrument's evaluation or pin check.
Prior art: Python sitecustomize/PYTHONPATH; ENG1 seal and run-root guards (2026).
Taken: existing tools; new: selecting the new seal/order and restricting outputs
to A2/audit. Activated explicitly for this CPU audit and inherited by children.
"""
import os
if os.environ.get('ENG1_A2_AUDIT_ROUTING') == '1':
    from pathlib import Path
    import evidence
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('ENG1_AUDIT') == '1'
    evidence.SEAL_NAME = 'staging_seal_a2.json'
    evidence.ORDER = 'orders/ENG1_A2_REV2_FIXES.md'
    def a2_guard(runs):
        runs = Path(runs).resolve()
        allowed = evidence.ROOT / 'artifacts/eng1/a2/audit'
        if runs != allowed:
            raise ValueError('A2 audit RUNS must equal ' + str(allowed))
        return runs
    evidence.guard = a2_guard
