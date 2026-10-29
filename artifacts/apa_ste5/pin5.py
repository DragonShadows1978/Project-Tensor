#!/usr/bin/env python3
"""Immutable input hashes before gates; not a new acceptance registration.

Prior art: NIST FIPS 180-4 (2015), SHA-256; STE4 input pinning (2026).
Unverified - lead to check. Lead's STE5 order owns every threshold.
"""
from datetime import datetime,timezone
import common as c


def pin():
    c.cpu_environment()
    files=[c.HERE/p for p in ('common.py','capture5.py','verify5.py','reference5.py','timing5.py',
        'stage5.py','bundle5.py','pin5.py','tests/test_verify5.py','GPU_COMMANDS.sh','STAGING.json','OPERAND_CENSUS.md')]
    files += [c.ROOT/p for p in ('orders/APA_DK_STE5.md','orders/APA_DK_STE4.md',
        'tensor_cuda/tensor_cuda/__init__.py','tensor_cuda/src/kernels.cu','tensor_cuda/src/ops.cpp','tensor_cuda/src/bindings.cpp',
        'artifacts/apa_ste1/FINAL_REPORT.md','artifacts/apa_ste1/gpu_gate.py','artifacts/apa_ste1/reference.py',
        'artifacts/apa_ste1/INPUTS.json','artifacts/apa_ste_rev1/scripts/oracle_lib.py')]
    files += [c.fixture(name) for name in c.SHAPES]
    files += [p/name for p in (c.ARM_PACKAGE,c.BASE_PACKAGE) for name in
             ('__init__.py',c.SO_NAME,'apa_sp2.py','functional.py','nn.py','optim.py','quant.py','quantization/__init__.py','quantization/affine.py')]
    def key(p):
        try: return str(p.relative_to(c.ROOT))
        except ValueError: return str(p)
    c.put(c.HERE/'INPUT_PINS.json',{'pinned_utc':datetime.now(timezone.utc).isoformat(),
        'order':'orders/APA_DK_STE5.md','seat_registers_no_gate':True,
        'threshold_authority':'immutable lead order; no threshold changes allowed after data',
        'N':3,'variants':['g1','h'],'G-G5':{'cosine_min':.995,'slope_min':.98,'slope_max':1.02,'relative_L2_slack':.01},
        'GPU_data':'no STE5 captures; CPU-only seat; historical data used only for reference parity and disclosure',
        'pins':{key(p):c.digest(p) for p in files},
        'mutable_outputs_not_pinned':['ENGINE_MANIFEST.json','BUNDLE.json','FINAL_REPORT.md','LEDGER.md','receipts/','lead/'],
        'amendments':'immutable; any amendment is a separate file'})


if __name__=='__main__': pin()
