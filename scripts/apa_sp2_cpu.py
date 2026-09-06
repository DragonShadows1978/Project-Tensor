#!/usr/bin/env python3
"""CPU G1 wrapper, importing unchanged SP1 tests against this seat's new build."""
import json
import subprocess
import sys
import time
from apa_sp2_common import ART, ROOT, load_runtime, registration, sha, write_new


def main():
    registration();tc=load_runtime()
    # Only test-runtime dependency injection: no old harness or receipt edits.
    import apa_sp1_gpu
    apa_sp1_gpu.load_runtime=load_runtime
    import pytest
    paths=['test_apa_sp1_reference.py','test_apa_sp1_host.py','test_apa_sp1_harness.py',
           'test_apa_sp1_1.py','test_apa_sp2.py']
    result=pytest.main(['-q','-p','no:cacheprovider','--tb=short']+[str(ROOT/'tensor_cuda/tests'/p) for p in paths])
    record=dict(status='PASS' if result==0 else 'FAIL',returncode=int(result),
                evidence_class='CPU suite run and host extension; not GPU execution',
                registration_sha256=sha(ART/'registration.json'),
                build_manifest_sha256=sha(ART/'build/manifest.json'),
                runtime_module=str(tc._C.__file__),tests={p:sha(ROOT/'tensor_cuda/tests'/p) for p in paths},
                runtime_adapter='unchanged SP1 host tests injected with SP2 build loader; original harness remains pinned')
    write_new(ART/f'g1_{time.time_ns()}.json',record)
    if result!=0:raise SystemExit(result)
    from apa_sp2_mutations import run
    run()


if __name__=='__main__':main()
