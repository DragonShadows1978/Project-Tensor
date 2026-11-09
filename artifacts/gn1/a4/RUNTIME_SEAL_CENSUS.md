# Runtime seal census

Collection: 534 unique origins. Final local/external import maps: 17 local and 537 external origins, excluding arm maps. The runtime union adds 29 origins, including 11 numpy.random and 5 numpy.testing origins. All sealed digests match at final verification. No unsealed modules occur in either honest full stand-in root or in the native audit line. The explicit unsealed hostile control imports a new scratch module and reports it without a veto.

| CPU stand-in suite | Module entries | IDs | Calls | Pytest exit | Setup errors |
|---|---|---|---|---|---|
| focus | 548 | 412 | 412 | 1 | 0 |
| mutation | 545 | 405 | 405 | 0 | 0 |
| regression | 562 | 468 | 453 | 1 | 15 |

The stand-in does not implement unrelated engine operations. Focus keeps 7 failures; regression keeps 33 call failures, 6 skips and 15 setup errors. These are import-census outcomes, not GPU regressions or a passing engine suite. NumPy testing calibration is explicit in each trace; unseen real-device paths remain unsealed and must be reviewed by the lead.

Added origins:

- `$HOME/.local/lib/python3.12/site-packages/numpy/ma/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/ma/core.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/ma/extras.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_bounded_integers.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_common.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_generator.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_mt19937.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_pcg64.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_philox.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_pickle.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_sfc64.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/bit_generator.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/mtrand.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/_private/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/_private/extbuild.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/_private/utils.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/overrides.py`
- `/usr/lib/python3.12/_sysconfigdata__x86_64-linux-gnu.py`
- `/usr/lib/python3.12/getpass.py`
- `/usr/lib/python3.12/lib-dynload/termios.cpython-312-x86_64-linux-gnu.so`
- `/usr/lib/python3.12/secrets.py`
- `/usr/lib/python3.12/sysconfig.py`
- `artifacts/gn1/a4/cpu_shim.py`
- `artifacts/gn1/a4/evidence.py`
- `artifacts/gn1/a4/run_gpu_tests.py`
- `artifacts/gn1/a4/runtime_evidence.py`
- `artifacts/gn1/a4/trace_runtime.py`
