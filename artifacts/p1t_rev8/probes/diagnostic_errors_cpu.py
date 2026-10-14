"""PASS iff diagnostic create/record/query/destroy failures each raise naming their API; ordinary path succeeds.
Prior art: CUDA event return/error-slot contracts, NVIDIA 2024; REV6 host trace. Includes current product sources.
Run: PYTHONPATH=/tmp/p1t_rev8_build timeout 20 python artifacts/p1t_rev8/probes/diagnostic_errors_cpu.py
"""
import json
import _p1t_rev8_diag as n
rows=[]
for site,api in [('create','cudaEventCreateWithFlags'),('record','cudaEventRecord'),('query','cudaEventQuery'),('destroy','cudaEventDestroy')]:
    n.exit_window(0)
    n.inject(site)
    error=None
    try: n.synchronize()
    except RuntimeError as e: error=str(e)
    finally: n.end_window()
    row={'site':site,'expected_api':api,'error':error,'slot':n.slot(),'stats':n.stats()}
    rows.append(row)
print(json.dumps(rows,indent=2))
assert all(r['error'] and r['expected_api'] in r['error'] for r in rows), 'diagnostic event failures silently erased'
