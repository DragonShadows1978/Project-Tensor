"""APA-P1-F1 L60: BOUNDARY.json from the LF3/LF4 results on Build F and the BND line on Build B.

For each refusal added this round: the input, Build F's refusal (exception text and kernel
launches during it), and Build B's behaviour on the same input, measured on the BND line
(fault / zero output / silent acceptance / the same refusal). Python-level refusals that
the GPU lines do not exercise on Build B are classified from Build B's code (P1-D arm B_v2),
and say so. Prior art: none; receipt assembly. CPU only.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'artifacts/apa_p1f1/gpu'


def classify_b(rec):
    if rec is None:
        return 'not measured'
    raised = rec.get('raised') or rec.get('refusal') or ''
    if rec.get('exception') and 'misaligned' in rec.get('exception', ''):
        return 'fault: ' + rec['exception'][:120] + '; context ' + str(rec.get('context_after_exception'))
    if raised in ('DID NOT RAISE', None, '') or rec.get('note') == 'DID NOT RAISE':
        return 'silent acceptance (no exception)'
    if raised.startswith('TypeError') and 'delta' in raised:
        return 'API mismatch: Build B apa_eb2b has no delta/carry arguments (' + raised[:100] + ')'
    launches = rec.get('launches') if 'launches' in rec else rec.get('launches_during_refusal')
    return f'raised {raised[:140]} (kernel launches during the call: {launches})'


def main():
    f3 = {r['row']: r for r in json.loads((GPU / 'LF3/RESULT.json').read_text())['rows']}
    b3 = {r['row']: r for r in json.loads((GPU / 'BND/LF3/RESULT_B.json').read_text())['rows']}
    f4 = {c['id']: c for c in json.loads((GPU / 'LF4/RESULT.json').read_text())['checks']}
    b4 = {c['id']: c for c in json.loads((GPU / 'BND/LF4/RESULT_B.json').read_text())['checks']}
    rows = []
    for row, r in f3.items():
        if row.startswith('dense') or row.startswith('r2public'):
            continue
        rows.append(dict(rule='R2', input=f'{row}: operand {r.get("operand")} re-imported with a {r.get("byte_offset")}-byte offset',
                         build_F=f'{r.get("refusal")} (launches during it: {r.get("launches_during_refusal")})',
                         build_B=classify_b(b3.get(row)),
                         build_B_detail={k: b3.get(row, {}).get(k) for k in ('refusal', 'exception', 'context_after_exception',
                                                                              'launches_during_refusal', 'child_exit')}))
    for cid, c in f4.items():
        if not ('raises' in cid or 'int4_stats' in cid):
            continue
        b = b4.get(cid)
        rule = ('R6' if 'int4_stats' in cid else 'R11' if ('empty' in cid or 'dots' in cid or 'carry' in cid or 'TC_APA_SP' in cid)
                else 'R7' if 'knob' in cid else 'R4' if 'EB2b' in cid else 'R11')
        bdesc = classify_b(b)
        if 'int4_stats' in cid and b is not None:
            bdesc = f'child exit {b.get("exit")} ({"SIGFPE, process killed" if b.get("exit") == -8 else b.get("result")})'
        rows.append(dict(rule=rule, input=cid, build_F=(c.get('raised') or c.get('result')) and
                         f'{c.get("raised") or c.get("result")} (launches: {c.get("launches")})', build_B=bdesc))
    code_read = [
        dict(rule='R4', input='apa_eb2b(...) without delta or carry', build_F='TypeError (required keyword arguments)',
             build_B='silent acceptance: delta 8.1 and row carry hard-wired at the launch site (eb2b.cuh:168 at f01201cf); code reading'),
        dict(rule='R3', input='apa_eb3 / apa_eb4 without carry', build_F='carry="partition" (default; not a refusal)',
             build_B='carry="row" by default; code reading of the B_v2 package and bindings'),
        dict(rule='R8', input='ApaSpq1Workspace with a different K of the same row count, or an in-place-edited K',
             build_F='ValueError (exact-K identity: source object, generation, storage revision)',
             build_B='silent acceptance (valid_rows == k.shape[2] only); code reading, REV1 F08; CPU test m11 restores it and fails'),
        dict(rule='R9', input='packed API on a module built without TC_APA_PACKED',
             build_F='RuntimeError "...built without TC_APA_PACKED" before any allocation (CPU test on the production module)',
             build_B='ApaSpq1Workspace allocated, then AttributeError on None; apa_eb2b/apa_eb3/apa_eb4 absent; REV1 F09'),
        dict(rule='R9', input='packed build whose second initializer fails to load',
             build_F='ImportError at package import naming PyInit__tensor_cuda_spo2 and the real cause (CPU test, simulated)',
             build_B='ImportError swallowed: _spo2 = None, the packed API silently absent; REV1 F09'),
        dict(rule='R12', input='eb3.cuh compiled with CUDA_API_PER_THREAD_DEFAULT_STREAM',
             build_F='#error naming the single-stream ticket contract (CPU test, preprocessing)',
             build_B='compiles; tickets shared across streams with no refusal; REV1 F12'),
    ]
    rows.extend(code_read)
    out = ROOT / 'artifacts/apa_p1f1/BOUNDARY.json'
    with out.open('x') as f:
        json.dump(dict(source=dict(F_LF3='artifacts/apa_p1f1/gpu/LF3/RESULT.json', F_LF4='artifacts/apa_p1f1/gpu/LF4/RESULT.json',
                                   B_LF3='artifacts/apa_p1f1/gpu/BND/LF3/RESULT_B.json',
                                   B_LF4='artifacts/apa_p1f1/gpu/BND/LF4/RESULT_B.json'), rows=rows), f, indent=2)
        f.write('\n')
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
