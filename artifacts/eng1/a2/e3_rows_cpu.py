#!/usr/bin/env python3
"""CPU measurement-instrument check using all real E3 emulation operands.
Prior art: A1 arithmetic_receipts.py / REV2 R2-F02 (2026), NumPy finite masks
and array_equal(equal_nan=True). Taken: original operands/graphs, exact ordered
pattern formula; new: exercise the live row return expression on CPU without
calling GPU-only e3_measure setup. The instrument's got operand is explicitly
the CPU emulation, never represented as an engine measurement.
"""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

A = Path(__file__).resolve().parent
R = A.parents[2]
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
sys.path.insert(0, str(R / 'artifacts/eng1/tool/arms/candidate/tensor_cuda'))
source = R / 'tensor_cuda/tests/test_eng1.py'
spec = importlib.util.spec_from_file_location('eng1_e3_cpu', source)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
tree = ast.parse(source.read_text())
measure = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'e3_measure')
ret = next(n for n in measure.body if isinstance(n, ast.Return))
row_code = compile(ast.Expression(ret.value), str(source), 'eval')
pattern = next(k.value for k in ret.value.keywords if k.arg == 'nonfinite_pattern_equal')
pattern_code = compile(ast.Expression(pattern), str(source), 'eval')
rows = []
with np.errstate(over='ignore', invalid='ignore', under='ignore'):
    for cls in M.E3_CLASSES:
        for dtype in M.E3_DTYPES:
            for case in M.E3_CASES:
                x64 = M.e3_round(M.e3_inputs(cls, case), dtype)
                g64 = M.e3_round(M._e3_pattern(x64.shape, 1.1), dtype)
                C, _ = M._affine_shape(cls, x64.shape)
                w64 = M.e3_round(1 + .1 * np.arange(C), dtype)
                b64 = None if cls == 'RMSNorm' else M.e3_round(.05 * np.arange(C) - .1, dtype)
                refs = M.e3_reference(cls, x64, w64, b64, g64)
                emus = M.e3_emulation(cls, dtype, x64, w64, b64, g64)
                for index, direction in enumerate(('forward', 'backward')):
                    got, emu, ref = emus[index].copy(), emus[index], refs[index]
                    finite = bool(np.all(np.isfinite(got)))
                    scale = float(np.max(np.abs(ref)))
                    tol = M.e3_tolerance(cls, direction, dtype, scale)
                    err = float(np.max(np.abs(np.where(np.isfinite(got), got, np.inf) - ref)))
                    axes = (2, 3, 4) if cls == 'GroupNorm' else (-1,) if cls == 'RMSNorm' else M._e3_axes(cls, x64.shape)
                    xr = x64.reshape(x64.shape[0], 2, -1, *x64.shape[2:]) if cls == 'GroupNorm' else x64
                    stat = (xr * xr).mean(axis=axes) if cls == 'RMSNorm' else xr.var(axis=axes)
                    row = eval(row_code, M.__dict__, locals())
                    assert row['nonfinite_pattern_equal'] is True
                    want = 0.0 if np.any(np.isfinite(emu)) else None
                    assert row['engine_distance_from_emulation_finite'] == want
                    # Retain the actual emitted row fields, but explicitly identify
                    # the non-engine input and JSON-safe legacy infinite error.
                    if not math.isfinite(row['max_abs_err']):
                        row['max_abs_err'] = None
                        row['max_abs_err_status'] = 'legacy computed +inf; JSON represented as null'
                    row['engine_status'] = 'not executed; got is CPU emulation for row-expression instrument check only'
                    rows.append(row)
nonfinite = [r for r in rows if r['emulation_nonfinite']]
assert len(rows) == 40 and len(nonfinite) == 4
assert {r['cls']: r['emulation_nonfinite'] for r in nonfinite} == {
    'LayerNorm': 64, 'RMSNorm': 64, 'BatchNorm1D': 64, 'InstanceNorm2D': 96}
# Branch checks: mixed finite pairs; absent finite pairs; sign, NaN and position.
cases = [
    ('mixed_same_pattern', [1, np.inf, np.nan, -np.inf, 7], [4, np.inf, np.nan, -np.inf, 5], True, 3.0),
    ('inf_sign', [1, np.inf], [4, -np.inf], False, 3.0),
    ('inf_vs_nan', [np.inf], [np.nan], False, None),
    ('different_positions', [np.nan, 1], [1, np.nan], False, None),
    ('all_nonfinite_match', [np.nan, np.inf, -np.inf], [np.nan, np.inf, -np.inf], True, None),
    ('all_finite', [1, 3], [4, 1], True, 3.0),
    ('empty', [], [], True, None),
]
checks = []
for name, aa, bb, expected_pattern, expected_distance in cases:
    got, emu = np.array(aa, dtype=np.float64), np.array(bb, dtype=np.float64)
    equal = eval(pattern_code, M.__dict__, dict(got=got, emu=emu))
    distance = M.e3_distance(got, emu, finite_only=True)
    assert equal == expected_pattern and distance == expected_distance
    if len(got) and not np.all(np.isfinite(got) & np.isfinite(emu)):
        assert M.e3_distance(got, emu) is None
    checks.append(dict(case=name, nonfinite_pattern_equal=equal, finite_distance=distance, passed=True))
out = dict(evidence_class='CPU measurement-instrument check, no engine E3 run',
           row_expression_source='tensor_cuda/tests/test_eng1.py:e3_measure return',
           row_count=len(rows), nonfinite_row_count=len(nonfinite), rows=rows, branch_checks=checks)
(A / 'e3_rows_cpu.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print('E3 CPU instrument: 40 real emulation rows, 4 nonfinite rows, 7 branch checks pass; no engine E3 measurement')
