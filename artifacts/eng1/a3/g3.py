#!/usr/bin/env python3
"""A3 G3 scope, without git (the dispatch forbids every git command; the lead runs the literal
`git diff 97edd1fd -- . ':!artifacts/eng1/a3' ':!orders' ':!logs'`).
Inputs: worktree_before.json (taken before any A3 edit) and worktree_after.json (taken now), both by snapshot.py.
Checks (whole-file sha256 of bytes, no masks; L09):
  1. before -> after: every added/changed/removed path outside artifacts/eng1/a3/ is in the allowed set
     (tool/evidence.py changed, tool/staging_seal_a3.json added, tool/gpu_runs/a3_dry_* added).
  2. every forbidden file (order list) is byte-identical before -> after.
  3. dispatch state vs the A2-close receipts that 97edd1fd carries (no git): staging_seal_a2.json tool/arm/
     registration/engine maps, a2/protected_after.json, a2/tensor_cuda_after.json, a2/FILES_WRITTEN.json.
  4. engine pins as recorded in the seals (engine_sha256) and on disk.
Prior art: SHA-256 (NIST FIPS 180-4, 2015); manifest diffing as in mtree(8) (BSD; year unverified -- lead to check).
Ours: only the allowed/forbidden classification for this order.
"""
import fnmatch
import hashlib
import json
from pathlib import Path

A3 = Path(__file__).resolve().parent
ROOT = A3.parents[2]
E = 'artifacts/eng1/'
T = E + 'tool/'
before = json.loads((A3 / 'worktree_before.json').read_text())['files']
after = json.loads((A3 / 'worktree_after.json').read_text())['files']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def excluded(p):
    return p.startswith(E + 'a3/') or p.startswith('orders/') or p.startswith('logs/')


def allowed(p):
    return p == T + 'evidence.py' or p == T + 'staging_seal_a3.json' or fnmatch.fnmatch(p, T + 'gpu_runs/a3_dry_*/*')


added = sorted(p for p in after if p not in before and not excluded(p))
removed = sorted(p for p in before if p not in after and not excluded(p))
changed = sorted(p for p in before if p in after and before[p] != after[p] and not excluded(p))
outside = [p for p in added + removed + changed if not allowed(p)]
forbidden_patterns = [T + n for n in ('score_gpu.py', 'run_gpu_tests.py', 'run_gpu_line.py', 'runtime_evidence.py',
                                      'device_evidence.py', 'gpu_lines.json', 'STAGED_GPU_COMMANDS.sh',
                                      'ids_*', 'collect_*', 'arms/*', 'staging_seal.json', 'staging_seal_a1.json',
                                      'staging_seal_a1_preliminary.json', 'staging_seal_a2.json', 'stage.py',
                                      'start_gpu_run.py')] + \
    ['tensor_cuda/*', E + 'a1/*', E + 'a2/*', E + 'rev1/*', E + 'rev2/*', E + 'lead/*', E + 'mutants/*']
forbidden = sorted(p for p in before if any(fnmatch.fnmatch(p, pat) for pat in forbidden_patterns))
forbidden_moved = [p for p in forbidden if after.get(p) != before[p]]
# 3. dispatch (before) state vs A2-close receipts.
a2 = json.loads((ROOT / T / 'staging_seal_a2.json').read_text())
a3 = json.loads((ROOT / T / 'staging_seal_a3.json').read_text())
vs_a2 = dict(
    tool_files=[n for n, d in a2['tool_files'].items() if before.get(T + n) != d],
    registration_files=[n for n, d in a2['registration_files'].items() if before.get(n) != d],
    arms=[a + '/' + n for a, m in a2['arms'].items() for n, d in m.items() if before.get(T + 'arms/' + a + '/' + n) != d],
    protected_after=[n for n, d in json.loads((ROOT / E / 'a2/protected_after.json').read_text()).items()
                     if before.get(n) != d],
    tensor_cuda_after=[n for n, d in json.loads((ROOT / E / 'a2/tensor_cuda_after.json').read_text()).items()
                       if before.get(n) != d],
    a2_files_written=[r['path'] for r in json.loads((ROOT / E / 'a2/FILES_WRITTEN.json').read_text())['worktree_files']
                      if r['sha256'] is not None and before.get(r['path']) != r['sha256']])
tc_extra = sorted(p for p in before if p.startswith('tensor_cuda/') and p not in
                  json.loads((ROOT / E / 'a2/tensor_cuda_after.json').read_text()))
mod = 'tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
pins = {s: {arm: dict(seal=seal['engine_sha256'][arm], disk=sha(ROOT / T / 'arms' / arm / mod))
            for arm in ('baseline', 'candidate')}
        for s, seal in (('a1', json.loads((ROOT / T / 'staging_seal_a1.json').read_text())), ('a2', a2), ('a3', a3))}
pins_ok = all(v['baseline']['seal'].startswith('73ededa0') and v['baseline']['seal'].endswith('523e')
              and v['candidate']['seal'].startswith('3e6d5805') and v['candidate']['seal'].endswith('ad53')
              and v['baseline']['seal'] == v['baseline']['disk'] and v['candidate']['seal'] == v['candidate']['disk']
              for v in pins.values())
out = dict(n_before=len(before), n_after=len(after), added=added, removed=removed, changed=changed,
           outside_allowed=outside, n_forbidden_checked=len(forbidden), forbidden_moved=forbidden_moved,
           vs_a2_close_mismatches=vs_a2, tensor_cuda_files_not_in_a2_after=tc_extra, engine_pins=pins, pins_ok=pins_ok,
           a3_vs_a2_seal=dict(arms_equal=a3['arms'] == a2['arms'], engine_equal=a3['engine_sha256'] == a2['engine_sha256'],
                              tool_files_changed=sorted(n for n in a3['tool_files'] if a3['tool_files'][n] != a2['tool_files'][n])),
           added_summary=dict(evidence_py_changed=changed == [T + 'evidence.py'],
                              seal_added=T + 'staging_seal_a3.json' in added,
                              gpu_runs_files=len([p for p in added if p.startswith(T + 'gpu_runs/')])))
# Repair 1 (L56 b): the 'before' snapshot predates the A3 edit, so it must equal the A2 seal everywhere;
# r1 wrongly expected evidence.py to differ (receipt G3_SCOPE.json, g3_pass false with every check clean).
expected_a2_tool_mismatch = []
out['g3_pass'] = (not outside and not forbidden_moved and changed == [T + 'evidence.py'] and not removed
                  and vs_a2['tool_files'] == expected_a2_tool_mismatch
                  and not any(v for k, v in vs_a2.items() if k != 'tool_files') and pins_ok)
with (A3 / 'G3_SCOPE_r2.json').open('x') as f:
    f.write(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: out[k] for k in ('n_before', 'n_after', 'changed', 'removed', 'outside_allowed',
                                      'n_forbidden_checked', 'forbidden_moved', 'vs_a2_close_mismatches',
                                      'tensor_cuda_files_not_in_a2_after', 'pins_ok', 'a3_vs_a2_seal',
                                      'added_summary', 'g3_pass')}, indent=1))
print(json.dumps(pins, indent=1))
