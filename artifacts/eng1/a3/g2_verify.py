#!/usr/bin/env python3
"""A3 G2 receipt: list each post-device step the CPU audit drove, with a terminal label, read from the audit root.
Run under the A3 audit routing (PYTHONPATH=a3:tool, ENG1_A3_AUDIT_ROUTING=1, ENG1_AUDIT=1, CUDA hidden) so the
unchanged evidence.recompute_line re-derives the parent verdict with the A3 seal.
Prior art: GN1/ENG1 L45 parent-after-child receipts (2026); a2/verify.py. Ours: only the step listing.
No algorithm; no prior art beyond these known to me.
"""
import datetime
import json
import os
from pathlib import Path

import evidence

A3 = Path(__file__).resolve().parent
runs = A3 / 'audit'
seal_sha = evidence.sha(evidence.TOOL / evidence.SEAL_NAME)
assert evidence.SEAL_NAME == 'staging_seal_a3.json'
seal, start = evidence.verify_start(runs)          # step: START bound to the A3 seal and order
att = runs / 'candidate_eng1' / 'attempt-0001'
child = json.loads((att / 'child_result.json').read_text())
inv = json.loads((att / 'invocation.json').read_text())
parent = json.loads((att / 'parent_result.json').read_text())
origin = json.loads((att / 'origin_check.json').read_text())
forge = (att / 'forge.log').read_text().splitlines()
ts = lambda s: datetime.datetime.fromisoformat(s)
steps = []


def step(name, ok, detail):
    steps.append(dict(step=name, terminal='PASS' if ok else 'FAIL', detail=detail))


step('start_gpu_run.py: START.json', start['seal_sha256'] == seal_sha and start['audit'] is True
     and start['order_commit'] == seal['order_commit'], dict(seal_sha256=start['seal_sha256'], audit=start['audit']))
step('child: child_started.json marker', (att / 'child_started.json').is_file(), 'present')
step('child: origin_check.json (subprocess arm origin, engine pin)', origin['exit'] == 0 and
     origin['observed']['engine_sha256'] == seal['engine_sha256']['candidate'],
     dict(engine_sha256=origin['observed']['engine_sha256']))
step('child: invocation.json (seal, CPU device evidence)', inv['seal_sha256'] == seal_sha and inv['audit'] is True
     and inv['CUDA_VISIBLE_DEVICES'] == '' and inv['device_evidence']['query_error'] == 'CPU stand-in: engine not loaded',
     dict(seal_sha256=inv['seal_sha256'], CUDA_VISIBLE_DEVICES=inv['CUDA_VISIBLE_DEVICES']))
step('child: junit.xml', (att / 'junit.xml').is_file(), 'present')
step('child: child_result.json', child['seal_sha256'] == seal_sha and child['pytest_exit'] == 0
     and not child['collection_errors'],
     dict(pytest_exit=child['pytest_exit'], n_ids=len(child['collected_ids']), seal_sha256=child['seal_sha256']))
step('runner: forge.log row (stand-in)', len(forge) == 1 and forge[0].split('\t')[7:] == ['ran', '0'], forge)
step('parent: parent_result.json recorded after child exit', parent['passed'] is True and parent['exit'] == 0
     and parent['seal_sha256'] == seal_sha and parent['admission'] == 'admitted'
     and ts(parent['child_exited']) <= ts(parent['recorded']),
     {k: parent[k] for k in ('passed', 'exit', 'admission', 'forge_outcome', 'child_exited', 'recorded', 'seal_sha256')})
re = evidence.recompute_line(runs, 'candidate_eng1', seal, start)
step('evidence.recompute_line (A3 seal, evidence.py:88/245/348 seal reads)', re['passed'] and re['mode'] == 'cpu',
     dict(passed=re['passed'], mode=re['mode'], reasons=re['reasons']))
score = json.loads((A3 / 'g2_score.exit.json').read_text())
summary = json.loads((runs / 'G5_SUMMARY.json').read_text())
step('score_gpu.py on the audit root (expected exit 1: audit cannot establish GPU acceptance)',
     score['exit'] == 1 and 'audit run cannot establish GPU acceptance' in summary['errors']
     and summary['lines']['candidate_eng1']['passed'] is True,
     dict(exit=score['exit'], errors=summary['errors'], candidate_eng1_passed=summary['lines']['candidate_eng1']['passed']))
out = dict(audit_root=str(runs), seal_sha256=seal_sha, steps=steps, g2_pass=all(s['terminal'] == 'PASS' for s in steps))
with (A3 / 'G2_AUDIT_RECEIPT.json').open('x') as f:
    f.write(json.dumps(out, indent=1) + '\n')
for s in steps:
    print('%-4s %s' % (s['terminal'], s['step']))
print('g2_pass', out['g2_pass'])
