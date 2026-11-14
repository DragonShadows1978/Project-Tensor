#!/usr/bin/env python3
"""R2-F01 mechanical completion. Prior art: REV2 (2026) runtime read census,
PEP 578 (Dower 2019); A1 source-reader selectors. Taken: REV2's six missing
selectors and observed paths; new here: append them and publish the A2 table.
"""
import json
from pathlib import Path

A = Path(__file__).resolve().parent
R = A.parents[2]
T = R / 'artifacts/eng1/tool'
V = R / 'artifacts/eng1/rev2'
rows = json.loads((T / 'source_reading_tests.json').read_text())
assert len(rows) == 16
finding = json.loads((V / 'findings.json').read_text())['findings'][0]
missing_ids = finding['missing_ids']
missing = {i.split('[', 1)[0] for i in missing_ids}
observed = {r['test_id']: r for r in json.loads((V / 'q2_classify_reads.json').read_text())['rows']}
via = {
    'test_compiled_d96_flag_device_and_scalar_guards': 'common.load_runtime -> verify_sources (scripts/apa_sp3_common.py:181-182); may refuse before later reads',
    'test_a4_calibration_uses_actual_ppl_fraction_receipt': 'scripts/apa_sp4g_common.py: helper-mediated hashes of 14 script files',
    'test_a5_completed_RED_accepted_without_waiving_gate': 'scripts/apa_sp4g_common.py: helper-mediated hashes of 5 script files',
}
for selector in sorted(missing):
    name = selector.split('::')[1]
    rows.append(dict(test_id=selector, paths=observed[selector]['observed_source_reads'],
        via=via.get(name, "scripts/apa_sp4g_a5_model.py:40: source_sha256=sha(R/'tensor_cuda/src/kernels.cu')"),
        classification='source_path_reader', comparison='excluded on both arms',
        execution='candidate regression only', exact_ids=[i for i in missing_ids if i.split('[', 1)[0] == selector]))
assert len(rows) == 22 and len({r['test_id'] for r in rows}) == 22
(T / 'source_reading_tests.json').write_text(json.dumps(rows, indent=2) + '\n')
table = ['# Candidate source-path readers excluded from arm comparison — A2', '',
    'Supersedes ../a1/SOURCE_READING_TESTS.md, which is unchanged. The original sixteen selectors and paths are retained; six selectors (seven IDs) are added from REV2 R2-F01. All 22 selectors (23 exact IDs) run on the candidate; paired comparison excludes them on both arms.', '',
    'Paths are explicit reads or expanded provenance source maps; a pin failure can stop before later reads. Imports, traceback rendering, temporary tree walks and receipt/order reads are outside this class. Runtime coverage is recorded separately in q2_classify_reads.json; CPU observation cannot establish reads after a first failing CUDA call.', '',
    '| Exact ID(s) | Source paths / read site |', '|---|---|']
for r in rows:
    ids = ', '.join('`' + i + '`' for i in r['exact_ids'])
    paths = ', '.join('`' + p + '`' for p in r['paths'])
    table.append('| ' + ids + ' | ' + r['via'] + '; ' + paths + ' |')
table += ['', 'Prior art: REV2 (2026) q2_source_readers.py/q2_classify_reads.py, PEP 578 audit hooks (Dower, 2019), and A1 selector/path table. Taken: existing classification and the review’s exact missing IDs/paths; new here: completion and A2 publication. External literature unverified — lead to check: PEP 578 Dower 2019.']
(A / 'SOURCE_READING_TESTS.md').write_text('\n'.join(table) + '\n')
(A / 'added_source_readers.json').write_text(json.dumps(rows[16:], indent=2) + '\n')
print('source-reader selectors:', len(rows), 'exact IDs:', sum(len(r['exact_ids']) for r in rows))
