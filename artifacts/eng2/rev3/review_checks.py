"""Independent ENG2-REV3 receipt checks; no product or historical receipt writes.

Prior art: pytest project test collection; fnmatch shell-glob matching (Python
project) and mutation killing sets (DeMillo/Lipton/Sayward 1978, unverified --
lead to check); SHA-256 manifests (NIST FIPS 180-4 2015, unverified -- lead to
check). Reused: collection, pattern matching and hashing. Review-specific:
binding retired IDs to registered killers and manifest-verified G5 outcomes.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/eng2/rev3'
SCRATCH = Path(os.environ['TMPDIR'])
TOOL = ROOT / 'artifacts/eng2/tool'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def command(argv, **kw):
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240, **kw)
    record = dict(argv=argv, cwd=str(kw.get('cwd', ROOT)), exit=r.returncode,
                  stdout=r.stdout.decode(errors='replace'), stderr=r.stderr.decode(errors='replace'))
    with (OUT / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    return r


def q1():
    diff = command(['git', 'diff', 'e2f166b8', '10738fe3', '--', 'tensor_cuda'])
    (OUT / 'source.diff').write_bytes(diff.stdout)
    assert diff.returncode == 0
    names = command(['git', 'diff', '--name-only', 'e2f166b8', '10738fe3', '--', 'tensor_cuda'])
    assert names.stdout.decode().splitlines() == ['tensor_cuda/tests/test_eng1.py']
    old_pins = command(['git', 'show', 'e2f166b8:artifacts/eng2/L38_PINS.json'])
    pins = json.loads(old_pins.stdout)['expected_changes']
    print('OLD PINS SHAPE', type(pins).__name__, str(pins)[:350])
    results = {}
    for arm in ['baseline', 'candidate']:
        for commit in ['e2f166b8', '10738fe3']:
            folder = SCRATCH / 'collect' / arm / commit
            folder.mkdir(parents=True, exist_ok=True)
            src = command(['git', 'show', commit + ':tensor_cuda/tests/test_eng1.py'])
            assert src.returncode == 0
            test = folder / 'test_eng1.py'
            test.write_bytes(src.stdout)
            env = dict(os.environ, PYTHONPATH=str(TOOL / 'arms' / arm / 'tensor_cuda'),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
            argv = [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider',
                    '--confcutdir=' + str(folder), '--rootdir=' + str(folder), str(test)]
            r = command(argv, env=env, cwd=folder)
            (OUT / f'collect_{arm}_{commit}.txt').write_bytes(r.stdout + r.stderr)
            assert r.returncode == 0, r.stdout.decode() + r.stderr.decode()
            ids = sorted(line.strip() for line in r.stdout.decode().splitlines()
                         if line.startswith('test_eng1.py::'))
            assert ids and len(ids) == len(set(ids))
            (OUT / f'ids_{arm}_{commit}.txt').write_text('\n'.join(ids) + '\n')
            results[arm + '_' + commit] = dict(count=len(ids), source_sha256=sha(src.stdout), ids=ids)
    pin_ids = {x.split('::')[-1] for x in pins}
    for arm in ['baseline', 'candidate']:
        before = set(results[arm + '_e2f166b8']['ids'])
        after = set(results[arm + '_10738fe3']['ids'])
        removed = sorted(before - after)
        assert not after - before
        assert {x.split('::')[-1] for x in removed} == pin_ids
        assert len(removed) == 10
        results[arm + '_diff'] = dict(removed=removed, added=sorted(after - before))
    assert results['baseline_diff'] == results['candidate_diff']
    save('Q1_ids.json', results)
    print('Q1:', {k:v['count'] for k,v in results.items() if 'count' in v}, 'exact ten pins removed, none added')


class VerifiedRun:
    def __init__(self, side):
        self.side = side
        self.root = (ROOT if side == 'eng2' else ROOT.parent / 'tensor_fork_apa_wt_eng1')
        self.run = self.root / f'artifacts/{side}/tool/gpu_runs/g5_1114'
        self.manifest_path = ROOT / f'artifacts/{side}/lead/g5_1114/{side}_g5_MANIFEST.sha256'
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = sha(raw)
        self.entries = {}
        self.verified = []
        for line in raw.decode().splitlines():
            digest, path = line.split(None, 1)
            self.entries[path.strip().removeprefix('*').removeprefix('./')] = digest

    def read(self, rel):
        rel = str(rel)
        expected = self.entries[rel]
        data = (self.run / rel).read_bytes()
        actual = sha(data)
        assert actual == expected, (self.side, rel, expected, actual)
        self.verified.append(dict(path=rel, sha256=actual, manifest_match=True))
        return data


def pattern_matches(pattern, node):
    # Registry [] delimit pytest parameters literally; only * is a wildcard.
    # Equivalent to the registered fnmatch(p.replace('[','[[]')) semantics,
    # expressed as an independently constructed anchored regular expression.
    return re.fullmatch('.*'.join(re.escape(x) for x in pattern.split('*')), node.split('::')[-1]) is not None


def q2():
    q1data = json.loads((OUT / 'Q1_ids.json').read_text())
    removed = q1data['baseline_diff']['removed']
    kept = q1data['baseline_10738fe3']['ids']
    result = {'runs':{}, 'rows':[]}
    for side in ['eng1', 'eng2']:
        vr = VerifiedRun(side)
        mutants = json.loads((ROOT / f'artifacts/{side}/mutants/mutants.json').read_text())
        if side == 'eng1':
            mutants += json.loads((ROOT / 'artifacts/eng1/a1/MUTANTS.json').read_text())
        assert len(mutants) == (16 if side == 'eng1' else 15)
        for m in mutants:
            parent_paths = sorted(x for x in vr.entries if x.startswith(m['id'] + '/') and x.endswith('/parent_result.json'))
            assert parent_paths
            candidates = []
            for pp in parent_paths:
                parent = json.loads(vr.read(pp))
                candidates.append((pp, parent))
            admitted = [(p,d) for p,d in candidates if d.get('passed')]
            assert len(admitted) == 1, (m['id'], [(p, list(d), d.get('passed')) for p,d in candidates])
            pp, parent = admitted[0]
            folder = pp.rsplit('/',1)[0]
            child = json.loads(vr.read(folder + '/child_result.json'))
            xml = ET.fromstring(vr.read(folder + '/junit.xml'))
            xml_outcomes = {}
            for tc in xml.iter('testcase'):
                assert not any(c.tag == 'error' for c in tc), (m['id'], tc.attrib)
                state = 'failed' if tc.find('failure') is not None else 'skipped' if tc.find('skipped') is not None else 'passed'
                xml_outcomes[tc.attrib['name']] = state
            child_outcomes = {}
            for report in child['reports']:
                if report['when'] == 'call':
                    child_outcomes[report['id'].split('::')[-1]] = report['outcome']
            assert xml_outcomes == child_outcomes, (m['id'], 'JUnit/child mismatch')
            patterns = [x.split('::')[-1] for x in m['killers']]
            pattern_rows = []
            for pat in patterns:
                old_matches = [x for x in removed if pattern_matches(pat, x)]
                remaining = [x for x in xml_outcomes if pattern_matches(pat, x) and x not in {y.split('::')[-1] for y in removed}]
                pattern_rows.append(dict(pattern=pat, removed_matches=old_matches, remaining_matches=remaining,
                                         only_removed=bool(old_matches) and not remaining))
            surviving_failures = sorted(x for x,s in xml_outcomes.items() if s == 'failed'
                                        and x not in {y.split('::')[-1] for y in removed}
                                        and any(pattern_matches(p,x) for p in patterns))
            assert surviving_failures and not any(p['only_removed'] for p in pattern_rows), m['id']
            registered_actual = [x.split('::')[-1] for x in parent.get('actual_killers', [])]
            assert registered_actual and set(registered_actual) <= set(surviving_failures), m['id']
            row = dict(side=side, mutant=m['id'], parent=pp, patterns=pattern_rows,
                       remaining_failed_killers=surviving_failures, parent_actual_killers=registered_actual,
                       junit_child_agree=True, passed=True)
            result['rows'].append(row)
            print(side, m['id'], 'patterns', len(patterns), 'removed matches', sum(len(x['removed_matches']) for x in pattern_rows),
                  'remaining failed',len(surviving_failures), surviving_failures[0])
        result['runs'][side] = dict(root=str(vr.run), manifest=str(vr.manifest_path),
                                   manifest_sha256=vr.manifest_sha256, manifest_entries=len(vr.entries), verified=vr.verified)
    # Positive controls for literal brackets and wildcard matching over the real retired IDs.
    result['matcher_controls'] = {p:sum(pattern_matches(p,x) for x in removed)
                                  for p in ['test_e2_pow_grad[float64-*]', 'test_e2_pow_grad_bits[float64-p+3]', '*float64*']}
    assert list(result['matcher_controls'].values()) == [5, 1, 10]
    result['all_passed'] = len(result['rows']) == 31 and all(x['passed'] for x in result['rows'])
    save('Q2_mutants.json', result)


def q4():
    # The original module has no version CLI. Ask --version before use, record
    # the no-output response and identify the copied tool by its source digest.
    version = command([sys.executable, str(TOOL / 'evidence.py'), '--version'])
    assert version.returncode == 0
    sys.path.insert(0, str(TOOL))
    import evidence
    seal = evidence.verify_seal()
    for arm in seal['arms']:
        evidence.verify_arm(arm, seal)
    digest = sha((TOOL / 'staging_seal.json').read_bytes())
    assert digest == 'b424bdcfa0d01acb84cfe71eec21dc4597a40c2ebd77d99c9143aee4f17df826'
    lines = json.loads((TOOL / 'gpu_lines.json').read_text())
    dry = json.loads((ROOT / 'artifacts/eng2/a3/G4_DRY_dry_4.json').read_text())
    rows = dry['rows']
    assert {r['label'] for r in rows} == {r['label'] for r in lines} and len(rows) == 21
    checked = []
    for row in rows:
        line = next(x for x in lines if x['label'] == row['label'])
        s1 = row['s1_dry_run']
        assert row['verdict'] and row['s1_rc'] == 78 and s1['state'] == 'no run'
        assert row['start_record']['seal_sha256'] == digest
        assert row['seal_and_arms_verified'] and not row.get('child_started',False)
        assert s1['argv'][-3:] == ['run_gpu_tests.py', line['arm'], line['suite']]
        checked.append(dict(label=row['label'], registered=line, receipt=row))
    save('Q4_seal.json', dict(seal_sha256=digest, arms_verified=list(seal['arms']), arms_pending=seal['arms_pending'],
                              tool_version=dict(cli_supported=False, exit=version.returncode, stdout=version.stdout.decode(),
                                                evidence_sha256=sha((TOOL / 'evidence.py').read_bytes())),
                              dry_labels_checked=checked))
    print('Q4: seal verified;',len(seal['arms']),'arms;',len(checked),'registered dry-run labels')


def l60():
    baseline = json.loads((ROOT / 'artifacts/eng2/L60_dtype_census_baseline.json').read_text())['rows']
    candidate = json.loads((ROOT / 'artifacts/eng2/L60_dtype_census_candidate.json').read_text())['rows']
    changed = {'tc.tensor','tc.tensor(device=cuda,0 bytes)','tc.zeros','tc.ones','tc.randn','tc.rand',
               'tc.nn.parameter','nn.parameter','functional._causal_mask'}
    def clean(row):
        return {k:v for k,v in row.items() if k not in {'site'}}
    b = [clean(x) for x in baseline if x['entry'] not in changed]
    c = [clean(x) for x in candidate if x['entry'] not in changed]
    diffs = [(i,x,y) for i,(x,y) in enumerate(zip(b,c)) if x != y]
    save('Q3_L60.json',dict(baseline_total=len(baseline), candidate_total=len(candidate), unchanged_baseline=len(b),
                           unchanged_candidate=len(c), entries=sorted({x['entry'] for x in b}), differences=diffs,
                           equal=len(b)==len(c) and not diffs, baseline_rows=b, candidate_rows=c))
    print('L60:',len(b),'unchanged rows;',len(diffs),'differences',sorted({x['entry'] for x in b}))


if __name__ == '__main__':
    {'q1':q1,'q2':q2,'q4':q4,'l60':l60}[sys.argv[1]]()
