"""F4 GPU staging only at the seat; launch with dry=1 stops before lease.
Prior art: ENG3 F3 r2 pinned/split/prefix harness (2026), pytest (2004+),
NIST SHA-256 (2001), Python subprocess env (PSF, 2003+). Taken: phase parity,
pins and real host prefix. Ours: H-only full F3 lane and F4 registrations.
Unverified -- lead to check: pytest hook API, CUDA cudaGetDevice, SHA-256.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/eng3f4'
LINES = ('gpu-tests', 'g3-g1', 'g3-g2')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outdir(line):
    return ART / 'staged' / line / ('dry' if os.environ['FORGE_GPU_DRY_RUN'] == '1' else 'real')


def pin(arm):
    rec = json.loads((ART / 'MODULES.json').read_text())[arm]
    module = ROOT / rec['module']
    assert sha(module) == rec['sha256'], f'REFUSED BEFORE LEASE: {arm} module changed'
    return rec, module


def verify_inputs():
    rows = json.loads((ART / 'INPUT_PINS.json').read_text())
    for path, expected in rows.items():
        assert sha(ROOT / path) == expected, f'REFUSED BEFORE LEASE: input changed: {path}'
    return rows


def arms(line):
    return ('H',) if line == 'gpu-tests' else ('G', 'H')


def worker(arm, line, prefix):
    rec, module = pin(arm)
    package = str(module.parent.parent)
    os.environ.update(PYTHONPATH=package, ENG3F1_PACKAGE_PARENT=package,
                      ENG3F4_MODULE=str(module), ENG3_EXPECT=rec['sha256'], ENG3_LABEL=line+'_'+arm)
    sys.path.insert(0, package)
    import numpy as np
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve() == module.resolve()
    src = np.array([0.0, 1.0, -2.0], np.float32)
    host = tc.tensor(src, device='cpu')
    for t in (host, tc._C._core_clone(host), tc._C._core_to(host, 'cpu')):
        actual, expected = t.numpy().view(np.uint32), src.view(np.uint32)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        for value in (actual, expected):
            assert value.any() and (value != value[0]).any()
        assert np.array_equal(actual, expected)
    if prefix:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        print(json.dumps({'arm': arm, 'line': line, 'cpu_prefix_pass': True,
                          'module_sha256': rec['sha256'], 'gpu_run': False,
                          'stopped_before': 'first device call; GPU pytest not collected'}), flush=True)
        return 0
    assert os.environ['FORGE_GPU_DRY_RUN'] == '0'
    assert os.environ['ENG3_EXPECT_CUDA'] == '1'
    device = tc.tensor(src, device='cuda')  # FIRST DEVICE CALL, inside lead lease only.
    assert device.shape == src.shape
    if line != 'gpu-tests':
        split = json.loads((ROOT / 'artifacts/eng3/g3/SPLIT.json').read_text())[line.removeprefix('g3-')]
        os.environ['ENG3_FILES'] = ' '.join(split)
        code = runpy.run_path(str(ROOT / 'scripts/eng3f4_suite.py'))['main']()
        if code:
            return code
        os.environ.pop('ENG3_FILES', None)
    lane = 'gpu_all' if line == 'gpu-tests' else 'gpu_compare'
    return subprocess.run(['timeout', '150s', 'nice', '-n', '19', sys.executable, '-B', '-P',
                           str(ROOT / 'scripts/eng3f4_host.py'), arm, line+'_'+arm+'_gpu', lane],
                          cwd=ROOT).returncode


def compare(line):
    host = {arm: json.loads((ART / 'host' / (line+'_'+arm+'_gpu') / 'RESULT.json').read_text())
            for arm in arms(line)}
    amended = [i for i in json.loads((ART / 'TOUCHED_IDS.json').read_text())
               if '::test_f2_cpu_overflow_message[' in i]
    branches = []
    for i in amended:
        row = {'id': i}
        for arm, data in host.items():
            records = data['records'].get(i, [])
            row[arm] = {'outcome': data['outcomes'].get(i), 'records': records,
                        'query_success_overflow': len(records) == 1 and
                        'byte size overflows' in records[0]['message'] and
                        str(records[0]['shape']) in records[0]['message'] and
                        'dtype '+records[0]['dtype'] in records[0]['message']}
        branches.append(row)
    expected_n = 38 if line == 'gpu-tests' else 49
    host_ok = all(d['exit'] == 0 and not d['collection_errors'] and
                  len(d['outcomes']) == expected_n and set(d['outcomes'].values()) == {'passed'}
                  and d['ENG3_EXPECT_CUDA'] == '1' for d in host.values())
    host_ok = host_ok and len(branches) == 24 and all(r[a]['query_success_overflow'] for r in branches for a in host)
    # The H-only lane must execute all 14 F3 IDs, including all six children.
    if line == 'gpu-tests':
        f3 = [i for i in host['H']['outcomes'] if i.startswith('tensor_cuda/tests/test_eng3_f3.py::')]
        six = [i for i in f3 if '::test_f3_cpu_no_device_query[' in i]
        host_ok = host_ok and len(f3) == 14 and len(six) == 6
    else:
        host_ok = host_ok and host['G']['outcomes'] == host['H']['outcomes']
    result = {'pass': host_ok, 'host_pass': host_ok, 'n_host_ids_per_arm': expected_n,
              'amended_24': branches, 'host_receipts': {a: {'exit': d['exit'], 'counts': d['counts']}
                                                      for a, d in host.items()}}
    if line != 'gpu-tests':
        data = {a: json.loads((ART / 'g3' / (line+'_'+a) / 'outcomes.json').read_text()) for a in arms(line)}
        ids = sorted(set(data['G']['outcomes']) | set(data['H']['outcomes']))
        differences = [{'id': i, 'G': data['G']['outcomes'].get(i), 'H': data['H']['outcomes'].get(i)}
                       for i in ids if data['G']['outcomes'].get(i) != data['H']['outcomes'].get(i)]
        collect_diff = sorted(set(data['G']['collection_errors']) ^ set(data['H']['collection_errors']))
        observed = {a: d['worktree_writes_by_tests'] for a, d in data.items()}
        writes = {a: [p for p in rows if not (p.startswith('logs/eng3f4_') or p.startswith('artifacts/eng3f4/'))]
                  for a, rows in observed.items()}
        suite_ok = bool(ids) and not differences and not collect_diff and not any(writes.values())
        result.update(suite_pass=suite_ok, n_suite_ids=len(ids), differences=differences,
                      collection_file_differences=collect_diff, observed_writes=observed,
                      writes_by_tests=writes, suite_exits={a: d['pytest_exit'] for a,d in data.items()},
                      suite_counts={a: d['counts'] for a,d in data.items()})
        result['pass'] = host_ok and suite_ok
    dump(outdir(line) / 'COMPARE.json', result)
    return result['pass']


def execute(line, prefix):
    verify_inputs()
    for arm in arms(line):
        pin(arm)
    out = outdir(line)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in arms(line):
        env = dict(os.environ)
        if prefix:
            env['CUDA_VISIBLE_DEVICES'] = ''
        cmd = ['timeout', '490s', 'nice', '-n', '19', sys.executable, '-B', '-P',
               __file__, 'worker', arm, line, '1' if prefix else '0']
        log = out / (('prefix_' if prefix else 'suite_')+arm+'.log')
        with log.open('x') as f:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        results.append({'arm': arm, 'exit': p.returncode, 'argv': cmd,
                        'log': str(log.relative_to(ROOT)), 'log_sha256': sha(log)})
        if p.returncode:
            break
    ok = len(results) == len(arms(line)) and all(r['exit'] == 0 for r in results)
    if ok and not prefix:
        ok = compare(line)
    dump(out / ('CPU_PREFIX.json' if prefix else 'INNER_RESULT.json'),
         {'pass': ok, 'prefix_only': prefix, 'inner_exit': 0 if ok else 1, 'results': results})
    return 0 if ok else 1


def launch(line):
    assert os.environ['FORGE_GPU_DRY_RUN'] in ('0', '1')
    assert os.environ['ENG3_EXPECT_CUDA'] == '1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    inputs = verify_inputs()
    modules = {a: pin(a)[0] for a in arms(line)}
    out = outdir(line)
    out.mkdir(parents=True, exist_ok=False)
    cmd = ['timeout', '550s', 'forge-gpu-run', '--min', '30', '--max', '540',
           '--label', 'eng3f4-'+line, '--', 'timeout', '510s', 'nice', '-n', '19',
           sys.executable, '-B', '-P', __file__, 'run', line]
    prelease = {'line': line, 'full_environment': dict(os.environ), 'module_pins': modules,
                'input_pins': inputs, 'forge_argv': cmd, 'gpu_run': False}
    dump(out / 'PRELEASE.json', prelease)
    if execute(line, True):
        return 1
    verify_inputs()
    for arm in arms(line):
        pin(arm)
    log = out / 'GPU.log'
    with log.open('x') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    inner = out / 'INNER_RESULT.json'
    result = json.loads(inner.read_text()) if inner.exists() else None
    dry = os.environ['FORGE_GPU_DRY_RUN'] == '1'
    ok = p.returncode == 78 and result is None if dry else p.returncode == 0 and result is not None and result['inner_exit'] == 0
    rc = (78 if dry else 0) if ok else (p.returncode or 1)
    dump(out / 'RESULT.json', {'line': line, 'forge_exit': p.returncode,
         'inner_exit': result['inner_exit'] if result else None, 'launcher_exit': rc,
         'dry_run': dry, 'gpu_run': not dry, 'gpu_log_sha256': sha(log),
         'expected_forge_exit': 78 if dry else 0, 'expected_inner_exit': None if dry else 0,
         'pass': ok})
    print(json.dumps({'line': line, 'forge_exit': p.returncode, 'inner_exit': result['inner_exit'] if result else None,
                      'launcher_exit': rc, 'gpu_log_sha256': sha(log)}), flush=True)
    return rc


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'worker':
        sys.exit(worker(sys.argv[2], sys.argv[3], sys.argv[4] == '1'))
    line = sys.argv[2]
    assert line in LINES
    if mode == 'launch':
        sys.exit(launch(line))
    assert mode == 'run' and os.environ['FORGE_GPU_DRY_RUN'] == '0'
    sys.exit(execute(line, False))
