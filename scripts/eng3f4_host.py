"""Pinned F4 row runner. Prior art: ENG3 F3 r2 (2026), pytest hooks
(Krekel et al. 2004+), SHA-256 (NIST 2001). Reuse phase folding and row
capture; ours: F4 selections and CPU-only GPU-row skips outside shipped code.
External citations unverified -- lead to check: pytest hook API, SHA-256.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/eng3f4'
HOST = 'tensor_cuda/tests/test_eng3_host.py'
F3 = 'tensor_cuda/tests/test_eng3_f3.py'


def main():
    arm, label, lane = sys.argv[1:]
    assert lane in ('all', 'cpu', 'unset', 'power', 'gpu_all', 'gpu_compare')
    pins = json.loads((ART / 'MODULES.json').read_text())
    rec = pins[arm]
    module = ROOT / rec['module']
    assert hashlib.sha256(module.read_bytes()).hexdigest() == rec['sha256']
    package = str(module.parent.parent)
    os.environ.update(PYTHONPATH=package, ENG3F1_PACKAGE_PARENT=package,
                      ENG3F4_MODULE=str(module))
    sys.path.insert(0, package)
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve() == module.resolve()
    import pytest
    base = runpy.run_path(str(ROOT / 'scripts/eng3f4_suite.py'))['Reports']

    class Reports(base):
        def __init__(self):
            super().__init__()
            self.records = {}
            self.exceptions = {}
            self.current_id = None
            self.cpu_lane_gpu_skips = []

        def pytest_collection_modifyitems(self, items):
            for item in items:
                if lane == 'all' and (item.originalname.startswith('test_g2_') or
                    (item.originalname == 'test_g1b_bool_numpy' and item.callspec.params['device'] == 'cuda')):
                    item.add_marker(pytest.mark.skip(reason='Registered G1 CPU lane: host GPU-only row'))
                    self.cpu_lane_gpu_skips.append(item.nodeid)
            for mod in {item.module for item in items}:
                if hasattr(mod, 'record'):
                    def capture(name, row):
                        self.records.setdefault(self.current_id, []).append({'record_name': name, **row})
                    mod.record = capture

        def pytest_runtest_setup(self, item):
            self.current_id = item.nodeid

        def pytest_runtest_makereport(self, item, call):
            if call.when == 'call' and call.excinfo:
                self.exceptions[item.nodeid] = {'exception_type': call.excinfo.type.__name__,
                                                'message': str(call.excinfo.value)}

    out = ART / 'host' / label
    out.mkdir(parents=True, exist_ok=False)
    args = ['-q', '-s', '-p', 'no:cacheprovider', '--rootdir=' + str(ROOT),
            '--basetemp=/tmp/eng3f4_host/' + label,
            '--junitxml=' + str(out / 'junit.xml')]
    touched = json.loads((ART / 'TOUCHED_IDS.json').read_text())
    cuda24 = [i for i in touched if '::test_f2_cpu_overflow_message[' in i]
    if lane in ('all', 'cpu'):
        args += [HOST, F3]
        if lane == 'cpu':
            args += ['-k', 'cpu or test_f1_gpu_device_index']
    elif lane == 'unset':
        args += [F3, '-k', 'test_f3_cpu_no_device_query']
        assert 'CUDA_VISIBLE_DEVICES' not in os.environ
    elif lane == 'power':
        args += cuda24
        assert os.environ['ENG3_EXPECT_CUDA'] == '1'
    elif lane == 'gpu_all':
        args += [F3] + cuda24
        assert os.environ['ENG3_EXPECT_CUDA'] == '1'
    else:
        args += [HOST, F3, '-k', 'test_f2_cpu_overflow_message or test_f1_gpu_device_index']
        assert os.environ['ENG3_EXPECT_CUDA'] == '1'
    if lane in ('all', 'cpu', 'power'):
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plugin = Reports()
    rc = int(pytest.main(args, plugins=[plugin]))
    counts = {v: list(plugin.ids.values()).count(v) for v in set(plugin.ids.values())}
    result = {'arm': arm, 'lane': lane, 'module_sha256': rec['sha256'],
              'exit': rc, 'args': args, 'outcomes': plugin.ids, 'counts': counts,
              'collection_errors': plugin.collection_errors, 'failures': plugin.longrepr,
              'unhandled_exceptions': plugin.exceptions, 'records': plugin.records,
              'cpu_lane_gpu_skips': plugin.cpu_lane_gpu_skips,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'ENG3_EXPECT_CUDA': os.environ.get('ENG3_EXPECT_CUDA')}
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('arm', 'lane', 'exit', 'counts', 'module_sha256')}))
    return rc


if __name__ == '__main__':
    sys.exit(main())
