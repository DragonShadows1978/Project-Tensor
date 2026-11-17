"""Pinned CPU/GPU host-row runner, ENG3-F3 Amendment 1 (2026).
Prior art: ENG3-F2/F3 runners (2026), pytest hooks (2004+), SHA-256
(NIST 2001). Taken: pinned imports and phase folding; ours: r2 exception
receipts per amended row. No new product algorithm.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/eng3f3"


def main():
    arm, label, lane = sys.argv[1:4]
    assert label.endswith("_r2") and lane in ("cpu", "gpu")
    receipt = ROOT / ("artifacts/eng3f2/build/BUILD_G.json" if arm == "G"
                      else "artifacts/eng3f3/build/BUILD_H.json")
    rec = json.loads(receipt.read_text())
    module = ROOT / rec["module"]
    assert hashlib.sha256(module.read_bytes()).hexdigest() == rec["module_sha256"]
    package = str(module.parent.parent)
    sys.path.insert(0, package)
    os.environ.update(PYTHONPATH=package, ENG3F1_PACKAGE_PARENT=package,
                      ENG3F3_MODULE=str(module))
    os.environ.pop("ENG3_RECORD_DIR", None)
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve() == module.resolve()
    import pytest
    base = runpy.run_path(str(ROOT / "scripts/eng3f3_suite_r2.py"))["Reports"]

    class Reports(base):
        def __init__(self):
            super().__init__()
            self.records = {}
            self.exceptions = {}
            self.current_id = None

        def pytest_collection_modifyitems(self, items):
            for mod in {item.module for item in items}:
                if not hasattr(mod, "record"):
                    continue

                def capture(name, row):
                    self.records.setdefault(self.current_id, []).append(
                        {"record_name": name, **row})

                mod.record = capture  # Receipts in one _r2 file; no test edits.

        def pytest_runtest_setup(self, item):
            self.current_id = item.nodeid

        def pytest_runtest_makereport(self, item, call):
            if call.when == "call" and call.excinfo:
                self.exceptions[item.nodeid] = {
                    "exception_type": call.excinfo.type.__name__,
                    "message": str(call.excinfo.value)}

    out = ART / "host" / label
    out.mkdir(parents=True, exist_ok=False)
    selection = ("cpu or test_f1_gpu_device_index" if lane == "cpu" else
                 "test_f2_cpu_overflow_message or test_f1_gpu_device_index")
    args = ["-q", "-s", "-p", "no:cacheprovider", "--rootdir=" + str(ROOT),
            "--basetemp=/tmp/eng3f3_host_r2/pt_" + label,
            "--junitxml=" + str(out / "junit_r2.xml"), "-k", selection,
            str(ROOT / "tensor_cuda/tests/test_eng3_host.py"),
            str(ROOT / "tensor_cuda/tests/test_eng3_f3.py")]
    if lane == "cpu":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    else:
        import ctypes
        current = ctypes.c_int(-1)
        assert ctypes.CDLL("libcudart.so.12").cudaGetDevice(ctypes.byref(current)) == 0
    plugin = Reports()
    rc = int(pytest.main(args, plugins=[plugin]))
    counts = {}
    for outcome in plugin.ids.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    result = {"arm": arm, "lane": lane, "module_sha256": rec["module_sha256"],
              "exit": rc, "args": args, "outcomes": plugin.ids, "counts": counts,
              "collection_errors": plugin.collection_errors,
              "failures": plugin.longrepr, "unhandled_exceptions": plugin.exceptions,
              "records": plugin.records,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (out / "RESULT_r2.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                     ("arm", "lane", "exit", "counts", "module_sha256")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
