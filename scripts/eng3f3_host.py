"""Pinned host diagnostic/inherited test runner.
Prior art: ENG3-F1 (2026), pytest hooks (2004+), NIST SHA-256 (2001).
Taken: pinned modules, per-phase outcomes; ours: F2 census and receipts.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/eng3f3"

def main():
    arm, label, selection = sys.argv[1:4]
    recpath = ROOT / ("artifacts/eng3f2/build/BUILD_G.json" if arm == "G" else
                      "artifacts/eng3f1/build/BUILD_f1m02_malloc_unchecked.json" if arm == "f1m02_malloc_unchecked" else
                      "artifacts/eng3f3/build/BUILD_" + arm + ".json")
    rec = json.loads(recpath.read_text())
    module = ROOT / rec["module"]
    assert hashlib.sha256(module.read_bytes()).hexdigest() == rec["module_sha256"]
    package = str(module.parent.parent)
    sys.path.insert(0, package)
    os.environ.update(ENG3F1_PACKAGE_PARENT=package, PYTHONPATH=package,
                      ENG3_RECORD_DIR=str(ART / "host" / label))
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve() == module.resolve()
    import pytest
    # Reuse inherited report folding without its suite entry point.
    import runpy
    os.environ["ENG3F3_MODULE"] = str(module)
    ns = runpy.run_path(str(ROOT / "scripts/eng3f3_suite.py"))
    plugin = ns["Reports"]()
    out = ART / "host" / label
    out.mkdir(parents=True, exist_ok=False)
    args = ["-q", "-s", "-p", "no:cacheprovider", "--rootdir=" + str(ROOT),
            "--basetemp=/tmp/eng3f3_scratch/pt_" + label,
            "--junitxml=" + str(out / "junit.xml"), "-k", selection,
            str(ROOT / "tensor_cuda/tests/test_eng3_host.py"),
            str(ROOT / "tensor_cuda/tests/test_eng3_f3.py")]
    rc = int(pytest.main(args, plugins=[plugin]))
    counts = {}
    for outcome in plugin.ids.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    result = {"arm": arm, "module_sha256": rec["module_sha256"], "exit": rc,
              "args": args, "outcomes": plugin.ids, "counts": counts,
              "collection_errors": plugin.collection_errors, "failures": plugin.longrepr,
              "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES")}
    (out / "RESULT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("outcomes", "failures", "args")}))
    return rc

if __name__ == "__main__":
    sys.exit(main())
