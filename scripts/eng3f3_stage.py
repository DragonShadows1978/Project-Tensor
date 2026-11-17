"""Inherited F1 g3 CPU/GPU suite forms with G/H pins and pre-lease checks.
Prior art: ENG3-F1 per-ID/split/prefix harness (2026), pytest (2004+),
NIST SHA-256 (2001). Taken: split census, phase outcomes, pinning;
ours: explicit F3 environment and receipts. No new numeric algorithm.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/eng3f3"

def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def pin(arm):
    path = ROOT / ("artifacts/eng3f2/build/BUILD_G.json" if arm == "G" else
                   "artifacts/eng3f3/build/BUILD_H.json")
    rec = json.loads(path.read_text())
    module = ROOT / rec["module"]
    got = hashlib.sha256(module.read_bytes()).hexdigest()
    if got != rec["module_sha256"]:
        raise RuntimeError(f"REFUSED BEFORE LEASE: {arm} module {got} != {rec['module_sha256']}")
    return rec, module

def worker(arm, line, prefix):
    rec, module = pin(arm)
    label = line + "_" + arm
    package = str(module.parent.parent)
    os.environ.update(PYTHONPATH=package, ENG3F1_PACKAGE_PARENT=package,
        ENG3F3_MODULE=str(module), ENG3_EXPECT=rec["module_sha256"], ENG3_LABEL=label)
    sys.path.insert(0, package)
    import numpy as np
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve() == module.resolve()
    # Actual shared CPU prefix: the next tensor call allocates on device.
    src = np.array([0.0, 1.0, -2.0], np.float32)
    host = tc.tensor(src, device="cpu")
    for tensor in (host, tc._C._core_clone(host), tc._C._core_to(host, "cpu")):
        got, ref = tensor.numpy().view(np.uint32), src.view(np.uint32)
        assert got.dtype == ref.dtype and got.shape == ref.shape
        for value in (got, ref):
            assert value.any() and (value != value[0]).any()
        assert np.array_equal(got, ref)
    if prefix:
        print(json.dumps({"arm": arm, "line": line, "module_sha256": rec["module_sha256"],
            "cpu_prefix_pass": True, "gpu_run": False,
            "stopped_before": "production CPU suite" if line == "g3-cpu" else
                              "tc.tensor(src,device=cuda)"}), flush=True)
        return 0
    if line != "g3-cpu":
        device = tc.tensor(src, device="cuda")  # FIRST DEVICE CALL
        assert device.shape == src.shape
        split = line.removeprefix("g3-")
        files = json.loads((ROOT / "artifacts/eng3/g3/SPLIT.json").read_text())[split]
        os.environ["ENG3_FILES"] = " ".join(files)
    else:
        os.environ.pop("ENG3_FILES", None)
    ns = runpy.run_path(str(ROOT / "scripts/eng3f3_suite.py"))
    return ns["main"]()

def compare(line):
    data = [json.loads((ART / "g3" / (line + "_" + arm) / "outcomes.json").read_text())
            for arm in ("G", "H")]
    a, b = data
    ids = sorted(set(a["outcomes"]) | set(b["outcomes"]))
    differences = [{"id": i, "G": a["outcomes"].get(i), "H": b["outcomes"].get(i)}
        for i in ids if a["outcomes"].get(i) != b["outcomes"].get(i)]
    result = {"n_ids": len(ids), "differences": differences, "intended_changes": [],
        "collection_file_differences": sorted(set(a["collection_errors"]) ^ set(b["collection_errors"])),
        "suite_exits": [d["pytest_exit"] for d in data],
        "counts": {arm: d["counts"] for arm, d in zip(("G", "H"), data)},
        "writes": {arm: d["worktree_writes_by_tests"] for arm, d in zip(("G", "H"), data)}}
    ok = bool(ids) and not differences and not result["collection_file_differences"] and not any(result["writes"].values())
    result["pass"] = ok
    dump(ART / "staged" / line / "COMPARE.json", result)
    print(json.dumps(result), flush=True)
    return ok

def execute(line, prefix):
    assert line in ("g3-cpu", "g3-g1", "g3-g2")
    # Validate BOTH before running either worker, also used before the lease.
    for arm in ("G", "H"):
        pin(arm)
    results = []
    for arm in ("G", "H"):
        env = dict(os.environ)
        if prefix or line == "g3-cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = ["timeout", "500s", "nice", "-n", "19", sys.executable, "-B", "-P",
               __file__, "worker", arm, line, "1" if prefix else "0"]
        out = ART / "staged" / line
        out.mkdir(parents=True, exist_ok=True)
        logpath = out / (("prefix_" if prefix else "suite_") + arm + ".log")
        with logpath.open("w") as f:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        results.append({"arm": arm, "argv": cmd, "exit": p.returncode,
                        "log": str(logpath.relative_to(ROOT)),
                        "log_sha256": hashlib.sha256(logpath.read_bytes()).hexdigest(),
                        "module_sha256": pin(arm)[0]["module_sha256"]})
        print(json.dumps(results[-1]), flush=True)
        if p.returncode:
            break
    ok = len(results) == 2 and all(row["exit"] == 0 for row in results)
    if ok and not prefix:
        ok = compare(line)
    dump(ART / "staged" / line / ("CPU_PREFIX.json" if prefix else "INNER_RESULT.json"),
         {"line": line, "prefix_only": prefix, "pass": ok, "inner_exit": 0 if ok else 1,
          "results": results, "time": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return 0 if ok else 1

def finish(line, rc):
    out = ART / "staged" / line
    inner = out / "INNER_RESULT.json"
    result = json.loads(inner.read_text()) if inner.exists() else None
    log = out / "GPU.log"
    dump(out / ("DRY_RESULT.json" if os.environ.get("FORGE_GPU_DRY_RUN") == "1" else "RESULT.json"),
         {"line": line, "forge_exit": int(rc),
          "inner_exit": result.get("inner_exit") if result else None,
          "gpu_run": os.environ.get("FORGE_GPU_DRY_RUN") != "1",
          "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
    return int(rc)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "worker":
        sys.exit(worker(sys.argv[2], sys.argv[3], sys.argv[4] == "1"))
    if mode == "finish":
        sys.exit(finish(sys.argv[2], sys.argv[3]))
    assert mode in ("prefix", "run")
    sys.exit(execute(sys.argv[2], mode == "prefix"))
