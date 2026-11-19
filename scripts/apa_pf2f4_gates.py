"""Run PF2-F4 using MAINCHK3's unchanged suite runner and result folding."""
import argparse
from collections import Counter
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from apa_pf2f4_scope import ART, NEW, OLD, P1D, P1F2, RETAINED, ROOT, save

MAIN = ROOT.parent / "tensor_fork_apa_wt_mainchk3"
REF = MAIN / "artifacts/mainchk3"
PF2_TREE = REF / "trees/PF2"
SCRATCH = Path("/tmp/pf2f4_run")
TEST_TREE = SCRATCH / "PF2"
F3_SHA = "1036ec930a1e15d6c21213d977be22fd7c3846814d783f8fea40949ebb8f2347"
ARM = "PF2"
GIB = 1024**3
HEADROOM = 12_000_000_000
DID, OLD_DID, FID = (f"{p}::{name}" for p, name in [(P1D, NEW), (P1D, OLD), (P1F2, RETAINED)])
FAKE_EXCEPTION = "RuntimeError: PF2-F4 power: FakeSpo2.packed_prefill called"
REFUSAL = "ValueError: apa_sp2: prefill (L > 1) on packed keys is not supported; quantize with storage='dense' for prefill"


def load_python(name):
    path = MAIN / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rebind_common():
    common = load_python("mainchk3_common")
    common.ROOT, common.ART, common.SCRATCH = ROOT, ART, SCRATCH
    return common


def read(path):
    return json.loads(path.read_text())


def linked_children(source, dest, exceptions=()):
    dest.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.name not in exceptions:
            (dest / child.name).symlink_to(child, target_is_directory=child.is_dir())


def stage(common):
    # Prior art: MAINCHK3 (Project-Tensor, 2026) pinned archive, R6 fixture and
    # native import. PF2-F4 overlays copies of the two edited tests with links
    # to the same read-only base inputs and uses a separate scratch test tree.
    (ART / "packages").mkdir()
    (ART / "packages/PF2").symlink_to(REF / "packages/PF2", target_is_directory=True)
    (ART / "trees").mkdir()
    (ART / "trees/R6_FIXTURE").symlink_to(REF / "trees/R6_FIXTURE", target_is_directory=True)
    linked_children(PF2_TREE, TEST_TREE, ("tensor_cuda",))
    linked_children(PF2_TREE / "tensor_cuda", TEST_TREE / "tensor_cuda", ("tests",))
    linked_children(PF2_TREE / "tensor_cuda/tests", TEST_TREE / "tensor_cuda/tests", (P1D.name, P1F2.name))
    for relative in [P1D, P1F2]:
        shutil.copyfile(ROOT / relative, TEST_TREE / relative)
    rec = read(REF / "MODULES.json")[ARM]
    assert rec["sha256"] == F3_SHA
    assert common.sha(ROOT / "artifacts/apa_pf2f3/build_F3/lib" / Path(rec["module"]).name) == F3_SHA
    rec["test_root"] = str(TEST_TREE)
    save("MODULES.json", {ARM: rec})
    common.check_arm(ARM)
    save("G1_FILES.json", {"files": [DID, FID]})

    source = TEST_TREE / "tensor_cuda/tensor_cuda"
    old_source = REF / "trees/PROD/tensor_cuda/tensor_cuda/apa_packed.py"
    old_code = old_source.read_text()
    old_start = old_code.index("    if L != 1:", old_code.index("def packed_attention("))
    old_end = old_code.index("    spo2 = importlib.import_module", old_start)
    new_code = (source / "apa_packed.py").read_text()
    start = new_code.index("    spo2 = importlib.import_module", new_code.index("def packed_attention("))
    stop = new_code.index("    result = spo2.packed_run", start)
    # Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978). Restore
    # the baseline branch verbatim and preserve the current decode dispatch.
    mutant_route = SCRATCH / "power_route"
    linked_children(source, mutant_route, ("apa_packed.py",))
    mutated_code = new_code[:start] + old_code[old_start:old_end] + new_code[start:new_code.index("\n", start) + 1] + new_code[stop:]
    (mutant_route / "apa_packed.py").write_text(mutated_code)
    write_mutation("G3_ROUTE.diff", new_code, mutated_code, "apa_packed.py")
    save("G3_ROUTE_FILES.json", {"files": [DID]})

    mutant_fake = TEST_TREE / "power_fake"
    (mutant_fake / P1F2.parent).mkdir(parents=True)
    (mutant_fake / "artifacts").symlink_to(TEST_TREE / "artifacts", target_is_directory=True)
    (mutant_fake / "tensor_cuda/tensor_cuda").symlink_to(source, target_is_directory=True)
    normal = (TEST_TREE / P1F2).read_text()
    result = '            return ("packed_prefill_out", "packed_mask") if args[8] else "packed_prefill_out"'
    assert normal.count(result) == 1
    changed = normal.replace(result, f'            raise RuntimeError("{FAKE_EXCEPTION.split(": ", 1)[1]}")')
    (mutant_fake / P1F2).write_text(changed)
    write_mutation("G3_FAKE.diff", normal, changed, str(P1F2))
    save("G3_FAKE_FILES.json", {"files": [f"power_fake/{FID}"]})

    file_deps = [
        MAIN / "scripts/mainchk3_common.py", MAIN / "scripts/mainchk3_suite.py",
        REF / "MODULES.json", REF / "R6_FIXTURE.json", REF / "NOISE.json",
        REF / "cpu/PF2_r1/RESULT.json", REF / "cpu/PF2_r2/RESULT.json",
        old_source, ROOT / "orders/APA_PF2_F4_REPIN_P1_REFUSAL_TESTS.md",
        ROOT / "docs/APA.md",
    ]
    save("COPY_DEPS.json", {
        "read_only_file_pins": {str(p): common.sha(p) for p in file_deps},
        "unchanged_imports": [str(s) for s in file_deps[:2]],
        "rebindings": {
            "mainchk3_common.ROOT": str(ROOT),
            "mainchk3_common.ART and mainchk3_suite.ART": str(ART),
            "mainchk3_common.SCRATCH and mainchk3_suite.SCRATCH": str(SCRATCH),
            "mainchk3_suite.CPU_AS_BYTES": "per-job limit selected by the registered resource rail",
            "MODULES.json PF2 test_root": str(TEST_TREE),
            "suite junit.xml and RESULT.json": str(ART / "cpu/<label>"),
            "suite pytest basetemp": str(SCRATCH / "<label>/pytest"),
            "environment TMPDIR and caches": str(SCRATCH / "<label>"),
        },
        "read_only_links": {
            str(ART / "packages/PF2"): str(REF / "packages/PF2"),
            str(ART / "trees/R6_FIXTURE"): str(REF / "trees/R6_FIXTURE"),
            str(TEST_TREE): str(PF2_TREE),
        },
        "copied_before_execution": {str(p): common.sha(ROOT / p) for p in [P1D, P1F2]},
        "native_import": {"path": rec["module"], "sha256": rec["sha256"]},
    })
    print(f"Stage pinned Build F3 {F3_SHA}", flush=True)


def write_mutation(name, before, after, path):
    with (ART / name).open("x") as stream:
        stream.writelines(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}"
        ))


def available():
    measurement = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                       if line.startswith("MemAvailable:"))
    return int(measurement.split()[1]) * 1024


def select_limit(small):
    limits = [8 * GIB] if small else [24 * GIB, 23 * GIB]
    deadline = time.monotonic() + (0 if small else 1800)
    while True:
        mem = available()
        for limit in limits:
            if mem >= limit + HEADROOM:
                return limit, mem
        if time.monotonic() >= deadline:
            raise RuntimeError(f"STOP startup memory rail: {mem=}, minimum={limits[-1] + HEADROOM}")
        time.sleep(10)


def run(common, label, selection=None, overrides=None):
    env = common.environment(ARM, "cpu", label)
    env.update(overrides or {})
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_CEILING_DIRECTORIES"] = str(ROOT)
    limit, mem = select_limit(small=selection is not None)
    env["PF2F4_CPU_AS_BYTES"] = str(limit)
    command = ["timeout", "3500s", "nice", "-n", "19", sys.executable, "-B", "-P"]
    # -P omits this script's directory. The wrapper imports its sibling using
    # the absolute scripts entry already in MAINCHK3's PYTHONPATH plus this one.
    env["PYTHONPATH"] += os.pathsep + str(ROOT / "scripts")
    command.extend([str(Path(__file__).resolve()), "worker"])
    if selection:
        command.append(str(ART / selection))
    log = ROOT / "logs" / f"apa_pf2f4_{label.lower()}.log"
    started = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {
        "argv": command, "exit": result.returncode, "elapsed_seconds": time.monotonic() - started,
        "log": str(log), "log_sha256": common.sha(log),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"], "eng3_expect_cuda": env["ENG3_EXPECT_CUDA"],
        "thread_limits": {key: value for key, value in env.items() if key.endswith("_NUM_THREADS")},
        "startup_available_bytes": mem, "rlimit_as_bytes": limit, "startup_headroom_bytes": mem - limit,
        "tmpdir": env["TMPDIR"],
    }
    save(f"{label}_COMMAND.json", record)
    if result.returncode:
        raise RuntimeError(f"{label}: suite harness exit {result.returncode}. See {log}")
    receipt = read(ART / f"cpu/{label}/RESULT.json")
    assert receipt["module_sha256"] == F3_SHA
    assert receipt["complete"] and receipt["rlimit_as_bytes"] == [limit] * 2
    return receipt


def worker(selection):
    rebind_common()
    suite = load_python("mainchk3_suite")
    suite.CPU_AS_BYTES = int(os.environ["PF2F4_CPU_AS_BYTES"])
    sys.argv = [str(MAIN / "scripts/mainchk3_suite.py"), *([selection] if selection else [])]
    return suite.main()


def gates(common, action):
    if action == "g1":
        receipt = run(common, "G1", "G1_FILES.json")
        status = receipt["pytest_exit"] == 0 and receipt["outcomes"] == {DID: "passed", FID: "passed"}
        save("G1.json", {"status": "PASS" if status else "FAIL", "outcomes": receipt["outcomes"]})
        return status
    if action == "g2":
        receipt = run(common, "G2")
        # Prior art: MAINCHK3 (Project-Tensor, 2026) per-ID outcomes and
        # McKeeman (1998) differential testing. PF2-F4 registers three exact
        # key/outcome changes caused by two edited tests and one rename.
        actual = receipt["outcomes"]
        wanted = {OLD_DID: ["failed", None], DID: [None, "passed"], FID: ["failed", "passed"]}
        repeats = {}
        for idx in (1, 2):
            path = REF / f"cpu/PF2_r{idx}/RESULT.json"
            baseline = read(path)
            assert baseline["complete"] and baseline["module_sha256"] == F3_SHA
            old = baseline["outcomes"]
            changes = {item: [old.get(item), actual.get(item)] for item in sorted(old.keys() | actual.keys())
                       if old.get(item) != actual.get(item)}
            repeats[f"PF2_r{idx}"] = {
                "reference": str(path), "reference_sha256": common.sha(path),
                "unchanged": len(old.keys() & actual.keys()) - sum(None not in row for row in changes.values()),
                "changed": changes,
                "unexpected": {item: row for item, row in changes.items() if wanted.get(item) != row},
                "missing_expected_changes": sorted(wanted.keys() - changes.keys()),
                "status": "PASS" if changes == wanted else "FAIL",
            }
        status = all(row["status"] == "PASS" for row in repeats.values())
        save("G2.json", {"status": "PASS" if status else "FAIL", "comparisons": repeats, "ids": len(actual),
                         "counts": dict(Counter(actual.values())), "pytest_exit": receipt["pytest_exit"],
                         "rlimit_as_bytes": receipt["rlimit_as_bytes"]})
        return status
    checks = [
        ("G3_ROUTE", {"APA_P1D_SRC": str(SCRATCH / "power_route")}, DID, REFUSAL),
        ("G3_FAKE", {}, f"power_fake/{FID}", FAKE_EXCEPTION),
    ]
    results = {}
    for label, overrides, node, exception in checks:
        receipt = run(common, label, f"{label}_FILES.json", overrides)
        failures = receipt["failures"].get(node, [])
        status = (receipt["pytest_exit"] == 1 and receipt["outcomes"] == {node: "failed"}
                  and len(failures) == 1 and failures[0]["when"] == "call"
                  and exception in failures[0]["longrepr"])
        results[label] = {"status": "PASS" if status else "FAIL", "node": node, "failures": failures}
    save("G3.json", results)
    return all(result["status"] == "PASS" for result in results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "g1", "g2", "g3", "worker"))
    parser.add_argument("selection", nargs="?")
    args = parser.parse_args()
    if args.action == "worker":
        sys.exit(worker(args.selection))
    common = rebind_common()
    if args.action == "stage":
        stage(common)
    else:
        sys.exit(0 if gates(common, args.action) else 1)
