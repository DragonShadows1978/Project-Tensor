#!/usr/bin/env python3
"""ENG1 staging generator (CPU only). Steps (each a separate command, each under 20 minutes):
  arms      build arms/<arm>/tensor_cuda/{tensor_cuda/<package + pinned module>, tests/<test_eng1.py,
            test_terrain_render.py>} for baseline, candidate, every built mutant and C_UBSAN (copies, never links)
  census    collect each suite's test IDs on CPU with the arm package preloaded (ids_<census>.txt)
  lines     write gpu_lines.json, regression_files.json, baseline_expected_failures.json
  seal      write staging_seal.json (order, tools, registration files, arm maps, engine pins)
  script    write STAGED_GPU_COMMANDS.sh
Prior art: GN1 A8 staging (regenerate_arms.py, seal_staging.py) re-implemented for per-arm engines; SHA-256 pins.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
ENG1 = TOOL.parent
ROOT = TOOL.parents[2]
BASE_COMMIT = "1680764dbeea6d4215eba06b1a450cf345e7db11"
ORDER_COMMIT = "007137ded4b115919afc3f001626762c8f83cadb"
MODULE = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
MUTANTS = json.loads((ENG1 / "mutants" / "mutants.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_show(path):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:{path}"], capture_output=True,
                          check=True).stdout


def package_files(commit_tree=False):
    out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", BASE_COMMIT, "tensor_cuda/tensor_cuda"],
                         capture_output=True, text=True, check=True).stdout.split()
    return [p for p in out if p.endswith(".py")]


def make_arm(arm, module_src, base=False, terrain_text=None):
    dest = TOOL / "arms" / arm / "tensor_cuda"
    if dest.exists():
        print("exists", arm)
        return
    pkg, tests = dest / "tensor_cuda", dest / "tests"
    for rel in package_files():
        target = dest / Path(rel).relative_to("tensor_cuda")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git_show(rel) if base else (ROOT / rel).read_bytes())
    shutil.copy2(module_src, pkg / MODULE)
    tests.mkdir(parents=True)
    shutil.copy2(ROOT / "tensor_cuda/tests/test_eng1.py", tests / "test_eng1.py")
    if terrain_text is not None:
        (tests / "test_terrain_render.py").write_text(terrain_text)
    elif base:
        (tests / "test_terrain_render.py").write_bytes(git_show("tensor_cuda/tests/test_terrain_render.py"))
    else:
        shutil.copy2(ROOT / "tensor_cuda/tests/test_terrain_render.py", tests / "test_terrain_render.py")
    print("built", arm, sha(pkg / MODULE))


def step_arms():
    # The package Python sources are unchanged between the arms (G1); the baseline's are taken from the base commit.
    make_arm("baseline", ROOT / "tensor_cuda/build-eng1-base/module" / MODULE, base=True)
    make_arm("candidate", ROOT / "tensor_cuda/build-eng1/module" / MODULE)
    cand_terrain = (ROOT / "tensor_cuda/tests/test_terrain_render.py").read_text()
    for m in MUTANTS:
        if m["kind"] == "python":
            assert cand_terrain.count(m["old"]) == 1
            make_arm(m["id"], ROOT / "tensor_cuda/build-eng1/module" / MODULE,
                     terrain_text=cand_terrain.replace(m["old"], m["new"], 1))
        elif (ENG1 / "mutants" / m["id"] / MODULE).exists():
            make_arm(m["id"], ENG1 / "mutants" / m["id"] / MODULE)
        else:
            print("pending (module not built)", m["id"])
    if (ENG1 / "mutants" / "C_UBSAN" / MODULE).exists():
        make_arm("C_UBSAN", ENG1 / "mutants" / "C_UBSAN" / MODULE)
    if (ENG1 / "mutants" / "_unmutated_copy" / MODULE).exists():
        # C_COPY: the unmutated candidate sources built by the mutant route (scratch tree, incremental); its bytes
        # differ from the worktree candidate's (embedded source path), so it is a control that must pass.
        make_arm("C_COPY", ENG1 / "mutants" / "_unmutated_copy" / MODULE)


COLLECT = r'''
import json, os, sys
from pathlib import Path
package, out, *files = sys.argv[1:]
sys.path.insert(0, package)
import tensor_cuda, pytest
assert Path(tensor_cuda.__file__).resolve().parent.parent == Path(package).resolve()
class P:
    ids, errors = [], []
    def pytest_collection_finish(self, session): P.ids = [i.nodeid for i in session.items]
    def pytest_collectreport(self, report):
        if report.failed: P.errors.append(str(report.longrepr)[-3000:])
extra = os.environ.get("ENG1_COLLECT_K")
args = ["-q", "-p", "no:cacheprovider", os.environ["ENG1_IMPORT_MODE"], "--collect-only", "--rootdir=" + os.environ["ENG1_ROOT"]]
if extra: args += ["-k", extra]
rc = pytest.main(args + files, plugins=[P()])
Path(out).write_text(json.dumps(dict(rc=int(rc), ids=P.ids, errors=P.errors), indent=1) + "\n")
'''


def norm(i):
    import re
    m = re.match(r"artifacts/eng1/tool/arms/[^/]+/(tensor_cuda/.*)$", i)
    return m.group(1) if m else i


def collect(arm, suite, k=None):
    pkg = TOOL / "arms" / arm / "tensor_cuda"
    if suite == "regression":
        files = [str(ROOT / f) for f in json.loads((TOOL / "regression_files.json").read_text())]
    elif suite == "terrain":
        files = [str(pkg / "tests/test_terrain_render.py")]
    else:
        files = [str(pkg / "tests/test_eng1.py")]
    tag = "" if not k else "_cpu" if k == "cpu" else "_noe6"
    out = Path("/tmp/eng1_tmp") / f"collect_{arm}_{suite}{tag}.json"
    env = dict(os.environ, ENG1_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               ENG1_IMPORT_MODE="--import-mode=prepend" if suite in ("regression", "terrain") else "--import-mode=importlib")
    if k:
        env["ENG1_COLLECT_K"] = k
    p = subprocess.run([sys.executable, "-B", "-c", COLLECT, str(pkg), str(out), *files], env=env, cwd="/tmp/eng1_tmp",
                       capture_output=True, text=True, timeout=1100)
    data = json.loads(out.read_text())
    (TOOL / f"collect_{arm}_{suite}{tag}.json").write_text(json.dumps(
        dict(rc=data["rc"], n=len(data["ids"]), errors=data["errors"], stderr_tail=p.stderr[-2000:]), indent=1) + "\n")
    return data


def step_census():
    results = {}
    for arm, suite, k, name in (("baseline", "eng1", None, "eng1"), ("candidate", "eng1", None, "eng1_candidate"),
                                ("candidate", "eng1", "cpu", "audit"),
                                ("candidate", "eng1", "not test_e6_terrain_writes_nothing", "eng1_noe6"),
                                ("baseline", "regression", None, "regression"),
                                ("candidate", "regression", None, "regression_candidate"),
                                ("baseline", "terrain", None, "terrain"), ("candidate", "terrain", None, "terrain_candidate")):
        data = collect(arm, suite, k)
        ids = sorted(norm(i) for i in data["ids"])
        results[name] = (data["rc"], len(ids), len(data["errors"]))
        (TOOL / f"ids_{name}.txt").write_text("".join(i + "\n" for i in ids))
    assert (TOOL / "ids_eng1.txt").read_text() == (TOOL / "ids_eng1_candidate.txt").read_text(), "eng1 IDs differ by arm"
    assert (TOOL / "ids_regression.txt").read_text() == (TOOL / "ids_regression_candidate.txt").read_text(), "regression IDs differ by arm"
    assert (TOOL / "ids_terrain.txt").read_text() == (TOOL / "ids_terrain_candidate.txt").read_text(), "terrain IDs differ by arm"
    print(json.dumps(results))


def bounds(inner):
    return dict(inner=inner, minimum=inner + 10, maximum=inner + 60, outer=inner + 90)


def step_lines():
    rows = [dict(label="baseline_eng1", arm="baseline", suite="eng1", role="baseline", census="eng1",
                 expected_child_exits=[1], expected_parent_exit=0, **bounds(ENG1_INNER)),
            dict(label="candidate_eng1", arm="candidate", suite="eng1", role="candidate", census="eng1",
                 expected_child_exits=[0], expected_parent_exit=0, **bounds(ENG1_INNER)),
            dict(label="baseline_regression", arm="baseline", suite="regression", role="regression", census="regression",
                 expected_child_exits=[0, 1], expected_parent_exit=0, **bounds(REG_INNER)),
            dict(label="candidate_regression", arm="candidate", suite="regression", role="regression",
                 census="regression", expected_child_exits=[0, 1], expected_parent_exit=0, **bounds(REG_INNER)),
            dict(label="baseline_terrain", arm="baseline", suite="terrain", role="regression", census="terrain",
                 expected_child_exits=[0, 1], expected_parent_exit=0, **bounds(TER_INNER)),
            dict(label="candidate_terrain", arm="candidate", suite="terrain", role="regression", census="terrain",
                 expected_child_exits=[0, 1], expected_parent_exit=0, **bounds(TER_INNER)),
            dict(label="C_COPY", arm="C_COPY", suite="control", role="control", census="eng1_noe6",
                 control_of="M01-M08,M10-M13", expected_child_exits=[0], expected_parent_exit=0, **bounds(MUT_INNER)),
            dict(label="C_UBSAN", arm="C_UBSAN", suite="control", role="control", census="eng1_noe6", control_of="M09",
                 expected_child_exits=[0], expected_parent_exit=0, **bounds(ENG1_INNER))]
    for m in MUTANTS:
        py = m["kind"] == "python"
        rows.append(dict(label=m["id"], arm=m["id"], suite="mutation_e6" if py else "mutation", role="mutant",
                         census="eng1" if py else "eng1_noe6", expected_child_exits=[1], expected_parent_exit=0,
                         **bounds(ENG1_INNER if py else MUT_INNER)))
    (TOOL / "gpu_lines.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Registered before any GPU run: the baseline's expected failures, per ID (report: derivation per class).
    ids = (TOOL / "ids_eng1.txt").read_text().split()
    pred = json.loads((ENG1 / "cpu_checks.json").read_text())["e2_emulated_predictions"]
    fail = []
    for i in ids:
        name = i.split("::", 1)[1]
        if name.startswith(("test_e1_infer[", "test_e1_infer_empty_cpu[")) or name == "test_e1_view_shares_storage":
            fail.append(i)
        elif name.startswith(("test_e1_refuse[", "test_e1_refuse_empty_cpu[")) and "count_mismatch" not in name:
            fail.append(i)
        elif name.startswith("test_e2_pow_grad[") and pred[i]["baseline"]["predicted"] == "fail":
            fail.append(i)
        elif name.startswith("test_e4_refuse[topk-"):
            fail.append(i)
        elif name.startswith("test_e5_refuse_cpu[") or name == "test_e6_terrain_writes_nothing":
            fail.append(i)
    (TOOL / "baseline_expected_failures.json").write_text(json.dumps(sorted(fail), indent=1) + "\n")
    print(len(rows), "lines;", len(fail), "registered baseline failures of", len(ids))


def step_seal():
    arms = sorted(p.name for p in (TOOL / "arms").iterdir())
    from evidence import arm_files
    seal = dict(order_commit=ORDER_COMMIT, order_sha256=sha(ROOT / "orders/ENG1_ENGINE_DEFECTS.md"),
                base_commit=BASE_COMMIT, module_name=MODULE,
                tool_files={p.name: sha(p) for p in sorted(TOOL.iterdir())
                            if p.is_file() and p.name not in ("staging_seal.json",) and not p.name.startswith("collect_")},
                registration_files={str(p.relative_to(ROOT)): sha(p) for p in
                                    [ROOT / "tensor_cuda/tests/test_eng1.py", ENG1 / "mutants/mutants.json"] +
                                    [ROOT / f for f in json.loads((TOOL / "regression_files.json").read_text())]},
                arms={a: arm_files(a) for a in arms},
                engine_sha256={a: sha(TOOL / "arms" / a / "tensor_cuda/tensor_cuda" / MODULE) for a in arms})
    for a in arms:  # every arm runs the worktree's current test_eng1.py
        assert sha(TOOL / "arms" / a / "tensor_cuda/tests/test_eng1.py") == sha(ROOT / "tensor_cuda/tests/test_eng1.py"), a
    labels = {r["arm"] for r in json.loads((TOOL / "gpu_lines.json").read_text())}
    missing = sorted(labels - set(arms))
    seal["arms_pending"] = missing
    (TOOL / "staging_seal.json").write_text(json.dumps(seal, indent=1, sort_keys=True) + "\n")
    print("sealed arms:", arms, "pending:", missing, "seal", sha(TOOL / "staging_seal.json"))


def step_script():
    rows = json.loads((TOOL / "gpu_lines.json").read_text())
    out = ['#!/usr/bin/env bash',
           '# ENG1 staged GPU lines (adapted from GN1 A8; the lead runs them; the seat never did on a device).',
           '# Prior art: GNU timeout, POSIX shell; GN1 L45 completed-child supervision (parent receipt after child exit).',
           '# Expected parent exit for EVERY line is 0 (the parent passes when the child exit matches the registration',
           '# and the receipts check out). Child exits and bounds are listed per label below and in gpu_lines.json.',
           '# Usage: RUNS=/abs/path/under/artifacts/eng1/tool/gpu_runs/<root> TMPDIR=/tmp/eng1_g5 \\',
           '#        CUDA_VISIBLE_DEVICES=0 timeout 1190s bash STAGED_GPU_COMMANDS.sh LABEL',
           '# Run the labels in gpu_lines.json order with the SAME RUNS (mutants need a passing candidate_eng1;',
           '# candidate_regression needs baseline_regression, candidate_terrain needs baseline_terrain).',
           '# Afterwards: RUNS=<same> CUDA_VISIBLE_DEVICES= \\',
           '#        timeout 300s python3 score_gpu.py   (expected exit 0).',
           '# CPU plumbing audit: RUNS=/abs/audit/root TMPDIR=/tmp/eng1_audit timeout 1190s bash STAGED_GPU_COMMANDS.sh --audit',
           'set -euo pipefail',
           ': "${RUNS:?RUNS is required; use a fresh run root and reuse it across the labels}"',
           '[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }',
           'TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
           'cd -- "$TOOL"',
           'export RUNS PYTHONDONTWRITEBYTECODE=1',
           'export TMPDIR="${TMPDIR:?TMPDIR must be /tmp/eng1_*}"',
           'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
           'GPU_RUN=forge-gpu-run',
           'label="${1:?select one registered label, or --audit}"',
           'if [[ "$label" == --audit ]]; then',
           '    export ENG1_AUDIT=1 CUDA_VISIBLE_DEVICES=',
           '    GPU_RUN="$TOOL/forge_gpu_run_standin.py"',
           '    label=candidate_eng1',
           'else',
           '    unset ENG1_AUDIT',
           'fi',
           'mkdir -p -- "$TMPDIR"',
           'timeout 60s python3 start_gpu_run.py',
           'case "$label" in']
    for r in rows:
        exp = ",".join(str(v) for v in r["expected_child_exits"])
        out.append(f'  {r["label"]}) # expected child {exp}, parent 0; inner {r["inner"]}s, --max {r["maximum"]}s, outer {r["outer"]}s')
        out.append(f'    timeout {r["outer"] + 20}s python3 run_gpu_line.py {r["label"]} {exp} {r["outer"]} "$GPU_RUN" '
                   f'--min {r["minimum"]} --max {r["maximum"]} --label eng1-{r["label"]} -- timeout --kill-after=10s '
                   f'{r["inner"]}s python3 run_gpu_tests.py {r["arm"]} {r["suite"]}')
        out.append('    ;;')
    out += ['  *) echo "unknown ENG1 label: $label" >&2; exit 64 ;;', 'esac', '']
    (TOOL / "STAGED_GPU_COMMANDS.sh").write_text("\n".join(out))
    os.chmod(TOOL / "STAGED_GPU_COMMANDS.sh", 0o755)
    print("script", sha(TOOL / "STAGED_GPU_COMMANDS.sh"))


ENG1_INNER = int(os.environ.get("ENG1_INNER", "900"))
REG_INNER = int(os.environ.get("ENG1_REG_INNER", "1050"))
MUT_INNER = int(os.environ.get("ENG1_MUT_INNER", "420"))
TER_INNER = int(os.environ.get("ENG1_TER_INNER", "1050"))

if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    sys.path.insert(0, str(TOOL))
    for step in sys.argv[1:]:
        dict(arms=step_arms, census=step_census, lines=step_lines, seal=step_seal, script=step_script)[step]()
