#!/usr/bin/env python3
"""ENG2 staging generator (CPU only), copied from ENG1's tool/stage.py and re-targeted. ENG1's step structure is kept
(arms, census, lines, seal, script); the bodies change as ledgered (artifacts/eng2/LEDGER.md, COPY_DEPS.json):
  registry  write artifacts/eng2/mutants/mutants.json once (open 'x'): the 15 registered mutants, each one edit (one
            contiguous replacement inside one class's forward or one F1 line), its hunk lines in the mutated file, and
            its predicted killer IDs (F1: from the code; N1: artifacts/eng2/N1_EMULATION.json)
  arms      arms/<arm>/tensor_cuda/{tensor_cuda/<package + pinned module>, tests/<test_eng2.py, test_eng1.py>}:
            baseline = artifacts/eng2/base_pkg (bc02d3a1's package, checked against worktree_before.json) + the
            build-eng2/base module; candidate = the worktree package + the build-eng2/cand module; each mutant = the
            candidate with its one edit (copies, never links). ENG1 took the baseline from `git show`; ENG2 runs no git.
  census    collect each suite's test IDs on CPU with the arm package preloaded (ids_<census>.txt)
  lines     write gpu_lines.json, regression_files.json, baseline_expected_failures.json
  seal      write staging_seal.json (order, tools, registration files, arm maps, engine pins)
  script    write STAGED_GPU_COMMANDS.sh
Prior art: GN1 A8 staging and ENG1's stage.py (2026), re-targeted; SHA-256 pins (NIST FIPS 180-4); DeMillo, Lipton
and Sayward (1978) for the mutant registry. No new algorithm.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
ENG2 = TOOL.parent
ROOT = TOOL.parents[2]
ORDER = "orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md"
ORDER_COMMIT = "dfdbc974429e942281a55f2ece64e69fa1235e16"  # read from the ref file refs/heads/eng2 (no git command)
BASE_COMMIT = "bc02d3a1b150572af5521725d60b1d0d54305681"
SEAL_NAME = "staging_seal.json"
MODULE = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
BASE_MODULE = ROOT / "tensor_cuda/build-eng2/base/module" / MODULE
CAND_MODULE = ROOT / "tensor_cuda/build-eng2/cand/module" / MODULE
P = "tensor_cuda/tests/test_eng2.py::"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ the mutant list
INIT, NN = "tensor_cuda/tensor_cuda/__init__.py", "tensor_cuda/tensor_cuda/nn.py"
SCOPES = {"LN": "LayerNorm", "RMS": "RMSNorm", "BN": "BatchNorm1D", "IN": "InstanceNorm2D"}
MN1 = {"LN": ("        return (xc * inv * weight + bias).astype(dtype)", "        return xc * inv * weight + bias"),
       "RMS": ("        return out.astype(dtype) if half else out", "        return out"),
       "BN": ("        return out.astype(dtype) if half else out", "        return out"),
       "IN": ("        return (xn * weight.reshape([1, self.C, 1, 1]) + bias.reshape([1, self.C, 1, 1])).astype(dtype)",
              "        return xn * weight.reshape([1, self.C, 1, 1]) + bias.reshape([1, self.C, 1, 1])")}
MN2 = ('        half = dtype in ("float16", "bfloat16")', '        half = dtype in ("float16",)')
MN3 = {"LN": ('        if half:\n            x = x.astype("float32")\n        mean = x.mean([-1], True)\n'
              '        xc = x - mean\n        var = (xc * xc).mean([-1], True)\n        inv = (var + self.eps).pow(-0.5)',
              '        mean = x.mean([-1], True)\n        xc = x - mean\n        var = (xc * xc).mean([-1], True)\n'
              '        if half:\n            xc, var = xc.astype("float32"), var.astype("float32")\n'
              '        inv = (var + self.eps).pow(-0.5)'),
       "RMS": ('        if half:\n            x = x.astype("float32")\n        ms = (x * x).mean([-1], True)',
               '        ms = (x * x).mean([-1], True)\n        if half:\n'
               '            x, ms = x.astype("float32"), ms.astype("float32")'),
       "BN": ('            mean = x.mean(axes, True); var = x.var(axes, True)',
              '            mean = x.astype(dtype).mean(axes, True).astype("float32"); '
              'var = x.astype(dtype).var(axes, True).astype("float32")'),
       "IN": ('        if half:\n            x = x.astype("float32")\n        mean = x.mean([2, 3], True); var = x.var([2, 3], True)',
              '        mean = x.mean([2, 3], True); var = x.var([2, 3], True)\n        if half:\n'
              '            x, mean, var = x.astype("float32"), mean.astype("float32"), var.astype("float32")')}
MN4 = ("            return xc * inv * self.weight + self.bias",
       '            return (xc * inv).astype("float16").astype("float32") * self.weight + self.bias')
F1_ENTRIES, F1_REFUSED, DEVICES = ["tensor", "zeros", "ones", "randn", "rand"], ["float64", "double", "int32", "not_a_dtype"], ["cpu", "cuda"]


def mutant_defs():
    defs = [dict(id="MF1", decision="F1: the silent fallback restored at __init__.py:48 (tc.tensor)", file=INIT, scope=None,
                 old="    arr = np.asarray(data, dtype=_np_dtype(dtype))  # ENG2 F1: refuses before any allocation",
                 new="    arr = np.asarray(data, dtype=_NP_DTYPE.get(dtype, np.float32))",
                 killers=[P + f"test_f1_refuse_cpu[tensor-{d}-{v}]" for d in F1_REFUSED for v in DEVICES]
                 + [P + f"test_f1_tensor_refuses_before_conversion_cpu[{d}]" for d in F1_REFUSED],
                 predicted_by="code reading; shown on CPU (artifacts/eng2/cpu_mutant_kills.json)"),
            dict(id="MF2", decision="F1: the silent fallback restored at __init__.py:60 (_factory)", file=INIT, scope=None,
                 old="        np_dtype = _np_dtype(dtype)  # ENG2 F1: refuses before np_fn allocates (or draws random numbers)",
                 new="        np_dtype = _NP_DTYPE.get(dtype, np.float32)",
                 killers=[P + f"test_f1_refuse_cpu[{e}-{d}-{v}]" for e in F1_ENTRIES[1:] for d in F1_REFUSED for v in DEVICES]
                 + [P + f"test_f1_factory_refuses_before_allocation_cpu[{e}-{d}]" for e in F1_ENTRIES[1:] for d in F1_REFUSED],
                 predicted_by="code reading; shown on CPU (artifacts/eng2/cpu_mutant_kills.json)")]
    emu = json.loads((ENG2 / "N1_EMULATION.json").read_text())["predicted_failures"]
    for k, (old, new, what) in [*((f"MN1_{s}", (*MN1[s], "the cast back omitted (output float32)")) for s in SCOPES),
                               *((f"MN2_{s}", (*MN2, "the upcast only for float16, not bfloat16")) for s in SCOPES),
                               *((f"MN3_{s}", (*MN3[s], "the statistics still in the input dtype")) for s in SCOPES),
                               ("MN4_LN", (*MN4, "the float32 path given an extra cast (a float16 round trip)"))]:
        short = k.split("_")[1]
        defs.append(dict(id=k, decision=f"N1 {SCOPES[short]}: {what}", file=NN, scope=SCOPES[short], old=old, new=new,
                         killers=emu[k], predicted_by="artifacts/eng2/N1_EMULATION.json (CPU emulation)"))
    return defs


def scope_span(text, scope):
    """(start, end) character span of `class <scope>(Module):` up to the next top-level class (or the file end)."""
    if scope is None:
        return 0, len(text)
    start = text.index(f"\nclass {scope}(Module):") + 1
    nxt = text.find("\nclass ", start)
    return start, (len(text) if nxt < 0 else nxt + 1)


def apply_mutant(text, m):
    a, b = scope_span(text, m["scope"])
    region = text[a:b]
    assert region.count(m["old"]) == 1, (m["id"], region.count(m["old"]))
    assert text.count(m["old"]) == 1 or m["scope"] is not None, m["id"]
    pos = a + region.index(m["old"])
    mutated = text[:pos] + m["new"] + text[pos + len(m["old"]):]
    first = mutated[:pos].count("\n") + 1
    return mutated, [first, first + m["new"].count("\n")]


def step_registry():
    out = ENG2 / "mutants" / "mutants.json"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for m in mutant_defs():
        _, hunk = apply_mutant((ROOT / m["file"]).read_text(), m)
        rows.append(dict(m, kind="python", hunk_lines=hunk))
    with out.open("x") as f:
        f.write(json.dumps(rows, indent=1) + "\n")
    print("registry", len(rows), "mutants", sha(out))


def registry():
    return json.loads((ENG2 / "mutants" / "mutants.json").read_text())


# ---------------------------------------------------------------------------------------------------------- arms
def make_arm(arm, package_src, module_src, edit=None):
    dest = TOOL / "arms" / arm / "tensor_cuda"
    assert not dest.exists(), "arms are written once: " + arm
    pkg, tests = dest / "tensor_cuda", dest / "tests"
    for src in sorted(Path(package_src).rglob("*.py")):
        if "__pycache__" in src.parts:
            continue
        target = pkg / src.relative_to(package_src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
    shutil.copyfile(module_src, pkg / MODULE)
    os.chmod(pkg / MODULE, 0o755)
    tests.mkdir(parents=True)
    for name in ("test_eng2.py", "test_eng1.py"):
        shutil.copyfile(ROOT / "tensor_cuda/tests" / name, tests / name)
    if edit is not None:
        path = dest.parent / edit["file"]
        mutated, hunk = apply_mutant(path.read_text(), edit)
        assert hunk == edit["hunk_lines"], (arm, hunk)
        path.write_text(mutated)
    print("built", arm, sha(pkg / MODULE))


def step_arms():
    before = json.loads((ENG2 / "worktree_before.json").read_text())["files"]
    base_pkg = ENG2 / "base_pkg" / "tensor_cuda"
    for src in base_pkg.rglob("*.py"):  # the baseline package is bc02d3a1's (pre-edit map)
        assert sha(src) == before["tensor_cuda/tensor_cuda/" + str(src.relative_to(base_pkg))], src
    for name in ("test_eng1.py",):
        assert sha(ROOT / "tensor_cuda/tests" / name) == before["tensor_cuda/tests/" + name], name
    make_arm("baseline", base_pkg, BASE_MODULE)
    make_arm("candidate", ROOT / "tensor_cuda/tensor_cuda", CAND_MODULE)
    for m in registry():
        make_arm(m["id"], ROOT / "tensor_cuda/tensor_cuda", CAND_MODULE, edit=m)


# -------------------------------------------------------------------------------------------------------- census
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
extra = os.environ.get("ENG2_COLLECT_K")
args = ["-q", "-p", "no:cacheprovider", os.environ["ENG2_IMPORT_MODE"], "--collect-only", "--rootdir=" + os.environ["ENG2_ROOT"]]
if extra: args += ["-k", extra]
rc = pytest.main(args + files, plugins=[P()])
Path(out).write_text(json.dumps(dict(rc=int(rc), ids=P.ids, errors=P.errors), indent=1) + "\n")
'''


def norm(i):
    import re
    m = re.match(r"artifacts/eng2/tool/arms/[^/]+/(tensor_cuda/.*)$", i)
    return m.group(1) if m else i


def collect(arm, suite, k=None):
    pkg = TOOL / "arms" / arm / "tensor_cuda"
    if suite in ("regnorm", "regdtype"):
        files = [str(ROOT / f) for f in json.loads((TOOL / "regression_files.json").read_text())[suite]]
    else:
        files = [str(pkg / "tests/test_eng2.py")]
    tag = "_cpu" if k == "cpu" else ""
    out = Path(os.environ["TMPDIR"]) / f"collect_{arm}_{suite}{tag}.json"
    env = dict(os.environ, ENG2_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               ENG2_IMPORT_MODE="--import-mode=prepend" if suite in ("regnorm", "regdtype") else "--import-mode=importlib")
    from arm_origin import bound_environment
    env = bound_environment(pkg, env)
    if k:
        env["ENG2_COLLECT_K"] = k
    p = subprocess.run([sys.executable, "-B", "-c", COLLECT, str(pkg), str(out), *files], env=env, cwd=os.environ["TMPDIR"],
                       capture_output=True, text=True, timeout=500)
    data = json.loads(out.read_text())
    (TOOL / f"collect_{arm}_{suite}{tag}.json").write_text(json.dumps(
        dict(rc=data["rc"], n=len(data["ids"]), errors=data["errors"], stderr_tail=p.stderr[-2000:]), indent=1) + "\n")
    return data


def step_census():
    # regression_files.json first: the regnorm/regdtype collections read it (ENG1 wrote its list by hand).
    (TOOL / "regression_files.json").write_text(json.dumps(regression_files(), indent=1) + "\n")
    results = {}
    for arm, suite, k, name in (("baseline", "eng2", None, "eng2"), ("candidate", "eng2", None, "eng2_candidate"),
                                ("candidate", "eng2", "cpu", "audit"),
                                ("baseline", "regnorm", None, "regnorm"), ("candidate", "regnorm", None, "regnorm_candidate"),
                                ("baseline", "regdtype", None, "regdtype"),
                                ("candidate", "regdtype", None, "regdtype_candidate")):
        data = collect(arm, suite, k)
        ids = sorted(norm(i) for i in data["ids"])
        results[name] = (data["rc"], len(ids), len(data["errors"]))
        (TOOL / f"ids_{name}.txt").write_text("".join(i + "\n" for i in ids))
    for a, b in (("eng2", "eng2_candidate"), ("regnorm", "regnorm_candidate"), ("regdtype", "regdtype_candidate")):
        assert (TOOL / f"ids_{a}.txt").read_text() == (TOOL / f"ids_{b}.txt").read_text(), a + " IDs differ by arm"
    print(json.dumps(results))


# --------------------------------------------------------------------------------------------------------- lines
REGRESSION = {
    # line (c): files that build one of the five norm classes ...
    "regnorm": ["tensor_cuda/tests/test_eng1.py", "tensor_cuda/tests/test_groupnorm_gn1.py",
                "tensor_cuda/tests/test_attention_phase5.py", "tensor_cuda/tests/test_nn_phase3.py",
                "tensor_cuda/tests/test_ext_phase7.py"],
    # ... and every other existing test file that calls tc.tensor or a factory with a non-default dtype
    # (artifacts/eng2/P303_caller_census.json, the test files with a dtype other than the literal "float32").
    "regdtype": None,
}


def regression_files():
    census = json.loads((ENG2 / "P303_caller_census.json").read_text())
    root = str(ROOT) + "/"
    files = set()
    for r in census["rows"]:
        f = r["file"]
        if not f.startswith(root):
            continue
        rel = f[len(root):]
        base = os.path.basename(rel)
        if not (base.startswith("test_") and rel.endswith(".py")) or r["call"].split(".")[0] not in ("tc", "TC", "tensor_cuda"):
            continue
        if r["call"].split(".")[-1] == "parameter" or r["dtype"] in ("'float32'", '"float32"'):
            continue
        files.add(rel)
    out = dict(regnorm=list(REGRESSION["regnorm"]),
               regdtype=sorted(files - set(REGRESSION["regnorm"])))
    return out


def bounds(inner):
    return dict(inner=inner, minimum=inner + 10, maximum=inner + 60, outer=inner + 90)


def step_lines():
    assert json.loads((TOOL / "regression_files.json").read_text()) == regression_files()
    (TOOL / "source_reading_tests.json").write_text("[]\n")  # L38 trace: no line-(c) test reads nn.py/__init__.py
    rows = [dict(label="baseline_eng2", arm="baseline", suite="eng2", role="baseline", census="eng2",
                 expected_child_exits=[1], expected_parent_exit=0, **bounds(ENG2_INNER)),
            dict(label="candidate_eng2", arm="candidate", suite="eng2", role="candidate", census="eng2",
                 expected_child_exits=[0], expected_parent_exit=0, **bounds(ENG2_INNER))]
    for suite, inner in (("regnorm", REGNORM_INNER), ("regdtype", REGDTYPE_INNER)):
        rows.append(dict(label="baseline_" + suite, arm="baseline", suite=suite, role="regression", census=suite,
                         expected_child_exits=[0, 1], expected_parent_exit=0, **bounds(inner)))
        rows.append(dict(label="candidate_" + suite, arm="candidate", suite=suite, role="regression",
                         census=suite + "_candidate", expected_child_exits=[0, 1], expected_parent_exit=0, **bounds(inner)))
    for m in registry():
        rows.append(dict(label=m["id"], arm=m["id"], suite="mutation", role="mutant", census="eng2",
                         expected_child_exits=[1], expected_parent_exit=0, **bounds(ENG2_INNER)))
    (TOOL / "gpu_lines.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Registered before any GPU run: the baseline arm's expected failures, per ID. CPU IDs: the baseline arm's CPU run
    # (artifacts/eng2/g2/baseline_cpu.json); N1 IDs: the emulation; every other ID (F1 device tests) passes.
    ids = (TOOL / "ids_eng2.txt").read_text().split()
    cpu = json.loads((ENG2 / "g2" / "baseline_cpu.json").read_text())
    emu = set(json.loads((ENG2 / "N1_EMULATION.json").read_text())["predicted_failures"]["baseline"])
    fail = []
    for i in ids:
        name = i.split("::", 1)[1]
        if "cpu" in name.split("[")[0]:
            assert i in cpu["outcomes"], i
            if cpu["outcomes"][i] != "passed":
                fail.append(i)
        elif i in emu:
            fail.append(i)
    (TOOL / "baseline_expected_failures.json").write_text(json.dumps(sorted(fail), indent=1) + "\n")
    print(len(rows), "lines;", len(fail), "registered baseline failures of", len(ids))


# ---------------------------------------------------------------------------------------------------------- seal
def step_seal():
    arms = sorted(p.name for p in (TOOL / "arms").iterdir())
    from evidence import arm_files
    reg = json.loads((TOOL / "regression_files.json").read_text())
    seal = dict(order=ORDER, order_commit=ORDER_COMMIT, order_sha256=sha(ROOT / ORDER), base_commit=BASE_COMMIT,
                module_name=MODULE,
                tool_files={p.name: sha(p) for p in sorted(TOOL.iterdir())
                            if p.is_file() and not p.name.startswith(("staging_seal", "collect_"))},
                registration_files={str(p.relative_to(ROOT)): sha(p) for p in
                                    [ROOT / "tensor_cuda/tests/test_eng2.py", ROOT / "tensor_cuda/tests/test_eng1.py",
                                     ENG2 / "mutants/mutants.json", ENG2 / "N1_EMULATION.json",
                                     ENG2 / "g2/baseline_cpu.json", ENG2 / "L38_PINS.json"] +
                                    [ROOT / f for suite in sorted(reg) for f in reg[suite]]},
                arms={a: arm_files(a) for a in arms},
                engine_sha256={a: sha(TOOL / "arms" / a / "tensor_cuda/tensor_cuda" / MODULE) for a in arms})
    for a in arms:  # every arm runs the worktree's current test_eng2.py and its test_eng1.py dependency
        for name in ("test_eng2.py", "test_eng1.py"):
            assert sha(TOOL / "arms" / a / "tensor_cuda/tests" / name) == sha(ROOT / "tensor_cuda/tests" / name), (a, name)
    labels = {r["arm"] for r in json.loads((TOOL / "gpu_lines.json").read_text())}
    missing = sorted(labels - set(arms))
    assert not missing, missing
    seal["arms_pending"] = missing
    with (TOOL / SEAL_NAME).open("x") as f:
        f.write(json.dumps(seal, indent=1, sort_keys=True) + "\n")
    print("sealed arms:", arms, "seal", sha(TOOL / SEAL_NAME))


# -------------------------------------------------------------------------------------------------------- script
def step_script():
    rows = json.loads((TOOL / "gpu_lines.json").read_text())
    out = ['#!/usr/bin/env bash',
           '# ENG2 staged GPU lines (copied from ENG1 A3 / GN1 A8 tooling; the lead runs them; the seat never did on a device).',
           '# Prior art: GNU timeout, POSIX shell; GN1 L45 completed-child supervision (parent receipt after child exit).',
           '# Expected parent exit for EVERY line is 0 (the parent passes when the child exit matches the registration',
           '# and the receipts check out). Child exits and bounds are listed per label below and in gpu_lines.json.',
           '# Usage: RUNS=/abs/path/under/artifacts/eng2/tool/gpu_runs/<root> TMPDIR=/tmp/eng2_g5 \\',
           '#        CUDA_VISIBLE_DEVICES=0 timeout 1190s bash STAGED_GPU_COMMANDS.sh LABEL',
           '# Run the labels in gpu_lines.json order with the SAME RUNS (mutants need a passing candidate_eng2;',
           '# candidate_regnorm needs baseline_regnorm, candidate_regdtype needs baseline_regdtype).',
           '# Afterwards: RUNS=<same> CUDA_VISIBLE_DEVICES= \\',
           '#        timeout 300s python3 score_gpu.py   (expected exit 0).',
           '# CPU plumbing audit: RUNS=/abs/audit/root TMPDIR=/tmp/eng2_audit timeout 1190s bash STAGED_GPU_COMMANDS.sh --audit',
           'set -euo pipefail',
           ': "${RUNS:?RUNS is required; use a fresh run root and reuse it across the labels}"',
           '[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }',
           'TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
           'cd -- "$TOOL"',
           'export RUNS PYTHONDONTWRITEBYTECODE=1',
           'export TMPDIR="${TMPDIR:?TMPDIR must be /tmp/eng2_*}"',
           'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
           'GPU_RUN=forge-gpu-run',
           'label="${1:?select one registered label, or --audit}"',
           'if [[ "$label" == --audit ]]; then',
           '    export ENG2_AUDIT=1 CUDA_VISIBLE_DEVICES=',
           '    GPU_RUN="$TOOL/forge_gpu_run_standin.py"',
           '    label=candidate_eng2',
           'else',
           '    unset ENG2_AUDIT',
           'fi',
           'mkdir -p -- "$TMPDIR"',
           'timeout 60s python3 start_gpu_run.py',
           'case "$label" in']
    for r in rows:
        exp = ",".join(str(v) for v in r["expected_child_exits"])
        out.append(f'  {r["label"]}) # expected child {exp}, parent 0; inner {r["inner"]}s, --max {r["maximum"]}s, outer {r["outer"]}s')
        out.append(f'    timeout {r["outer"] + 20}s python3 run_gpu_line.py {r["label"]} {exp} {r["outer"]} "$GPU_RUN" '
                   f'--min {r["minimum"]} --max {r["maximum"]} --label eng2-{r["label"]} -- timeout --kill-after=10s '
                   f'{r["inner"]}s python3 run_gpu_tests.py {r["arm"]} {r["suite"]}')
        out.append('    ;;')
    out += ['  *) echo "unknown ENG2 label: $label" >&2; exit 64 ;;', 'esac', '']
    (TOOL / "STAGED_GPU_COMMANDS.sh").write_text("\n".join(out))
    os.chmod(TOOL / "STAGED_GPU_COMMANDS.sh", 0o755)
    print("script", sha(TOOL / "STAGED_GPU_COMMANDS.sh"))


ENG2_INNER = int(os.environ.get("ENG2_INNER", "600"))
REGNORM_INNER = int(os.environ.get("ENG2_REGNORM_INNER", "600"))
REGDTYPE_INNER = int(os.environ.get("ENG2_REGDTYPE_INNER", "1050"))

if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    sys.path.insert(0, str(TOOL))
    for step in sys.argv[1:]:
        dict(registry=step_registry, arms=step_arms, census=step_census, lines=step_lines, seal=step_seal,
             script=step_script)[step]()
