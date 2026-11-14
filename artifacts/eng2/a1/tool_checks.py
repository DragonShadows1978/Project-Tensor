#!/usr/bin/env python3
"""ENG2-A1 checks of the three fixed tool sites (REV1 E2R-F02, E2R-F03, E2R-F04), on CPU. Writes a1/TOOL_CHECKS.json.

PIN (E2R-F02, evidence.pin_failure_reasons via regression_reasons):
  P0  a REAL candidate receipt: the sealed child (tool/run_gpu_tests.py, unchanged) runs the candidate arm's regnorm
      suite with CUDA hidden, selecting the 10 pinned IDs by PYTEST_ADDOPTS (-k "test_e2_pow_grad and float64"), under
      a1/cpu_runner_standin.py, in the fresh root tool/gpu_runs/audit_2/cpu_pins. The baseline side is CONSTRUCTED
      (those 10 IDs passed), because the baseline runs them on the device (ENG1 G5 passed them on this engine).
      Expected: regression_reasons == [] (accepted from structured data only).
  P1..P7  copies of that real attempt with one structured field broken each; every one must be refused:
      P1 JUnit message = REV1's wrapped text (an AssertionError message quoting the registered text), class TypeError;
      P2 report class AssertionError, JUnit message exact (REV1's wrong_type_exact_text, on real data);
      P3 an extra failed teardown report for the pinned ID;    P4 the JUnit element is <error>, not <failure>;
      P5 no junit.xml;    P6 two <failure> elements;    P7 the JUnit message is the registered text plus a suffix.
MATCH (E2R-F03, evidence.matches), over the whole registered census tool/ids_eng2.txt:
  M1  for every mutant, the IDs the matcher accepts (raw arm-prefixed form and normalised form) == its killer set;
  M2  a pattern whose only difference is a parameter character ('?' or a bracket) matches nothing;
  M3  a test_eng1.py ID with a test_eng2.py killer's function and parameters is not matched.
CROSS (E2R-F04, score_gpu.keyed_rows and the unchanged main() loop, executed from its AST as REV1 did):
  C0  the real _record rows (a1/rev1_out/*_records): 35 and 20 rows, IDs equal, no difference, no census error;
  C1  a duplicate normalised ID is refused;  C2  a missing row is refused;  C3  an extra (unregistered) row is refused;
  C4  a row of another kind is refused;      C5  one changed candidate hash is reported as differing.
Prior art: negative (mutation-style) checks of a checker, DeMillo et al. (1978); no new algorithm.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

A1 = Path(__file__).resolve().parent
TOOL = A1.parent / "tool"
sys.path.insert(0, str(TOOL))
import evidence  # noqa: E402
import score_gpu  # noqa: E402

SCRATCH = Path("/tmp/eng2_a1_tc")
SCRATCH.mkdir(exist_ok=True)
out = {}

# ------------------------------------------------------------------------------------------------------------ PIN
RUNS = TOOL / "gpu_runs" / "audit_2" / "cpu_pins"
assert not RUNS.exists(), "fresh root required"
env = {k: v for k, v in os.environ.items() if k not in ("ENG2_AUDIT", "PYTHONOPTIMIZE")}
env.update(RUNS=str(RUNS), CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/eng2_a1_pins", PYTEST_ADDOPTS="-k cpu",
           PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1")
Path(env["TMPDIR"]).mkdir(exist_ok=True)
start = subprocess.run([sys.executable, "-B", "start_gpu_run.py"], cwd=TOOL, env=env, capture_output=True, text=True,
                       timeout=120)
assert start.returncode == 0, start.stderr
attempt = RUNS / "candidate_regnorm" / "attempt-0001"
attempt.mkdir(parents=True)
# The stand-in itself requires PYTEST_ADDOPTS == "-k cpu"; the child's own selection (the 10 pins) is set by env(1)
# inside the stand-in's command, so neither the stand-in nor the child is edited.
child = ["env", "PYTEST_ADDOPTS=-k 'test_e2_pow_grad and float64'", "timeout", "--kill-after=10s", "500s",
         sys.executable, "-B", "run_gpu_tests.py", "candidate", "regnorm"]
e = dict(env, ENG2_LINE="candidate_regnorm", ENG2_ATTEMPT=str(attempt), FORGE_GPU_RUN_LOG=str(attempt / "forge.log"))
with (attempt / "stdout.log").open("x") as log:
    rc = subprocess.run([sys.executable, "-B", str(A1 / "cpu_runner_standin.py"), "--label", "eng2a1-pins", "--",
                         "timeout", "--kill-after=10s", "540s", *child], cwd=TOOL, env=e, stdout=log,
                        stderr=subprocess.STDOUT, timeout=570).returncode
new = evidence.read(attempt / "child_result.json")
pins = evidence.read(A1.parent / "L38_PINS.json")["expected_changes"]
ids = new["collected_ids"]
assert sorted(evidence.norm_id(i) for i in ids) == sorted(pins), "the real receipt must hold exactly the 10 pins"
old = dict(collected_ids=ids, reports=[dict(id=i, when=w, outcome="passed", detail="", exception_class=None)
                                       for i in ids for w in ("setup", "call", "teardown")])
p0 = evidence.regression_reasons(old, new)
out["P0"] = dict(child_exit=rc, collected=len(ids), exception_classes=sorted({r["exception_class"] for r in new["reports"]
                                                                              if r["outcome"] == "failed"}),
                 reasons=p0, ok=(rc == 1 and p0 == []))
registered = next(iter(pins.values()))["detail_contains"]
first = sorted(ids)[0]


def variant(name, edit_result=None, edit_xml=None, drop_xml=False):
    d = SCRATCH / name
    shutil.copytree(attempt, d)
    data = evidence.read(d / "child_result.json")
    data["attempt_dir"] = str(d)
    if edit_result:
        edit_result(data)
    (d / "child_result.json").write_text(json.dumps(data))
    if drop_xml:
        (d / "junit.xml").rename(d / "junit.xml.withheld")
    elif edit_xml:
        tree = ET.parse(d / "junit.xml")
        case = next(c for c in tree.getroot().iter("testcase") if (c.get("classname"), c.get("name")) == evidence.xml_key(first))
        edit_xml(case)
        tree.write(d / "junit.xml")
    reasons = evidence.regression_reasons(old, data)
    return dict(reasons=reasons, ok=bool(reasons) and all("L38" in r for r in reasons))


def set_message(text):
    def f(case):
        next(e for e in case if e.tag == "failure").set("message", text)
    return f


def set_class(cls):
    def f(data):
        for r in data["reports"]:
            if r["id"] == first and r["when"] == "call":
                r["exception_class"] = cls
    return f


def extra_teardown(data):
    for r in data["reports"]:
        if r["id"] == first and r["when"] == "teardown":
            r.update(outcome="failed", detail="teardown failed", exception_class="RuntimeError")


def to_error(case):
    next(e for e in case if e.tag == "failure").tag = "error"


def two_failures(case):
    f = next(e for e in case if e.tag == "failure")
    case.append(ET.Element("failure", dict(f.attrib)))


out["P1"] = variant("P1_wrapped_other_reason", edit_xml=set_message(
    "AssertionError: expected error was " + repr(registered) + " but received corrupt output"))
out["P2"] = variant("P2_wrong_type_exact_text", edit_result=set_class("AssertionError"))
out["P3"] = variant("P3_extra_teardown_failure", edit_result=extra_teardown)
out["P4"] = variant("P4_error_element", edit_xml=to_error)
out["P5"] = variant("P5_no_junit", drop_xml=True)
out["P6"] = variant("P6_two_failures", edit_xml=two_failures)
out["P7"] = variant("P7_suffix", edit_xml=set_message(registered + " (and more)"))

# ---------------------------------------------------------------------------------------------------------- MATCH
census = (TOOL / "ids_eng2.txt").read_text().split()
mutants = json.loads((A1.parent / "mutants" / "mutants.json").read_text())
m1 = {}
for m in mutants:
    killers = sorted(evidence.norm_id(k) for k in m["killers"])
    raw = "artifacts/eng2/tool/arms/%s/" % m["id"]
    by_norm = sorted(i for i in census if evidence.matches(i, m["killers"]))
    by_raw = sorted(i for i in census if evidence.matches(raw + i, m["killers"]))
    m1[m["id"]] = dict(killers=len(killers), matched=len(by_norm), equal=by_norm == killers == by_raw)
out["M1"] = dict(per_mutant=m1, ok=all(v["equal"] for v in m1.values()))
k = "tensor_cuda/tests/test_eng2.py::test_f1_refuse_cpu[tensor-float64-cpu]"
probes = [k.replace("float64", "float6?"), k.replace("[", "?", 1), k.replace("[tensor", "[[tensor")]
out["M2"] = dict(patterns=probes, matched=[i for i in census for p in probes if evidence.matches(i, [p])],
                 ok=not any(evidence.matches(i, [p]) for i in census for p in probes))
other = k.replace("test_eng2.py", "test_eng1.py")
out["M3"] = dict(id=other, matched=evidence.matches(other, [k]), ok=not evidence.matches(other, [k]))

# ---------------------------------------------------------------------------------------------------------- CROSS
REC = A1 / "rev1_out"


def load(arm, kind):
    return [json.loads(p.read_text()) for p in sorted((REC / (arm + "_records")).glob(kind + "_*.json"))]


main = next(n for n in ast.parse((TOOL / "score_gpu.py").read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "kind")


def run_loop(edit=None):
    data = {(arm, kind): load(arm, kind) for arm in ("baseline", "candidate") for kind in ("f1bytes", "p302bits")}
    if edit:
        edit(data)
    ns = dict(vars(score_gpu), cross={}, summary={"constructed": True}, errors=[],
              rows=lambda label, pattern: data[(label.split("_")[0], pattern.split("_")[0])])
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(TOOL / "score_gpu.py"), "exec"), ns)
    return dict(errors=[e[:400] for e in ns["errors"]],
                cross={k: {q: v for q, v in c.items() if q != "differing"} | dict(differing=c["differing"][:3])
                       for k, c in ns["cross"].items()})


c0 = run_loop()
out["C0"] = dict(c0, ok=(not c0["errors"] and all(c["ids_equal"] and not c["differing"] and not c["census_errors"]
                                                    for c in c0["cross"].values())
                         and c0["cross"]["f1bytes"]["baseline"] == 35 and c0["cross"]["p302bits"]["baseline"] == 20))


def dup(d):
    d[("candidate", "f1bytes")].append(d[("baseline", "f1bytes")][0])  # the baseline's row: same normalised ID


def missing(d):
    d[("candidate", "p302bits")].pop()
    d[("baseline", "p302bits")].pop()


def extra(d):
    for arm in ("baseline", "candidate"):
        r = dict(d[(arm, "f1bytes")][0])
        r["test_id"] = r["test_id"].split("::")[0] + "::test_f1_unregistered_row"
        d[(arm, "f1bytes")].append(r)


def wrong_kind(d):
    for arm in ("baseline", "candidate"):
        d[(arm, "p302bits")][0] = dict(d[(arm, "p302bits")][0], kind="f1bytes")


def changed(d):
    d[("candidate", "f1bytes")][0] = dict(d[("candidate", "f1bytes")][0], sha256="2" * 64)


for name, edit, needle in (("C1", dup, "duplicate normalised ID"), ("C2", missing, "differ from the census"),
                           ("C3", extra, "differ from the census"), ("C4", wrong_kind, "row of kind"),
                           ("C5", changed, "differing ['tensor_cuda/tests/test_eng2.py::")):
    r = run_loop(edit)
    out[name] = dict(r, ok=any(needle in e for e in r["errors"]))

out["all_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict) and "ok" in v)
(A1 / "TOOL_CHECKS.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: (v["ok"] if isinstance(v, dict) and "ok" in v else v) for k, v in out.items()}))
print("P0", json.dumps(out["P0"]))
for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "C1", "C2", "C3", "C4", "C5"):
    print(k, json.dumps(out[k].get("reasons") or out[k].get("errors"))[:330])
sys.exit(0 if out["all_ok"] else 1)
