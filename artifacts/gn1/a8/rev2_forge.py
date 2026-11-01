#!/usr/bin/env python3
"""GN1-REV2 CPU stand-in for forge-gpu-run + child (Part C only). Never leases,
never runs the GPU, never imports the engine. Writes a fully bound child
result, a JUnit file that agrees with it ID by ID, and the installed runner's
nine-field log row, so exactly one targeted defect reaches A2's verifier.
Written independently of A2's hostile_forge.py (shapes taken from evidence.py).

Prior art: test stubs (Meszaros 2007). Ours: the modes.
"""
import datetime
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
TOOL = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL))
from evidence import census, xml_key, lines  # noqa: E402  (mirror copy, sealed bytes)

mode = os.environ.get("REV2_MODE", "normal")
label = os.environ["GN1_LINE"]
line = lines()[label]
att = Path(os.environ["GN1_ATTEMPT"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = json.loads((TOOL / "staging_seal.json").read_text())
pin = json.loads((TOOL / "engine_pin.json").read_text())
# A7 D2: child-owned identity; Meszaros (2007) explicit stand-in.
from device_evidence import capture
from runtime_evidence import atomic_json, module_census, npbool_index_check
atomic_json(att / "child_started.json", dict(label=label, pid=os.getpid()))
if mode == "started_no_result":
    atomic_json(att / "invocation.json", dict(arm=line["arm"], suite=line["suite"]))
    time.sleep(10)
    sys.exit(0)
arm, suite = line["arm"], line["suite"]
ids = census(arm, suite)
reports = {i: dict(id=i, when="call", outcome="passed", detail="", exception_class=None, groupnorm_refusal=False)
           for i in ids}


def fail(i, detail="AssertionError: forward values", cls="AssertionError"):
    reports[i].update(outcome="failed", detail=detail, exception_class=cls)


gn = [i for i in ids if "test_groupnorm_gn1.py::test_forward_values[" in i]
legacy = [i for i in ids if "test_groupnorm_gn1.py" not in i]
if suite == "mutation" and not line.get("equivalence"):
    fail(gn[0])
if label == "baseline_focus":
    for i in json.loads((TOOL / "baseline_expected_failures.json").read_text()):
        new_expected = json.loads((TOOL/"baseline_new_failures.json").read_text())
        cls = new_expected.get(i, {}).get("exception_class", "AssertionError")
        fail(i, "GroupNorm baseline registered D5 failure" if i in new_expected else "AssertionError: GroupNorm refused a registered valid input: reshape: element count mismatch", cls)
if suite == "regression":
    fail(legacy[0], "RuntimeError: shared legacy failure", "RuntimeError")
if mode == "hang_failure":
    fail(gn[0])
if mode == "v3_failure":
    fail(gn[0])
if mode == "skip_one":
    reports[gn[1]].update(outcome="skipped")
if mode == "baseline_extra":
    fail(next(i for i in ids if i not in json.loads((TOOL / "baseline_expected_failures.json").read_text())),
         "AssertionError: GroupNorm refused a registered valid input: reshape: element count mismatch")
# REV1 payloads with A2's bindings added (REV1 hostile_scorer.py STANDIN semantics).
if mode in ("rev1_h2", "rev1_h2c"):
    text = ("RuntimeError: CUDA error at reduce_lastdim: an illegal memory access was encountered" if mode == "rev1_h2"
            else "RuntimeError: cudaErrorIllegalAddress")
    for i in ids:
        fail(i, text, "RuntimeError")
if mode == "rev1_h3":  # child ran the BASELINE arm: reshape errors on value tests, the rest pass
    arm = "baseline"
    for i in ids:
        reports[i].update(outcome="passed", detail="", exception_class=None)
        if "values" in i:
            fail(i, "RuntimeError: reshape: element count mismatch", "RuntimeError")
if mode == "rev1_h3b":  # mutant line whose child claims a passing candidate focus
    arm, suite = "candidate", "focus"
    ids = census(arm, suite)
    reports = {i: dict(id=i, when="call", outcome="passed", detail="", exception_class=None, groupnorm_refusal=False)
               for i in ids}
# New REV2 cases for the kill rule (R1-F01/F02):
if mode == "kill_plus_teardown_device":  # honest kill; a device fault appears only in a teardown phase
    reports_extra = dict(id=gn[1], when="teardown", outcome="failed",
                         detail="RuntimeError: CUDA error at sync: unspecified launch failure",
                         exception_class="RuntimeError", groupnorm_refusal=False)
if mode == "kill_device_word_in_assert":  # honest AssertionError whose text merely mentions a pattern
    fail(gn[0], "AssertionError: forward values\nassert False\n  where note = 'checked: no out of memory here'")
collected = list(ids)
if mode == "census_drop":
    collected = collected[1:]
    reports.pop(ids[0])
rows = list(reports.values())
if mode == "kill_plus_teardown_device":
    rows.append(reports_extra)
rc = int(any(r["outcome"] == "failed" for r in rows))
if mode == "exit0_with_failure":
    rc = 0
data = dict(arm=arm, suite=suite, audit=False, cwd=str(TOOL), run_root=os.environ["RUNS"], attempt_dir=str(att),
            seal_sha256=sha(TOOL / "staging_seal.json"), engine_sha256=pin["sha256"],
            nn_sha256=seal["arms"][arm]["nn.py"], CUDA_VISIBLE_DEVICES="", files=[], pytest_exit=rc,
            collected_ids=collected, collection_errors=[], reports=rows)
# A8 D2: use the sealed child's own capture when it invokes this fixture.
# Prior art: A7 capture plus Python runpy init_globals (PSF).
data["device_evidence"] = globals().get("GN1_CHILD_DEVICE_EVIDENCE") or capture(__file__)
# A7 D2: child-owned pre-result record, also present in CPU fixtures.
atomic_json(att / "invocation.json", dict(arm=arm, suite=suite, device_evidence=data["device_evidence"]))
data["runtime_modules"] = module_census(TOOL.parents[2])
data["equivalence_runtime"] = npbool_index_check() if line.get("equivalence") and label != "V3" else None
if mode == "sealed_tamper":
    import runtime_sentinel
    target = Path(runtime_sentinel.__file__)
    target.write_bytes(target.read_bytes() + b"# tampered at runtime\n")
    data["runtime_modules"] = module_census(TOOL.parents[2])
if mode == "unsealed":
    path = Path(os.environ["TMPDIR"]) / "device_only_unsealed.py"
    path.write_text("# CPU control for a device-only import origin\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location("device_only", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["device_only"] = module
    spec.loader.exec_module(module)
    data["runtime_modules"] = module_census(TOOL.parents[2])
if mode == "equivalence_accepts":
    data["equivalence_runtime"]["accepts_numpy_bool"] = True
if mode == "truncated":
    (att / "child_result.json").write_text('{"pytest_exit":')
else:
    atomic_json(att / "child_result.json", data)
root = ET.Element("testsuites")
ts = ET.SubElement(root, "testsuite", name="pytest")
for i in collected:  # pytest 9 junitxml shape: one entry per ID, two only for call failure + teardown error
    rs = [r for r in rows if r["id"] == i]
    call = next((r for r in rs if r["when"] == "call"), None)
    errs = [r for r in rs if r["when"] != "call" and r["outcome"] == "failed"]
    cls, name = xml_key(i)
    case = ET.SubElement(ts, "testcase", classname=cls, name=name)
    if call and call["outcome"] == "failed":
        ET.SubElement(case, "failure", message=call["detail"]).text = call["detail"]
        if errs:
            case = ET.SubElement(ts, "testcase", classname=cls, name=name)
    elif call and call["outcome"] == "skipped" and not errs:
        ET.SubElement(case, "skipped", message="skip")
    for e in errs:
        ET.SubElement(case, "error", message=e["detail"]).text = e["detail"]
ET.ElementTree(root).write(att / "junit.xml", encoding="unicode")
if mode in ("hang", "hang_failure", "hang_after_result"):
    time.sleep(float(os.environ.get("A4_SLEEP", "3")))
if mode == "unattributable":
    rc = 124
logged = rc + 1 if mode == "log_mismatch" else rc
fields = [datetime.datetime.now(datetime.timezone.utc).isoformat(), str(TOOL), label, "halted", "-", "0", "0", "no-lock" if mode == "unattributable" else "ran",
          str(logged)]
if os.environ.get("REV2_NO_LOG") != "1":  # under the REAL forge-gpu-run, the runner writes its own row
    with open(os.environ["FORGE_GPU_RUN_LOG"], "a") as f:
        f.write("\t".join(fields) + "\n")
sys.exit(rc)
