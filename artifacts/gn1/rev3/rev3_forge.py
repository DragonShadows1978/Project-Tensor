#!/usr/bin/env python3
"""GN1-REV3 CPU stand-in child (and, when REV2_NO_LOG is unset, forge stand-in).
Derived from REV2's rev2_forge.py (2026); A3's adaptation idea for the baseline
D5 classes (a3/rev2_forge.py) is reused. Never leases, never runs the GPU,
never imports the engine. New modes: hang_after_result (child_result then
sleep), started_no_result (invocation.json as the real child writes it before
pytest, then sleep, never a result).
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
TOOL = Path(os.environ["REV3_MA3"])
sys.path.insert(0, str(TOOL))
from evidence import census, xml_key, lines  # noqa: E402

mode = os.environ.get("REV2_MODE", "normal")
label = os.environ["GN1_LINE"]
line = lines()[label]
att = Path(os.environ["GN1_ATTEMPT"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = json.loads((TOOL / "staging_seal.json").read_text())
pin = json.loads((TOOL / "engine_pin.json").read_text())
arm, suite = line["arm"], line["suite"]
if mode == "started_no_result":
    (att / "invocation.json").write_text(json.dumps(dict(arm=arm, suite=suite, note="child started")) + "\n")
    time.sleep(float(os.environ.get("REV3_SLEEP", "6")))
    sys.exit(0)
ids = census(arm, suite)
reports = {i: dict(id=i, when="call", outcome="passed", detail="", exception_class=None, groupnorm_refusal=False)
           for i in ids}


def fail(i, detail="AssertionError: forward values", cls="AssertionError"):
    reports[i].update(outcome="failed", detail=detail, exception_class=cls)


gn = [i for i in ids if "test_groupnorm_gn1.py::test_forward_values[" in i]
legacy = [i for i in ids if "test_groupnorm_gn1.py" not in i]
new_expected = json.loads((TOOL / "baseline_new_failures.json").read_text())
if suite == "mutation" and not line.get("equivalence"):
    fail(gn[0])
if label == "baseline_focus":
    for i in json.loads((TOOL / "baseline_expected_failures.json").read_text()):
        if i in new_expected:
            fail(i, "GroupNorm baseline registered D5 failure", new_expected[i]["exception_class"])
        else:
            fail(i, "AssertionError: GroupNorm refused a registered valid input: reshape: element count mismatch")
if suite == "regression":
    fail(legacy[0], "RuntimeError: shared legacy failure", "RuntimeError")
if mode == "v3_failure":
    fail(gn[0])
if mode == "skip_one":
    reports[gn[1]].update(outcome="skipped")
if mode == "baseline_extra":
    fail(next(i for i in ids if i not in json.loads((TOOL / "baseline_expected_failures.json").read_text())),
         "AssertionError: GroupNorm refused a registered valid input: reshape: element count mismatch")
if mode in ("rev1_h2", "rev1_h2c"):
    text = ("RuntimeError: CUDA error at reduce_lastdim: an illegal memory access was encountered" if mode == "rev1_h2"
            else "RuntimeError: cudaErrorIllegalAddress")
    for i in ids:
        fail(i, text, "RuntimeError")
if mode == "rev1_h3":
    arm = "baseline"
    for i in ids:
        reports[i].update(outcome="passed", detail="", exception_class=None)
        if "values" in i:
            fail(i, "RuntimeError: reshape: element count mismatch", "RuntimeError")
if mode == "rev1_h3b":
    arm, suite = "candidate", "focus"
    ids = census(arm, suite)
    reports = {i: dict(id=i, when="call", outcome="passed", detail="", exception_class=None, groupnorm_refusal=False)
               for i in ids}
reports_extra = None
if mode == "kill_plus_teardown_device":
    reports_extra = dict(id=gn[1], when="teardown", outcome="failed",
                         detail="RuntimeError: CUDA error at sync: unspecified launch failure",
                         exception_class="RuntimeError", groupnorm_refusal=False)
if mode == "kill_device_word_in_assert":
    fail(gn[0], "AssertionError: forward values\nassert False\n  where note = 'checked: no out of memory here'")
collected = list(ids)
if mode == "census_drop":
    collected = collected[1:]
    reports.pop(ids[0])
rows = list(reports.values())
if reports_extra:
    rows.append(reports_extra)
rc = int(any(r["outcome"] == "failed" for r in rows))
if mode == "exit0_with_failure":
    rc = 0
data = dict(arm=arm, suite=suite, audit=False, cwd=str(TOOL), run_root=os.environ["RUNS"], attempt_dir=str(att),
            seal_sha256=sha(TOOL / "staging_seal.json"), engine_sha256=pin["sha256"],
            nn_sha256=seal["arms"][arm]["nn.py"], CUDA_VISIBLE_DEVICES="", files=[], pytest_exit=rc,
            collected_ids=collected, collection_errors=[], reports=rows)
(att / "invocation.json").write_text(json.dumps(dict(arm=arm, suite=suite)) + "\n")
(att / "child_result.json").write_text(json.dumps(data, indent=2) + "\n")
root = ET.Element("testsuites")
ts = ET.SubElement(root, "testsuite", name="pytest")
for i in collected:
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
if mode in ("hang", "hang_after_result"):
    time.sleep(float(os.environ.get("REV3_SLEEP", "30")))
logged = rc + 1 if mode == "log_mismatch" else rc
fields = [datetime.datetime.now(datetime.timezone.utc).isoformat(), str(TOOL), label, "halted", "-", "0", "0", "ran",
          str(logged)]
if os.environ.get("REV2_NO_LOG") != "1":
    with open(os.environ["FORGE_GPU_RUN_LOG"], "a") as f:
        f.write("\t".join(fields) + "\n")
sys.exit(rc)
