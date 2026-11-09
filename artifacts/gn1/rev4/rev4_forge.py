#!/usr/bin/env python3
"""GN1-REV4 CPU stand-in child (and, unless REV2_NO_LOG=1, runner stand-in).
Derived from REV3's rev3_forge.py (2026), extended with A4's child fields
(atomic child_started.json first, runtime_modules census, equivalence_runtime).
Never leases, never runs the GPU, never imports the engine.
Modes: normal, hang, hang_after_result, started_no_result, marker_only_sleep,
ignore_term, nolock_row_hang, tamper_runtime, unsealed_runtime.
Prior art: test stubs (Meszaros 2007). Ours: the modes.
"""
import datetime
import hashlib
import json
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
TOOL = Path(os.environ["REV4_MA4"])
ROOT = TOOL.parents[2]
sys.path.insert(0, str(TOOL))
from runtime_evidence import atomic_json, module_census, npbool_index_check  # noqa: E402
mode = os.environ.get("REV2_MODE", "normal")
label = os.environ["GN1_LINE"]
att = Path(os.environ["GN1_ATTEMPT"])


def row(outcome, rc):
    fields = [datetime.datetime.now(datetime.timezone.utc).isoformat(), str(TOOL), label, "halted", "-", "0", "0",
              outcome, str(rc)]
    with open(os.environ["FORGE_GPU_RUN_LOG"], "a") as f:
        f.write("\t".join(fields) + "\n")


if mode == "nolock_row_hang":
    # Runner stand-in: logs a no-lock row (as the real runner's finally does),
    # then is still alive when the outer bound fires (the race window).
    row("no-lock", 76)
    time.sleep(60)
    sys.exit(76)
if mode != "no_marker":
    atomic_json(att / "child_started.json", dict(pid=os.getpid(), label=label))
if mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (att / "orphan_pid.txt").write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("REV3_SLEEP", "40")))
    sys.exit(0)
if mode == "marker_only_sleep":
    time.sleep(float(os.environ.get("REV3_SLEEP", "30")))
    sys.exit(0)
from evidence import census, xml_key, lines  # noqa: E402
line = lines()[label]
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
rows = list(reports.values())
rc = int(any(r["outcome"] == "failed" for r in rows))
modules = module_census(ROOT)
if mode == "tamper_runtime":
    modules.append(dict(module="numpy.random.mtrand", path=next(p for p in seal["external_modules"] if "numpy/random/mtrand" in p),
                        sha256="0" * 64))
if mode == "unsealed_runtime":
    modules.append(dict(module="planted", path="artifacts/gn1/a4/planted_rev4.py", sha256="1" * 64))
eq = npbool_index_check() if line.get("equivalence") and label != "V3" else None
if mode == "npbool_accepts":
    # Premise-false operand: what the record must look like if operator.index accepted numpy.bool_.
    eq = dict(numpy_version=eq["numpy_version"], value=1, accepts_numpy_bool=True)
data = dict(arm=arm, suite=suite, audit=False, cwd=str(TOOL), run_root=os.environ["RUNS"], attempt_dir=str(att),
            seal_sha256=sha(TOOL / "staging_seal.json"), engine_sha256=pin["sha256"],
            nn_sha256=seal["arms"][arm]["nn.py"], CUDA_VISIBLE_DEVICES="", files=[], pytest_exit=rc,
            collected_ids=list(ids), collection_errors=[], reports=rows, equivalence_runtime=eq,
            runtime_modules=modules)
(att / "invocation.json").write_text(json.dumps(dict(arm=arm, suite=suite)) + "\n")
root = ET.Element("testsuites")
ts = ET.SubElement(root, "testsuite", name="pytest")
for i in ids:
    r = reports[i]
    cls, name = xml_key(i)
    case = ET.SubElement(ts, "testcase", classname=cls, name=name)
    if r["outcome"] == "failed":
        ET.SubElement(case, "failure", message=r["detail"]).text = r["detail"]
ET.ElementTree(root).write(att / "junit.xml", encoding="unicode")
if mode == "truncated_result":
    text = json.dumps(data, indent=2)
    (att / "child_result.json").write_text(text[:len(text) // 3])
else:
    atomic_json(att / "child_result.json", data)
if mode in ("hang", "hang_after_result"):
    time.sleep(float(os.environ.get("REV3_SLEEP", "30")))
if os.environ.get("REV2_NO_LOG") != "1":
    row("ran", rc)
sys.exit(rc)
