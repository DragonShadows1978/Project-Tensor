#!/usr/bin/env python3
"""CPU hostile forge stand-in, adapted from REV1 hostile_scorer.py.
Prior art: Meszaros (2007) xUnit test stub/fault injection, REV1 cases H1–H6.
A2 emits full bound summaries and matching JUnit so one targeted defect at a
time reaches the verifier. No GPU invocation or leasing is possible here.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from evidence import TOOL, read, lines, census, sha, xml_key
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
mode = os.environ.get("HOSTILE_MODE", "normal")
label = os.environ["GN1_LINE"]
line = lines()[label]
output = Path(os.environ["GN1_ATTEMPT"])


def log(rc, cause="ran"):
    fields=[datetime.datetime.now(datetime.timezone.utc).isoformat(), str(TOOL), label, "halted", "-", "0", "0", cause, str(rc)]
    with Path(os.environ["FORGE_GPU_RUN_LOG"]).open("a") as f:
        f.write("\t".join(fields)+"\n")


if mode in ("refuse75", "refuse76"):
    rc = int(mode[-2:])
    log(rc, "no-window" if rc==75 else "no-lock")
    sys.exit(rc)
arm, suite = line["arm"], line["suite"]
ids = census(arm,suite)
reports = [dict(id=i,when="call",outcome="passed",detail="",exception_class=None,groupnorm_refusal=False) for i in ids]
nonref = next(i for i in ids if "test_groupnorm_gn1.py::test_forward_values[" in i)
reference = next(i for i in ids if "::test_reference_" in i)


def fail(id_, detail="AssertionError: injected finite control", cls="AssertionError", refusal=False):
    r=next(r for r in reports if r["id"]==id_)
    r.update(outcome="failed",detail=detail,exception_class=cls,groupnorm_refusal=refusal)


if suite=="mutation" and not line.get("equivalence"):
    fail(nonref)
if label=="baseline_focus":
    for i in read(TOOL/"baseline_expected_failures.json"):
        cls = read(TOOL/"baseline_new_failures.json").get(i, {}).get("exception_class", "AssertionError")
        fail(i,"GroupNorm baseline registered D5 failure" if i in read(TOOL/"baseline_new_failures.json") else "AssertionError: GroupNorm refused a registered valid input: reshape: element count mismatch",cls)
if suite=="regression":
    # Faithful semantics of an unrelated legacy failure shared by both arms.
    legacy=next(i for i in ids if "test_groupnorm_gn1.py" not in i)
    fail(legacy,"RuntimeError: registered synthetic legacy failure", "RuntimeError")
    if mode=="regression_new_failure":fail(nonref)
    if mode=="regression_changed_class":fail(legacy,"AssertionError: changed class")
    if mode=="regression_missing_pass":reports=[r for r in reports if r["id"]!=nonref]
    if mode=="regression_skip_pass":
        next(r for r in reports if r["id"]==nonref)["outcome"]="skipped"
    if mode in ("regression_legacy_error", "regression_error_changed_class"):
        next(r for r in reports if r["id"]==legacy)["when"]="setup"
        if mode=="regression_error_changed_class":
            fail(legacy,"TypeError: changed legacy setup class","TypeError")
    if mode=="regression_new_setup_error":
        fail(nonref,"RuntimeError: new setup failure","RuntimeError")
        next(r for r in reports if r["id"]==nonref)["when"]="setup"
    if mode=="regression_dual_failure":
        reports.append(dict(id=legacy,when="teardown",outcome="failed",detail="ValueError: legacy teardown error",exception_class="ValueError",groupnorm_refusal=False))
if mode in ("rev1_device", "rev1_device_control"):
    detail = ("RuntimeError: cudaErrorIllegalAddress" if mode=="rev1_device_control" else
              "RuntimeError: CUDA error at reduce_lastdim: an illegal memory access was encountered")
    for i in ids:fail(i,detail,"RuntimeError")
if mode=="device":
    fail(reference,"RuntimeError: "+os.environ["DEVICE_PATTERN"], "RuntimeError")
if mode=="reference_only":
    for r in reports:r.update(outcome="passed",detail="",exception_class=None,groupnorm_refusal=False)
    fail(reference)
if mode=="generic_runtime":
    fail(nonref,"RuntimeError: unrelated engine refusal", "RuntimeError")
if mode=="unregistered_refusal":
    fail(nonref,"TypeError: outside GroupNorm", "TypeError",False)
if mode=="registered_refusal":
    fail(nonref,"TypeError: GroupNorm registered refusal", "TypeError",True)
if mode=="foreign_arm":
    arm="baseline"
    for r in reports:r.update(outcome="passed",detail="",exception_class=None,groupnorm_refusal=False)
    for i in ids:
        if "values" in i:fail(i,"RuntimeError: reshape: element count mismatch","RuntimeError")
if mode=="suite_swap":
    arm,suite="candidate","focus"
    ids=census(arm,suite)
    reports=[dict(id=i,when="call",outcome="passed",detail="",exception_class=None,groupnorm_refusal=False) for i in ids]
# D2 controls use the real test file and real Reports plugin, not a synthetic
# exception_class fixture. Prior art: REV2 q3_probe.py; Meszaros (2007).
if mode.startswith("real_"):
    import contextlib
    import pytest
    from cpu_shim import install, Tensor
    from run_gpu_tests import Reports
    tc = install(TOOL / "arms" / arm / "tensor_cuda/nn.py")
    target = "test_constructor_defaults" if mode == "real_callee" else "test_forward_values[float32-r3-n2-g2-default]"
    target_id = "tensor_cuda/tests/test_groupnorm_gn1.py::" + target
    baseline_all = mode == "real_baseline"
    if mode in ("real_generic", "real_device", "real_candidate_fail"):
        message = ("CUDA error at pow_kernel: an illegal memory access was encountered"
                   if mode == "real_device" else "pow: unsupported exponent path")
        def bad_pow(self, power):
            raise RuntimeError(message)
        Tensor.pow = bad_pow
    if mode == "real_callee":
        real_factory = tc.tensor
        def bad_factory(data, device="cuda", dtype="float32", requires_grad=False):
            if requires_grad:
                raise TypeError("tensor(): incompatible function arguments (engine factory)")
            return real_factory(data, device, dtype, requires_grad)
        tc.tensor = bad_factory
    plugin = Reports()
    with (output / "real_pytest.log").open("x") as logfile, contextlib.redirect_stdout(logfile), contextlib.redirect_stderr(logfile):
        real_rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib",
                                 "--rootdir="+str(TOOL.parents[2]), str(TOOL.parents[2] / "tensor_cuda/tests/test_groupnorm_gn1.py")+("" if baseline_all else "::"+target)], plugins=[plugin]))
    assert (set(plugin.ids)=={i for i in ids if "test_groupnorm_gn1.py::" in i} if baseline_all else plugin.ids == [target_id]) and not plugin.collection_errors
    # Other IDs are unchanged all-pass binding scaffolding. Only the target
    # ID supplies the kill/refusal; its phases are actual pytest reports.
    reports = ([r for r in reports if r["id"] not in plugin.ids] if baseline_all else
               [dict(id=i,when="call",outcome="passed",detail="",exception_class=None,groupnorm_refusal=False)
                for i in ids if i not in plugin.ids]) + plugin.reports
    (output / "real_test_evidence.json").write_text(json.dumps(dict(
        target=target_id, pytest_exit=real_rc, reports=plugin.reports, source="real test file / real Reports / CPU stand-in"), indent=2)+"\n")
rc = int(any(r["outcome"]=="failed" for r in reports))
if mode=="suite_swap":rc=0
pin=read(TOOL/"engine_pin.json")
seal=read(TOOL/"staging_seal.json")
data=dict(arm=arm,suite=suite,audit=False,run_root=os.environ["RUNS"],attempt_dir=str(output),
          nn_sha256=seal["arms"][arm]["nn.py"],engine_sha256=pin["sha256"],
          seal_sha256=sha(TOOL/"staging_seal.json"),pytest_exit=rc,collected_ids=ids,
          collection_errors=[],reports=reports,CUDA_VISIBLE_DEVICES="")
if mode=="bad_nn_hash":data["nn_sha256"]="0"*64
if mode=="bad_engine_hash":data["engine_sha256"]="0"*64
if mode=="collection_error":data["collection_errors"]=["hostile collection error"]
if mode=="setup_error":
    reports[0].update(when="setup",outcome="failed",detail="RuntimeError: setup failed",exception_class="RuntimeError")
    rc=data["pytest_exit"]=1
(output/"child_result.json").write_text(json.dumps(data,indent=2)+"\n")
root=ET.Element("testsuites");ts=ET.SubElement(root,"testsuite",name="pytest")
for r in reports:
    if r["when"] != "call" and r["outcome"] == "passed":
        continue
    cls,name=xml_key(r["id"])
    c=ET.SubElement(ts,"testcase",classname=cls,name=name)
    o=r["outcome"]
    if mode=="junit_disagree" and r["id"]==nonref:o="failed" if o=="passed" else "passed"
    if o!="passed":
        ET.SubElement(c,"error" if r["when"]!="call" and o=="failed" else "failure" if o=="failed" else "skipped",message=r["detail"]).text=r["detail"]
if mode=="junit_foreign_id":
    ET.SubElement(ET.SubElement(ts,"testcase",name="x"),"failure",message="real failure")
if mode=="junit_duplicate_pass":
    cls,name=xml_key(nonref)
    ET.SubElement(ts,"testcase",classname=cls,name=name)
ET.ElementTree(root).write(output/"junit.xml",encoding="unicode")
if mode=="hang_after_result":
    # Parent timeout terminates only this foreground child tree. No polling.
    time.sleep(30)
if mode in ("timeout", "window-closed"):
    log(124,mode)
    sys.exit(124)
log(rc)
sys.exit(rc)
