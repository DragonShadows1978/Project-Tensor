"""A7 D2/D3 hostile receipt controls, no device execution.

Prior art: A6 D0_MODE_PROMOTION (2026), Meszaros test stubs/fault injection
(2007), SHA-256 FIPS180-4 (2015). Taken: actual parent/installed-runner CPU
receipts and their reversible, recorded mutations. Ours: the registered
finite per-line promotion census, device-field controls, and the D0 residual
probe. Constructed GPU-shaped data is explicitly never a device run.
"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from evidence import TOOL, ROOT, lines, read, sha
from device_evidence import CALLS, identity
from program_binding import PYTHON

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
BASE = TOOL / "a6_runs" / "honest_all_labels"
ROWS = []
LABELS = list(lines())
START = BASE / "START.json"
PARENTS = {label: BASE / label / "attempt-0001/parent_result.json" for label in LABELS}
CHILDREN = {label: BASE / label / "attempt-0001/child_result.json" for label in LABELS}
INVOCATIONS = {label: BASE / label / "attempt-0001/invocation.json" for label in LABELS}
ALL = [START, *PARENTS.values(), *CHILDREN.values(), *INVOCATIONS.values()]
ORIGINAL = {p: p.read_bytes() for p in ALL}
SEAL = read(TOOL / "staging_seal.json")
SEAL_HASH = sha(TOOL / "staging_seal.json")
LOGS = {p: sha(p) for p in BASE.rglob("*") if p.is_file() and (p.suffix == ".log" or p.name.startswith("parent_result.") and p.suffix == ".tmp")}


def put(p, d):
    p.write_text(json.dumps(d, indent=2) + "\n")


def restore():
    for p, b in ORIGINAL.items():
        if p.read_bytes() != b:
            p.write_bytes(b)


def promoted(root, labels):
    if root:
        d = read(START); d["control"] = False; put(START, d)
    for label in labels:
        p = PARENTS[label]; d = read(p); d["control"] = False
        for k in ("FORGE_GPU_SUMMARY", "FORGE_GPU_WINDOW_FILE", "FORGE_GPU_LOCK"):
            d["environment"].pop(k, None)
        put(p, d)


def gpu_data(labels):
    """Synthetic receipt edits, never claimed to be observed GPU evidence."""
    for label in labels:
        d = read(CHILDREN[label]); parent = read(PARENTS[label])
        e = dict(program=identity(TOOL / "run_gpu_tests.py"), CUDA_VISIBLE_DEVICES="0",
                 device_name="NVIDIA GeForce RTX 4070 SUPER", cuda_runtime_version="12060",
                 cuda_driver_version="13020", device_ordinal="0", native_status={k:"0" for k in CALLS},
                 engine_sha256=SEAL["engine_sha256"], provider="pinned-engine:libcudart",
                 query_error=None, captured_at=parent["start"])
        d["device_evidence"] = e; d["CUDA_VISIBLE_DEVICES"] = "0"; put(CHILDREN[label], d)
        inv = read(INVOCATIONS[label]); inv["device_evidence"] = e; put(INVOCATIONS[label], inv)


def score(name, target, expected, deciding_rule, edit):
    restore()
    edit()
    changed = {str(p.relative_to(BASE)):dict(before_sha256=hashlib.sha256(b).hexdigest(), after_sha256=sha(p)) for p,b in ORIGINAL.items() if p.read_bytes()!=b}
    argv = [PYTHON, "-B", "score_gpu.py"]
    e = dict(os.environ, RUNS=str(BASE))
    r = subprocess.run(["/usr/bin/timeout", "90s", *argv], cwd=TOOL, env=e, capture_output=True, text=True)
    s = read(BASE / "G5_SUMMARY.json")
    actual = bool(s["passed"])
    row = dict(control=name, target=target, passed=(r.returncode == (0 if expected else 1) and actual == expected),
               expected_acceptance=expected, actual_acceptance=actual, deciding_rule=deciding_rule,
               scorer_exit=r.returncode, stderr=r.stderr[-1000:], edits=changed,
               summary=dict(passed=actual, errors=s["errors"], passed_lines=sum(v["passed"] for v in s["lines"].values()),
                            device_lines=sum(v.get("mode")=="device" for v in s["lines"].values()),
                            reasons={k:v["reasons"] for k,v in s["lines"].items() if v["reasons"]}))
    ROWS.append(row)
    (TOOL / "G4_a7.json").write_text(json.dumps(dict(passed=all(v["passed"] for v in ROWS), controls=ROWS), indent=2)+"\n")
    print(name, "PASS" if row["passed"] else "FAIL", flush=True)
    return row


try:
    score("D0_MODE_PROMOTION", "A6 exact 59-file promotion refused", False, "D2 per-child evidence; D3 parent mode mismatch", lambda: promoted(True, LABELS))
    score("Promotion_root_only", "root declaration alone cannot promote", False, "D2 CPU evidence and D3 parent/root consistency", lambda: promoted(True, []))
    score("Promotion_parents_only", "all parent declarations alone cannot promote", False, "D2 CPU evidence and D3 mode mismatch", lambda: promoted(False, LABELS))
    # A line-local predicate plus the all-lines root conjunction covers the
    # general subset argument. Execute each singleton and its complement,
    # with both root declarations; do not claim to enumerate 2**58 subsets.
    for root in (False, True):
        for label in LABELS:
            for complement in (False, True):
                selected = [l for l in LABELS if (l != label if complement else l == label)]
                score(f"Promotion_r{int(root)}_{'except' if complement else 'only'}_{label}",
                      "partial parent promotion refused", False, "D2/D3 per-line mode binding and all-lines device conjunction",
                      lambda rr=root, ls=selected: promoted(rr, ls))

    def positive():
        promoted(True, LABELS); gpu_data(LABELS)
    positive_row = score("Constructed_GPU_shaped", "constructed GPU-shaped receipt schema accepted; no GPU execution claim", True,
                         "all 58 device conjunctions, sealed programs and exact argv", positive)
    # The positive shape also exposes the unchanged D0(a) authenticity gap:
    # only receipts are edited; the seal, logs, executables and argv stay put.
    score("D0_COORDINATED_CHILD_EVIDENCE_FORGERY", "D0(a) CPU root must refuse forged child evidence without seal/log rewrite", False,
          "D0(a) receipt forgery; no trusted authentication operand exists in D2", positive)

    # Actual real-path file substitution, while the seal stays fixed.
    # Prior art: Meszaros fault injection (2007); A6 sealed file identities.
    real_child = TOOL / "run_gpu_tests.py"
    original_child = real_child.read_bytes()
    try:
        real_child.write_bytes((TOOL / "rev2_forge.py").read_bytes())
        row = read(PARENTS["candidate_focus"])
        parent = subprocess.run(["/usr/bin/timeout", "30s", PYTHON, "-B", "run_gpu_line.py",
                    "candidate_focus", "0", str(row["outer_bound_seconds"]), *row["command"]],
                    cwd=TOOL, env=dict(os.environ,RUNS=str(BASE),GN1_CONTROL="1"),capture_output=True,text=True)
        attack = score("D4_actual_standin_at_real_child_path", "actual stand-in file under sealed real child path refused", False,
                       "tool seal mismatch: run_gpu_tests.py before child launch", lambda: None)
        attack["actual_parent_exit"] = parent.returncode
        attack["actual_parent_stderr"] = parent.stderr[-1500:]
        attack["passed"] = attack["passed"] and parent.returncode == 1 and "tool seal mismatch: run_gpu_tests.py" in parent.stderr
    finally:
        real_child.write_bytes(original_child)

    for label in (LABELS[0], "candidate_focus", LABELS[-1]):
        def false_child(l=label):
            positive()
            d=read(CHILDREN[l]); d["device_evidence"]["program"]=identity(TOOL/"rev2_forge.py");put(CHILDREN[l],d)
            inv=read(INVOCATIONS[l]);inv["device_evidence"]=d["device_evidence"];put(INVOCATIONS[l],inv)
        score("D4_standin_identity_"+label, "claimed device evidence with stand-in identity refused", False,
              "device evidence: child program differs from sealed run_gpu_tests.py", false_child)

    def field_case(name, change):
        def edit():
            positive(); d=read(CHILDREN["candidate_focus"]); change(d)
            put(CHILDREN["candidate_focus"],d)
            inv=read(INVOCATIONS["candidate_focus"]);inv["device_evidence"]=d.get("device_evidence");put(INVOCATIONS["candidate_focus"],inv)
        score(name, "malformed or incomplete device field refused", False, "D2 exact device conjunction; D3 mode mismatch", edit)
    field_case("D4_standin_at_real_path_sha_mismatch", lambda d:d["device_evidence"]["program"].__setitem__("sha256",sha(TOOL/"rev2_forge.py")))
    field_case("D4_empty_CUDA_VISIBLE_DEVICES", lambda d:(d.__setitem__("CUDA_VISIBLE_DEVICES",""),d["device_evidence"].__setitem__("CUDA_VISIBLE_DEVICES","")))
    for field in ("program","CUDA_VISIBLE_DEVICES","device_name","cuda_runtime_version","cuda_driver_version","device_ordinal","native_status","engine_sha256","provider","query_error","captured_at"):
        field_case("Device_field_missing_"+field,lambda d,f=field:d["device_evidence"].pop(f))
    for field in ("cuda_runtime_version","cuda_driver_version","device_ordinal"):
        field_case("Device_field_type_"+field,lambda d,f=field:d["device_evidence"].__setitem__(f,1))
    for call in CALLS:
        field_case("Device_call_failure_"+call,lambda d,c=call:d["device_evidence"]["native_status"].__setitem__(c,"100"))

    # Exhaustive 2x2x2 combinations at each operand class: root control,
    # all parent controls, all child-evidence edits. Other receipt bytes fixed.
    for root in (False,True):
        for parents in (False,True):
            for children in (False,True):
                def edit(r=root,p=parents,c=children):
                    promoted(r,LABELS if p else [])
                    if c:gpu_data(LABELS)
                score(f"Combination_r{int(root)}_p{int(parents)}_c{int(children)}",
                      "only consistent full constructed device schema can accept", root and parents and children,
                      "D2 per-child classification and D3 modes only demote",edit)

    # A device-shaped root may be demoted by either root flag or any parent
    # declaration. Child audit remains an independently checked mode operand.
    for name,which in [("Root_control_demotes","root_control"),("Root_audit_demotes","root_audit"),
                       ("Parent_control_demotes","parent_control"),("Parent_audit_demotes","parent_audit"),
                       ("Child_audit_disagrees","child_audit")]:
        def edit(w=which):
            positive()
            target=START if w.startswith("root") else PARENTS["candidate_focus"] if w.startswith("parent") else CHILDREN["candidate_focus"]
            d=read(target);d["control" if w.endswith("control") else "audit"]=True;put(target,d)
        score(name,"mode declaration can only demote",False,"D3 root demotion or per-line mode disagreement",edit)
finally:
    restore()
    subprocess.run(["/usr/bin/timeout","90s",PYTHON,"-B","score_gpu.py"],cwd=TOOL,
                   env=dict(os.environ,RUNS=str(BASE)),stdout=subprocess.DEVNULL,check=False)
    unchanged = SEAL_HASH == sha(TOOL/"staging_seal.json") and all(sha(p)==h for p,h in LOGS.items())
    assert unchanged
    assert all(p.read_bytes()==b for p,b in ORIGINAL.items())
    (TOOL/"D0_EVIDENCE_RESTORED.json").write_text(json.dumps(dict(passed=True,seal_and_logs_unchanged=True,receipt_bytes_restored=len(ORIGINAL),
        command_argv_and_program_edits=0,real_device_execution=False),indent=2)+"\n")

expected=[r for r in ROWS if r["control"]!="D0_COORDINATED_CHILD_EVIDENCE_FORGERY"]
assert all(r["passed"] for r in expected),[r["control"] for r in expected if not r["passed"]]
# A supplemental in-scope RED is a result, not a driver crash; the final gate
# consolidator reports its failure and withholds acceptance.
print("registered D4 targets met; D0 supplemental", "HELD" if all(r["passed"] for r in ROWS) else "FAILED")
