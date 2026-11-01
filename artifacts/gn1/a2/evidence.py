"""Shared, read-only evidence recomputation for the A2 parent and scorer.

Prior art: SHA-256 (NIST 2001); pytest/JUnit (Krekel et al., 2004+);
DeMillo/Lipton/Sayward (1978), valid mutation kills; exact paired regression
comparison from GN1 D6. A2 adds these order-specific binding conjunctions.
Literature unverified — lead to check named sources.
"""
import datetime
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
ORDER = "orders/GN1_A2_REV1_FIXES.md"
DEVICE_PATTERNS = (
    "cudaMalloc", "no CUDA-capable device", "CUDA driver", "cudaError",
    "initialization error", "invalid device ordinal", "CUDA error at",
    "cudaMallocAsync failed", "out of memory", "illegal memory access",
    "unspecified launch failure", "launch timed out",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def guard(runs):
    runs = Path(runs).resolve()
    if not runs.is_relative_to(TOOL) or runs == TOOL or runs.is_relative_to(TOOL / "arms"):
        raise ValueError("RUNS must be under a2 and outside arms")
    return runs


def lines():
    return {r["label"]: r for r in read(TOOL / "gpu_lines.json")}


def verify_seal():
    seal = read(TOOL / "staging_seal.json")
    assert sha(ROOT / ORDER) == seal["order_sha256"], "order seal mismatch"
    # order_commit is the immutable commit carrying this amendment. Product
    # sources are deterministically transformed by regenerate_arms.py (D9).
    assert re.fullmatch(r"[0-9a-f]{40}", seal["order_commit"]), "missing order commit"
    for name, digest in seal["tool_files"].items():
        assert sha(TOOL / name) == digest, "tool seal mismatch: " + name
    for name, digest in seal["test_files"].items():
        assert sha(ROOT / name) == digest, "test seal mismatch: " + name
    pin = read(TOOL / "engine_pin.json")
    assert sha(ROOT / pin["module"]) == pin["sha256"], "pinned engine mismatch"
    assert sha(TOOL.parent / "engine_sources_after.json") == pin["source_map_sha256"], "source map mismatch"
    assert pin["sha256"] == seal["engine_sha256"], "engine pin/seal mismatch"
    return seal


def verify_arm(arm, seal):
    package = TOOL / "arms" / arm / "tensor_cuda"
    assert arm in seal["arms"], "unregistered arm"
    assert not any(p.is_symlink() for p in (TOOL / "arms" / arm).rglob("*")), "arm contains symlink"
    pin = read(TOOL / "engine_pin.json")
    assert sha(package / Path(pin["module"]).name) == pin["sha256"], "engine copy differs from pin"
    files = {str(p.relative_to(package)): sha(p) for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert files == seal["arms"][arm], "arm file/seal mismatch: " + arm
    assert files[Path(pin["module"]).name] == pin["sha256"], "engine copy differs from pin"
    return package


def verify_start(runs):
    runs = guard(runs)
    seal = verify_seal()
    start = read(runs / "START.json")
    assert start["run_root"] == str(runs), "START run-root binding mismatch"
    assert start["seal_sha256"] == sha(TOOL / "staging_seal.json"), "START seal mismatch"
    assert start["order_sha256"] == seal["order_sha256"] and start["order_commit"] == seal["order_commit"], "START order mismatch"
    assert isinstance(start["audit"], bool), "START audit missing"
    datetime.datetime.fromisoformat(start["date"])
    return seal, start


def census(arm, suite, audit=False):
    name = "audit" if audit else "mutation" if suite == "mutation" else arm + "_" + suite
    return (TOOL / ("ids_" + name + ".txt")).read_text().splitlines()


def summary_outcomes(data):
    reports = data["reports"]
    seen = set()
    result = {}
    for r in reports:
        assert r["when"] in ("setup", "call", "teardown") and r["outcome"] in ("passed", "failed", "skipped"), "invalid phase/outcome"
        key = (r["id"], r["when"])
        assert key not in seen, "duplicate summary phase"
        seen.add(key)
        assert r["id"] in data["collected_ids"], "uncollected report ID"
    for id_ in data["collected_ids"]:
        rs = [r for r in reports if r["id"] == id_]
        errors = [r for r in rs if r["when"] != "call" and r["outcome"] == "failed"]
        calls = [r for r in rs if r["when"] == "call"]
        if errors:
            result[id_] = "error"
        elif calls:
            result[id_] = calls[0]["outcome"]
        elif any(r["outcome"] == "skipped" for r in rs):
            result[id_] = "skipped"
        else:
            result[id_] = "missing"
    return result


def xml_key(id_):
    path, *names = id_.split("::")
    return (".".join([path.removesuffix(".py").replace("/", "."), *names[:-1]]), names[-1])


def junit_outcomes(path, ids, dual_ids=()):
    mapping = {xml_key(i): i for i in ids}
    assert len(mapping) == len(ids), "nonunique JUnit ID mapping"
    outcomes = {}
    parts = {}
    details = []
    for case in ET.parse(path).getroot().iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        assert key in mapping, "unregistered JUnit ID: " + str(key)
        id_ = mapping[key]
        tags = {e.tag for e in case if e.tag in ("failure", "error", "skipped")}
        assert len(tags) <= 1, "multiple JUnit outcomes"
        outcome = "error" if "error" in tags else "failed" if "failure" in tags else "skipped" if "skipped" in tags else "passed"
        if id_ in outcomes:
            # pytest 9 emits two entries for a call failure plus teardown
            # error (native receipt junit_real_shape.json). Permit precisely
            # that pair when the independently read phase reports confirm it.
            assert id_ in dual_ids and len(parts[id_]) == 1 and {parts[id_][0], outcome} == {"failed", "error"}, "duplicate JUnit ID: " + id_
            outcomes[id_] = "error"
            parts[id_].append(outcome)
        else:
            outcomes[id_] = outcome
            parts[id_] = [outcome]
        details.extend(" ".join(e.itertext()) + " " + e.get("message", "") for e in case if e.tag in ("error", "failure"))
    return outcomes, details


def forge_outcome(path, rc):
    # forge-gpu-run own log is tab-separated; installed script supervise()
    # records timeout/window-closed separately even though both return 124.
    if not path.exists():
        return "missing"
    rows = path.read_text().splitlines()
    if not rows:
        return "missing"
    fields = rows[-1].split("\t")
    if len(fields) != 9 or fields[7] not in ("ran", "timeout", "window-closed", "no-window", "no-lock"):
        return "unrecognized"
    try:
        logged_exit = int(fields[8])
    except ValueError:
        return "unrecognized"
    if logged_exit != rc:
        return "log-exit-mismatch"
    return fields[7]



def acceptable_killers(data):
    # A refusal must originate in the GroupNorm frame as captured by pytest,
    # not simply have a TypeError spelling in its message.
    return sorted(r["id"] for r in data["reports"]
                  if r["when"] == "call" and r["outcome"] == "failed"
                  and r["id"].startswith("tensor_cuda/tests/test_groupnorm_gn1.py::")
                  and not r["id"].split("::")[-1].startswith("test_reference_")
                  and (r.get("exception_class") == "AssertionError" or
                       r.get("exception_class") in ("ValueError", "TypeError") and r.get("groupnorm_refusal") is True))


def evaluate(runs, attempt, line, rc, audit, seal):
    reasons = []
    data = None
    killers = []
    outcomes = {}
    if rc not in line["expected_child_exits"]:
        reasons.append("expected exits %s, got %s" % (line["expected_child_exits"], rc))
    try:
        data = read(attempt / "child_result.json")
        assert type(data["pytest_exit"]) is int, "child exit is not an integer"
        assert data["pytest_exit"] == rc, "child exit/summary mismatch"
        assert data["arm"] == line["arm"] and data["suite"] == line["suite"], "child arm/suite binding mismatch"
        assert data["audit"] == audit and data["run_root"] == str(runs) and data["attempt_dir"] == str(attempt), "child run-root/audit binding mismatch"
        package = verify_arm(line["arm"], seal)
        pin = read(TOOL / "engine_pin.json")
        assert data["nn_sha256"] == seal["arms"][line["arm"]]["nn.py"] == sha(package / "nn.py"), "nn.py binding mismatch"
        assert data["engine_sha256"] == seal["engine_sha256"] == pin["sha256"], "engine binding mismatch"
        assert data["seal_sha256"] == sha(TOOL / "staging_seal.json"), "child seal mismatch"
        assert not data["collection_errors"], "collection errors"
        ids = data["collected_ids"]
        assert len(ids) == len(set(ids)) and sorted(ids) == sorted(census(line["arm"], line["suite"], audit)), "test-ID census mismatch"
        outcomes = summary_outcomes(data)
        dual_ids = {id_ for id_ in ids
                    if any(r["id"] == id_ and r["when"] == "call" and r["outcome"] == "failed" for r in data["reports"])
                    and any(r["id"] == id_ and r["when"] == "teardown" and r["outcome"] == "failed" for r in data["reports"])}
        xml, xml_details = junit_outcomes(attempt / "junit.xml", ids, dual_ids)
        assert xml == outcomes, "JUnit/summary outcomes disagree ID by ID"
        failed_details = [r["detail"] for r in data["reports"] if r["outcome"] == "failed"] + xml_details
        if any(p.casefold() in t.casefold() for t in failed_details for p in DEVICE_PATTERNS):
            reasons.append("device-pattern failure: void line")
        if line["suite"] != "regression" and any(r["when"] != "call" and r["outcome"] == "failed" for r in data["reports"]):
            reasons.append("setup/teardown failures")
        if any(o == "missing" for o in outcomes.values()):
            reasons.append("missing test calls")
        if line["suite"] != "regression" and any(o == "skipped" for o in outcomes.values()):
            reasons.append("unregistered skip")
        failures = {r["id"]: r for r in data["reports"] if r["when"] == "call" and r["outcome"] == "failed"}
        if audit or line["arm"] == "candidate" and line["suite"] == "focus":
            if any(o != "passed" for o in outcomes.values()):
                reasons.append("candidate/reference IDs did not all pass")
        elif line["suite"] == "mutation":
            if line.get("equivalence"):
                assert line["equivalence"] == seal["equivalences"].get(line["label"]), "unsealed equivalence proof"
                if any(o != "passed" for o in outcomes.values()):
                    reasons.append("equivalent mutant did not pass")
            else:
                killers = acceptable_killers(data)
                if not killers:
                    reasons.append("no real non-reference GroupNorm assertion/refusal kill")
        elif line["arm"] == "baseline" and line["suite"] == "focus":
            registered = set(read(TOOL / "baseline_expected_failures.json"))
            if set(failures) != registered:
                reasons.append("baseline failures differ from registration")
            if any("reshape: element count mismatch" not in r["detail"]
                   for id_, r in failures.items() if "refusal" not in id_.split("::")[-1]):
                reasons.append("baseline numerical failure is not registered reshape error")
        any_failure = any(r["outcome"] == "failed" for r in data["reports"])
        if rc == 0 and any_failure or rc == 1 and not any_failure:
            reasons.append("pytest exit inconsistent with failures")
    except (AssertionError, KeyError, ValueError, TypeError, OSError, ET.ParseError) as exc:
        reasons.append(str(exc) or type(exc).__name__)
    cause = forge_outcome(attempt / "forge.log", rc)
    if cause != "ran":
        reasons.append("forge outcome: " + cause)
    return dict(passed=not reasons, reasons=reasons, actual_killers=killers if not reasons else [],
                outcomes=outcomes, forge_outcome=cause, timeout=cause == "timeout")


def attempts(runs, label):
    base = runs / label
    if not base.exists():
        return []
    dirs = sorted(base.glob("attempt-*"))
    assert [p.name for p in dirs] == [f"attempt-{i:04d}" for i in range(1, len(dirs)+1)], "nonconsecutive attempts"
    return dirs


def regression_reasons(old, new):
    before, after = summary_outcomes(old), summary_outcomes(new)
    errors = []
    lost = sorted(i for i, o in before.items() if o == "passed" and after.get(i) != "passed")
    if lost:
        errors.append("baseline-passing IDs no longer pass: " + repr(lost))
    def fails(data):
        result = {}
        for r in data["reports"]:
            if r["outcome"] == "failed":
                result.setdefault(r["id"], set()).add(r.get("exception_class"))
        return result
    old_fail, new_fail = fails(old), fails(new)
    fresh = sorted(i for i, classes in new_fail.items()
                   if None in classes or not classes.issubset(old_fail.get(i, set())))
    if fresh:
        errors.append("candidate failure absent/different exception class on baseline: " + repr(fresh))
    return errors


def recompute_line(runs, label, seal, start, stack=()):
    assert label not in stack, "dependency cycle"
    line = lines()[label]
    inventory = []
    selected = None
    result = dict(passed=False, reasons=["no admitted attempt"], actual_killers=[], outcomes={})
    for attempt in attempts(runs, label):
        receipt = read(attempt / "parent_result.json")
        assert receipt["label"] == label and receipt["attempt_dir"] == str(attempt), "parent attempt binding mismatch"
        assert receipt["expected_child_exits"] == line["expected_child_exits"], "parent expected exits differ from gpu_lines.json"
        assert receipt["arm"] == line["arm"] and receipt["suite"] == line["suite"] and receipt["audit"] == start["audit"], "parent line binding mismatch"
        assert receipt["command"][-3:] == ["run_gpu_tests.py", line["arm"], line["suite"]], "parent command arm/suite mismatch"
        assert receipt["cwd"] == str(TOOL), "parent cwd mismatch"
        assert receipt["run_root"] == str(runs) and receipt["seal_sha256"] == sha(TOOL / "staging_seal.json"), "parent run-root/seal mismatch"
        assert isinstance(receipt["exit"], int) and not isinstance(receipt["exit"], bool), "exit is not an integer"
        assert datetime.datetime.fromisoformat(start["date"]) <= datetime.datetime.fromisoformat(receipt["start"]) <= datetime.datetime.fromisoformat(receipt["child_exited"]) <= datetime.datetime.fromisoformat(receipt["recorded"]), "parent completion order invalid"
        cause = forge_outcome(attempt / "forge.log", receipt["exit"])
        assert receipt["forge_outcome"] == cause, "stored forge cause differs from log"
        not_admitted = receipt["exit"] in (75, 76) and not (attempt / "child_result.json").exists() and cause in ("no-window", "no-lock")
        assert receipt["admission"] == ("not_admitted" if not_admitted else "admitted"), "admission mismatch"
        inventory.append(dict(attempt=attempt.name, admission=receipt["admission"], exit=receipt["exit"], forge_outcome=cause))
        if not_admitted:
            assert not receipt["passed"] and not receipt["actual_killers"], "refused attempt claims pass/kill"
            assert receipt["reasons"] == ["not admitted: " + cause] and receipt["outcomes"] == {} and receipt["timeout"] is False, "stored refused receipt disagrees with recomputation"
            continue
        selected = attempt
        result = evaluate(runs, attempt, line, receipt["exit"], start["audit"], seal)
        if not start["audit"] and line["suite"] == "mutation":
            dependency = recompute_line(runs, "candidate_focus", seal, start, (*stack, label))
            if not dependency["passed"]:
                result["reasons"].append("mutations require recomputed passing candidate focus")
        if not start["audit"] and label == "candidate_regression":
            baseline = recompute_line(runs, "baseline_regression", seal, start, (*stack, label))
            if not baseline["passed"]:
                result["reasons"].append("missing valid completed baseline regression")
            elif (attempt / "child_result.json").exists():
                result["reasons"] += regression_reasons(read(Path(baseline["selected_attempt"]) / "child_result.json"), read(attempt / "child_result.json"))
        result["passed"] = not result["reasons"]
        if not result["passed"]:
            result["actual_killers"] = []
        for key in ("passed", "reasons", "actual_killers", "forge_outcome", "timeout", "outcomes"):
            assert receipt[key] == result[key], "stored parent receipt disagrees with recomputation: " + key
    result.update(selected_attempt=str(selected) if selected else None, attempts=inventory)
    return result
