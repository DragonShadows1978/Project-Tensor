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
from runtime_evidence import runtime_modules
from device_evidence import classify
from program_binding import reasons as program_reasons, RUNNER, TIMEOUT, PYTHON

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
ORDER = "orders/GN1_A7_MODE_EVIDENCE.md"
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
        raise ValueError("RUNS must be under a7 and outside arms")
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
    for name, digest in seal["import_modules"].items():
        assert sha(ROOT / name) == digest, "import module seal mismatch: " + name
    for name, digest in seal["external_modules"].items():
        assert sha(Path(name.replace("$HOME/", str(Path.home())+"/", 1))) == digest, "external module seal mismatch: " + name
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
    assert type(start["control"]) is bool, "START control mode missing"
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


# Prior art: REV2 runner attribution rule; GNU timeout/POSIX signal exits.
# A3 applies the registered D4 mapping, independently in parent and scorer.
RUNNER_OUTCOMES = ("ran", "timeout", "window-closed", "no-window", "no-lock", "rechecked-out")


def scratch_guard(value):
    resolved = Path(value).resolve()
    assert resolved.parent == Path("/tmp") and resolved.name.startswith("gn1_a7_"), "TMPDIR must resolve to /tmp/gn1_a7_*"
    return resolved


def forge_row(path):
    if not path.exists() or not path.read_text().splitlines():
        return None
    fields = path.read_text().splitlines()[-1].split("\t")
    if len(fields) != 9 or fields[7] not in RUNNER_OUTCOMES:
        return ("unrecognized", None)
    try:
        return fields[7], int(fields[8])
    except ValueError:
        return ("unrecognized", None)


def forge_outcome(path, rc):
    row = forge_row(path)
    if row is None:
        return "missing"
    if row[0] == "unrecognized":
        return "unrecognized"
    return row[0] if row[1] == rc else "log-exit-mismatch"


def timeout_cause(path, rc):
    # A5 D4: total attribution. Prior art: GNU coreutils timeout(1), POSIX
    # signal exits; REV4 N5 (2026). No novel attribution rule claimed.
    try:
        row = forge_row(path)
    except Exception:
        return "outer-timeout" if rc in (124, 137) else "unattributed"
    if rc not in (124, 137):
        return None if row is not None and row[0] in RUNNER_OUTCOMES else "unattributed"
    if row is None or row[0] not in ("ran", "timeout", "window-closed"):
        return "outer-timeout"
    if row[1] in (143, 137) and (row[0] != "ran" or row[1] != rc):
        return "outer-timeout"
    if row == ("timeout", rc):
        return "runner-deadline"
    if row == ("window-closed", rc):
        return "window-closed"
    if row == ("ran", rc):
        return "inner-timeout"
    return "unattributed"


def not_admitted(attempt, rc, cause):
    # D2: a single positive start operand burns the label, even if the parent
    # or child was interrupted. Prior art: REV3's evidence-based admission
    # rule (2026); no prior art known to me beyond the order for this predicate.
    row = forge_row(attempt / "forge.log")
    return not (any((attempt / name).exists() for name in
                    ("child_started.json", "invocation.json", "junit.xml", "child_result.json"))
                or row is not None and row[0] in ("ran", "timeout", "window-closed"))


def acceptable_killers(data):
    # D2 lead ruling (a): a wrapped engine refusal counts only in a
    # GroupNorm call phase, with original class/message present. evaluate()
    # applies the device veto; recompute_line() requires same-root candidate
    # pass on each killer ID. Ordinary value assertions remain valid kills.
    # D3 unwrapped refusals must originate at GroupNorm's raising frame.
    return sorted(r["id"] for r in data["reports"]
                  if r["when"] == "call" and r["outcome"] == "failed"
                  and r["id"].startswith("tensor_cuda/tests/test_groupnorm_gn1.py::")
                  and not r["id"].split("::")[-1].startswith("test_reference_")
                  and (r.get("exception_class") == "AssertionError" and (not r.get("wrapped_engine_refusal") or
                       r.get("engine_refusal") is True and bool(r.get("original_exception_class")) and isinstance(r.get("original_exception_message"), str)) or
                       r.get("exception_class") in ("ValueError", "TypeError") and r.get("groupnorm_refusal") is True))


def evaluate(runs, attempt, line, rc, audit, seal):
    reasons = []
    data = None
    killers = []
    outcomes = {}
    unsealed = []
    try:
        if rc not in line["expected_child_exits"]:
            reasons.append("expected exits %s, got %s" % (line["expected_child_exits"], rc))
        data = read(attempt / "child_result.json")
        assert type(data["pytest_exit"]) is int, "child exit is not an integer"
        assert data["pytest_exit"] == rc, "child exit/summary mismatch"
        assert data["arm"] == line["arm"] and data["suite"] == line["suite"], "child arm/suite binding mismatch"
        assert data["audit"] == audit and data["run_root"] == str(runs) and data["attempt_dir"] == str(attempt), "child run-root/audit binding mismatch"
        assert (attempt / "child_started.json").is_file(), "missing child start marker"
        package = verify_arm(line["arm"], seal)
        pin = read(TOOL / "engine_pin.json")
        assert data["nn_sha256"] == seal["arms"][line["arm"]]["nn.py"] == sha(package / "nn.py"), "nn.py binding mismatch"
        assert data["engine_sha256"] == seal["engine_sha256"] == pin["sha256"], "engine binding mismatch"
        assert data["seal_sha256"] == sha(TOOL / "staging_seal.json"), "child seal mismatch"
        runtime_errors, unsealed = runtime_modules(data["runtime_modules"], seal, TOOL, ROOT)
        reasons.extend(runtime_errors)
        assert not data["collection_errors"], "collection errors"
        ids = data["collected_ids"]
        assert len(ids) == len(set(ids)) and sorted(ids) == sorted(census(line["arm"], line["suite"], audit)), "test-ID census mismatch"
        outcomes = summary_outcomes(data)
        dual_ids = {id_ for id_ in ids
                    if any(r["id"] == id_ and r["when"] == "call" and r["outcome"] == "failed" for r in data["reports"])
                    and any(r["id"] == id_ and r["when"] == "teardown" and r["outcome"] == "failed" for r in data["reports"])}
        xml, xml_details = junit_outcomes(attempt / "junit.xml", ids, dual_ids)
        assert xml == outcomes, "JUnit/summary outcomes disagree ID by ID"
        failed_details = [r["detail"] + " " + str(r.get("original_exception_class", "")) + " " + str(r.get("original_exception_message", "")) for r in data["reports"] if r["outcome"] == "failed"] + xml_details
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
                if line["label"] != "V3":
                    live = data["equivalence_runtime"]
                    assert live["numpy_version"] == seal["numpy_version"], "equivalence NumPy version mismatch"
                    assert live["accepts_numpy_bool"] is False and live["exception_class"] == "TypeError", "operator.index accepts numpy.bool_: equivalence invalid"
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
            new_expected = read(TOOL / "baseline_new_failures.json")
            if any(failures.get(i, {}).get("exception_class") != row["exception_class"] for i, row in new_expected.items()):
                reasons.append("baseline new D5 exception class differs from registration")
            if any("reshape: element count mismatch" not in r["detail"]
                   for id_, r in failures.items() if id_ not in new_expected and "refusal" not in id_.split("::")[-1]):
                reasons.append("baseline numerical failure is not registered reshape error")
        any_failure = any(r["outcome"] == "failed" for r in data["reports"])
        if rc == 0 and any_failure or rc == 1 and not any_failure:
            reasons.append("pytest exit inconsistent with failures")
    except Exception as exc:
        reasons.append(str(exc) or type(exc).__name__)
    cause, timeout_reason = "unreadable", None
    try:
        cause = forge_outcome(attempt / "forge.log", rc)
        if cause != "ran":
            reasons.append("forge outcome: " + cause)
        timeout_reason = timeout_cause(attempt / "forge.log", rc)
        if timeout_reason == "unattributed":
            reasons.append("runner attribution: unattributed")
    except Exception as exc:
        reasons.append("runner attribution: " + (str(exc) or type(exc).__name__))
    return dict(passed=not reasons, reasons=reasons, actual_killers=killers if not reasons else [],
                outcomes=outcomes, forge_outcome=cause, timeout=cause == "timeout",
                timeout_cause=timeout_reason, unsealed=unsealed)


def bound_reasons(receipt, line):
    """D7 exact integer binding. Prior art: GN1 registration/REV3; no novel rule."""
    reasons = []
    try:
        assert type(receipt["outer_bound_seconds"]) is int and receipt["outer_bound_seconds"] == line["outer"], "outer bound differs from gpu_lines.json"
        command = receipt["command"]
        split = command.index("--")
        runner, inner = command[:split], command[split+1:]
        # A5 D2 / REV4 F01: closed exact-form grammar. Prior art: Python
        # argparse allow_abbrev (PSF, Python 3.5, 2015), GNU timeout(1).
        # We deliberately accept only the order's argv, not parser aliases.
        assert (receipt.get("control") is True or receipt.get("audit") is True or runner[0] == RUNNER), "runner command[0] differs from pinned runner"
        assert runner[1:] == ["--min", str(line["minimum"]), "--max", str(line["maximum"]),
                              "--label", line["label"]], "runner argv differs from gpu_lines.json"
        assert inner == (["timeout", "--kill-after=10s", f'{line["inner"]}s', "python3",
                         "run_gpu_tests.py", line["arm"], line["suite"]] if (receipt.get("control") is True or receipt.get("audit") is True) and inner[0] == "timeout" else
                         [TIMEOUT, "--kill-after=10s", f'{line["inner"]}s', PYTHON,
                         "run_gpu_tests.py", line["arm"], line["suite"]]), "inner argv differs from gpu_lines.json"
        assert receipt["expected_child_exits"] == line["expected_child_exits"] and all(type(v) is int for v in receipt["expected_child_exits"]), "expected exits differ from gpu_lines.json"
    except Exception as exc:
        reasons.append("bound binding: " + (str(exc) or type(exc).__name__))
    return reasons


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
        if not (attempt / "parent_result.json").exists():
            refused = not_admitted(attempt, None, None)
            inventory.append(dict(attempt=attempt.name, admission="not_admitted" if refused else "admitted", incomplete=True))
            if refused:
                continue
            assert selected is None, "multiple admitted attempts"
            selected = attempt
            result = evaluate(runs, attempt, line, None, start["audit"], seal)
            # L45: a completed child can expose a failure but cannot establish
            # acceptance without its parent's post-exit record. Never retry it.
            result["reasons"].append("incomplete: missing parent completion receipt")
            result["passed"] = False
            result["actual_killers"] = []
            continue
        receipt = read(attempt / "parent_result.json")
        assert not receipt.get("receipt_error"), "; ".join(receipt.get("reasons", ["minimal parent error receipt"]))
        assert receipt["label"] == label and receipt["attempt_dir"] == str(attempt), "parent attempt binding mismatch"
        assert receipt["expected_child_exits"] == line["expected_child_exits"], "parent expected exits differ from gpu_lines.json"
        assert receipt["arm"] == line["arm"] and receipt["suite"] == line["suite"] and receipt["audit"] == start["audit"], "parent line binding mismatch"
        assert receipt["command"][-3:] == ["run_gpu_tests.py", line["arm"], line["suite"]], "parent command arm/suite mismatch"
        assert receipt["cwd"] == str(TOOL), "parent cwd mismatch"
        assert receipt["run_root"] == str(runs) and receipt["seal_sha256"] == sha(TOOL / "staging_seal.json"), "parent run-root/seal mismatch"
        assert isinstance(receipt["exit"], int) and not isinstance(receipt["exit"], bool), "exit is not an integer"
        # A6 D3: recompute Python negative-signal / POSIX128+signal exit.
        assert type(receipt["raw_exit"]) is int, "raw_exit is not an integer"
        assert receipt["exit"] == (receipt["raw_exit"] if receipt["raw_exit"] >= 0 else 128 - receipt["raw_exit"]), "raw_exit normalisation mismatch"
        assert datetime.datetime.fromisoformat(start["date"]) <= datetime.datetime.fromisoformat(receipt["start"]) <= datetime.datetime.fromisoformat(receipt["child_exited"]) <= datetime.datetime.fromisoformat(receipt["recorded"]), "parent completion order invalid"
        cause = forge_outcome(attempt / "forge.log", receipt["exit"])
        assert receipt["forge_outcome"] == cause, "stored forge cause differs from log"
        refused = not_admitted(attempt, receipt["exit"], cause)
        assert receipt["timeout_cause"] == timeout_cause(attempt / "forge.log", receipt["exit"]), "stored timeout cause differs from log"
        assert receipt["admission"] == ("not_admitted" if refused else "admitted"), "admission mismatch"
        inventory.append(dict(attempt=attempt.name, admission=receipt["admission"], exit=receipt["exit"], forge_outcome=cause))
        if refused:
            assert not receipt["passed"] and not receipt["actual_killers"], "refused attempt claims pass/kill"
            assert receipt["reasons"] == ["not admitted: " + cause] and receipt["outcomes"] == {} and receipt["timeout"] is False, "stored refused receipt disagrees with recomputation"
            continue
        assert selected is None, "multiple admitted attempts"
        selected = attempt
        result = evaluate(runs, attempt, line, receipt["exit"], start["audit"], seal)
        if not start["audit"] and line["suite"] == "mutation":
            dependency = recompute_line(runs, "candidate_focus", seal, start, (*stack, label))
            if not dependency["passed"] or any(dependency["outcomes"].get(i) != "passed" for i in result["actual_killers"]):
                result["reasons"].append("mutations require recomputed passing candidate focus on killer IDs")
        if not start["audit"] and label == "candidate_regression":
            baseline = recompute_line(runs, "baseline_regression", seal, start, (*stack, label))
            if not baseline["passed"]:
                result["reasons"].append("missing valid completed baseline regression")
            elif (attempt / "child_result.json").exists():
                result["reasons"] += regression_reasons(read(Path(baseline["selected_attempt"]) / "child_result.json"), read(attempt / "child_result.json"))
        result["passed"] = not result["reasons"]
        if not result["passed"]:
            result["actual_killers"] = []
        for key in ("passed", "reasons", "actual_killers", "forge_outcome", "timeout", "timeout_cause", "outcomes"):
            assert receipt[key] == result[key], "stored parent receipt disagrees with recomputation: " + key
        assert receipt["unsealed"] == result["unsealed"], "stored runtime census differs from recomputation"
        # A7 D2/D3: derive mode solely from the child-owned evidence.
        # Prior art: CUDA runtime queries/SHA-256, see device_evidence.py;
        # this exact evidence/mode conjunction is mandated by A7.
        data = read(attempt / "child_result.json")
        mode, device_errors = classify(data, seal)
        result["mode"] = mode
        result["reasons"].extend(device_errors)
        if mode == "device":
            invocation = read(attempt / "invocation.json")
            assert invocation["device_evidence"] == data["device_evidence"], "pre-test child device evidence mismatch"
            captured = datetime.datetime.fromisoformat(data["device_evidence"]["captured_at"])
            assert datetime.datetime.fromisoformat(receipt["start"]) <= captured <= datetime.datetime.fromisoformat(receipt["child_exited"]), "device evidence timestamp outside child run"
        declared_cpu = receipt["control"] or receipt["audit"]
        if (mode == "cpu") != declared_cpu:
            result["reasons"].append("mode binding: parent declaration disagrees with child device evidence")
        result["reasons"].extend(bound_reasons(receipt, line))
        result["reasons"].extend(program_reasons(receipt, seal, start["control"], start["audit"]))
        result["passed"] = not result["reasons"]
        if not result["passed"]:
            result["actual_killers"] = []
    result.update(selected_attempt=str(selected) if selected else None, attempts=inventory)
    return result
