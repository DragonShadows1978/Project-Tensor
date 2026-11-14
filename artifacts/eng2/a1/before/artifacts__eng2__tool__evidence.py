"""Shared, read-only evidence recomputation for the ENG2 parent and scorer (copied from ENG1's tool as A3 left it,
which was adapted from GN1 A8 evidence.py).
ENG2 changes (ledgered in artifacts/eng2/LEDGER.md and COPY_DEPS.json): ORDER/SEAL_NAME bound to the ENG2 order and
staging_seal.json; every artifacts/eng1 path and /tmp/eng1_* root rebound to eng2; the A1-suffixed census and
expectation files renamed (ids_<census>.txt, baseline_expected_failures.json); the mutant registry is
artifacts/eng2/mutants/mutants.json; the mutant dependency label is candidate_eng2; ENG1's terrain/E6 record dropped
(ENG2 has no terrain suite, so e6_record.py and its git call are not imported); L41: a mutant line needs a failing
registered killer whose child trace shows the mutated hunk executing (mutant_trace).

Prior art: SHA-256 (NIST 2001); pytest/JUnit (Krekel et al., 2004+); DeMillo/Lipton/Sayward (1978), valid mutation
kills; exact paired regression comparison from GN1 D6. Literature unverified -- lead to check named sources.
ENG1 changes (ledgered): per-arm engine pins (the arms differ in C++); ENG1 suites (eng1, mutation, regression);
mutant killers from mutants/mutants.json; the baseline's registered per-ID failures; test IDs normalised by
stripping the arm prefix; GN1's A4-A8 control/hostile/program-binding conjunctions dropped (GN1 contract C; their
program_binding.py is not in the tree).
"""
import datetime
import fnmatch
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from device_evidence import classify
from arm_origin import validate_origin

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
ORDER = "orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md"  # ENG2: was orders/ENG1_A3_DEVICE_SEAL_ROUTING.md
SEAL_NAME = "staging_seal.json"  # ENG2: was staging_seal_a3.json (one seal, written by artifacts/eng2/tool/stage.py)
DEVICE_PATTERNS = (
    "cudaMalloc", "no CUDA-capable device", "CUDA driver", "cudaError",
    "initialization error", "invalid device ordinal", "CUDA error at",
    "cudaMallocAsync failed", "out of memory", "illegal memory access",
    "unspecified launch failure", "launch timed out",
)
RUNNER_OUTCOMES = ("ran", "timeout", "window-closed", "no-window", "no-lock", "rechecked-out")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def guard(runs):
    runs = Path(runs).resolve()
    if not runs.is_relative_to(TOOL) or runs == TOOL or runs.is_relative_to(TOOL / "arms"):
        raise ValueError("RUNS must be under artifacts/eng2/tool and outside arms")
    return runs


def lines():
    return {r["label"]: r for r in read(TOOL / "gpu_lines.json")}


def arm_files(arm):
    base = TOOL / "arms" / arm
    # (ENG1 excluded arms/<arm>/artifacts/, its terrain write target; ENG2 keeps the same exclusion, unused.)
    return {str(p.relative_to(base)): sha(p) for p in sorted(base.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.relative_to(base).parts[0] != "artifacts"}


def verify_seal():
    seal = read(TOOL / SEAL_NAME)
    assert sha(ROOT / ORDER) == seal["order_sha256"], "order seal mismatch"
    assert re.fullmatch(r"[0-9a-f]{40}", seal["order_commit"]), "missing order commit"
    for name, digest in seal["tool_files"].items():
        assert sha(TOOL / name) == digest, "tool seal mismatch: " + name
    for name, digest in seal["registration_files"].items():
        assert sha(ROOT / name) == digest, "registration seal mismatch: " + name
    return seal


def verify_arm(arm, seal):
    assert arm in seal["arms"], "unregistered arm"
    base = TOOL / "arms" / arm
    assert not any(p.is_symlink() for p in base.rglob("*")), "arm contains symlink"
    files = arm_files(arm)
    assert files == seal["arms"][arm], "arm file/seal mismatch: " + arm
    module = "tensor_cuda/tensor_cuda/" + seal["module_name"]
    assert files[module] == seal["engine_sha256"][arm], "arm engine differs from its pin"
    return base / "tensor_cuda"


def verify_start(runs):
    runs = guard(runs)
    seal = verify_seal()
    start = read(runs / "START.json")
    assert start["run_root"] == str(runs), "START run-root binding mismatch"
    assert start["seal_sha256"] == sha(TOOL / SEAL_NAME), "START seal mismatch"
    assert start["order_sha256"] == seal["order_sha256"] and start["order_commit"] == seal["order_commit"], "START order mismatch"
    assert isinstance(start["audit"], bool), "START audit missing"
    datetime.datetime.fromisoformat(start["date"])
    return seal, start


def norm_id(id_):
    """Strip the arm prefix so IDs compare across arms: artifacts/eng2/tool/arms/<arm>/tensor_cuda/... ->
    tensor_cuda/..."""
    m = re.match(r"artifacts/eng2/tool/arms/[^/]+/(tensor_cuda/.*)$", id_)
    return m.group(1) if m else id_


def census(line, audit=False):
    name = "audit" if audit else line["census"]
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
    outcomes, parts, details = {}, {}, []
    for case in ET.parse(path).getroot().iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        assert key in mapping, "unregistered JUnit ID: " + str(key)
        id_ = mapping[key]
        tags = {e.tag for e in case if e.tag in ("failure", "error", "skipped")}
        assert len(tags) <= 1, "multiple JUnit outcomes"
        outcome = "error" if "error" in tags else "failed" if "failure" in tags else "skipped" if "skipped" in tags else "passed"
        if id_ in outcomes:
            assert id_ in dual_ids and len(parts[id_]) == 1 and {parts[id_][0], outcome} == {"failed", "error"}, "duplicate JUnit ID: " + id_
            outcomes[id_] = "error"
            parts[id_].append(outcome)
        else:
            outcomes[id_] = outcome
            parts[id_] = [outcome]
        details.extend(" ".join(e.itertext()) + " " + e.get("message", "") for e in case if e.tag in ("error", "failure"))
    return outcomes, details


def scratch_guard(value):
    resolved = Path(value).resolve()
    assert resolved.parent == Path("/tmp") and resolved.name.startswith("eng2_"), "TMPDIR must resolve to /tmp/eng2_*"
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
    row = forge_row(attempt / "forge.log")
    return not (any((attempt / name).exists() for name in
                    ("child_started.json", "invocation.json", "junit.xml", "child_result.json"))
                or row is not None and row[0] in ("ran", "timeout", "window-closed"))


def killers_for(label):
    m = {r["id"]: r for r in read(TOOL.parent / "mutants" / "mutants.json")}[label]
    return m["killers"]


def literal_brackets(pattern):
    """Registered killer patterns use '*' as the only wildcard; '[' and ']' are literal (pytest parameter IDs)."""
    return pattern.replace("[", "\x00").replace("]", "[]]").replace("\x00", "[[]")


def matches(nid, patterns):
    short = nid.split("::", 1)[1] if "::" in nid else nid
    return any(fnmatch.fnmatchcase(short, literal_brackets(p)) for p in patterns)


def evaluate(runs, attempt, line, rc, audit, seal):
    reasons, killers, outcomes = [], [], {}
    try:
        if rc not in line["expected_child_exits"]:
            reasons.append("expected exits %s, got %s" % (line["expected_child_exits"], rc))
        data = read(attempt / "child_result.json")
        assert type(data["pytest_exit"]) is int, "child exit is not an integer"
        assert data["pytest_exit"] == rc, "child exit/summary mismatch"
        assert data["arm"] == line["arm"] and data["suite"] == line["suite"], "child arm/suite binding mismatch"
        assert data["audit"] == audit and data["run_root"] == str(runs) and data["attempt_dir"] == str(attempt), "child run-root/audit binding mismatch"
        assert (attempt / "child_started.json").is_file(), "missing child start marker"
        verify_arm(line["arm"], seal)
        assert data["engine_sha256"] == seal["engine_sha256"][line["arm"]], "engine binding mismatch"
        origin = read(attempt / "origin_check.json")
        assert origin == data["origin_check"], "origin receipt/child binding mismatch"
        validate_origin(origin, TOOL / "arms" / line["arm"] / "tensor_cuda", seal["module_name"],
                        seal["engine_sha256"][line["arm"]], TOOL)
        assert data["seal_sha256"] == sha(TOOL / SEAL_NAME), "child seal mismatch"
        assert not data["collection_errors"], "collection errors"
        ids = data["collected_ids"]
        assert len(ids) == len(set(ids)), "duplicate IDs"
        assert sorted(norm_id(i) for i in ids) == sorted(census(line, audit)), "test-ID census mismatch"
        outcomes = summary_outcomes(data)
        dual_ids = {i for i in ids
                    if any(r["id"] == i and r["when"] == "call" and r["outcome"] == "failed" for r in data["reports"])
                    and any(r["id"] == i and r["when"] == "teardown" and r["outcome"] == "failed" for r in data["reports"])}
        xml, xml_details = junit_outcomes(attempt / "junit.xml", ids, dual_ids)
        assert xml == outcomes, "JUnit/summary outcomes disagree ID by ID"
        failed_details = [r["detail"] for r in data["reports"] if r["outcome"] == "failed"] + xml_details
        if any(p.casefold() in t.casefold() for t in failed_details for p in DEVICE_PATTERNS):
            reasons.append("device-pattern failure: void line")
        if line["role"] != "regression" and any(r["when"] != "call" and r["outcome"] == "failed" for r in data["reports"]):
            reasons.append("setup/teardown failures")
        if any(o == "missing" for o in outcomes.values()):
            reasons.append("missing test calls")
        if line["role"] != "regression" and any(o == "skipped" for o in outcomes.values()):
            reasons.append("unregistered skip")
        failed = sorted(norm_id(i) for i, o in outcomes.items() if o in ("failed", "error"))
        if audit or line["role"] in ("candidate", "control"):
            if failed:
                reasons.append("candidate/control/audit IDs did not all pass: " + repr(failed[:20]))
        elif line["role"] == "mutant":
            killers = [i for i in failed if matches(i, killers_for(line["label"]))]
            if not killers:
                reasons.append("no registered killer ID failed")
            # ENG2 L41: at least one failing registered killer executed the mutated hunk (child-recorded trace).
            trace = {norm_id(k): v for k, v in (data.get("mutant_trace") or {}).items()}
            if killers and not any(trace.get(i) for i in killers):
                reasons.append("L41: no failing registered killer executed the mutated hunk")
        elif line["role"] == "baseline":
            registered = sorted(read(TOOL / "baseline_expected_failures.json"))
            if failed != registered:
                reasons.append("baseline failures differ from registration: unexpected %r, missing %r" % (
                    sorted(set(failed) - set(registered))[:20], sorted(set(registered) - set(failed))[:20]))
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
                outcomes={norm_id(k): v for k, v in outcomes.items()}, forge_outcome=cause, timeout=cause == "timeout",
                timeout_cause=timeout_reason)


def attempts(runs, label):
    base = runs / label
    if not base.exists():
        return []
    dirs = sorted(base.glob("attempt-*"))
    assert [p.name for p in dirs] == [f"attempt-{i:04d}" for i in range(1, len(dirs)+1)], "nonconsecutive attempts"
    return dirs


def regression_reasons(old, new):
    # Explicit source-path readers run on candidate; they are not numerical arm comparisons (REV1 F02).
    excluded = {r["test_id"] for r in read(TOOL / "source_reading_tests.json")}
    before = {norm_id(k): v for k, v in summary_outcomes(old).items()
              if norm_id(k).split("[", 1)[0] not in excluded}
    after = {norm_id(k): v for k, v in summary_outcomes(new).items()
             if norm_id(k).split("[", 1)[0] not in excluded}
    # ENG2 L38: pins this order changes (artifacts/eng2/L38_PINS.json expected_changes) are not regressions; each must
    # instead show its registered baseline and candidate outcomes and, on the candidate, its registered failure text.
    pins = {k: v for k, v in read(TOOL.parent / "L38_PINS.json")["expected_changes"].items() if k in before or k in after}
    lost = sorted(i for i, o in before.items() if o == "passed" and after.get(i) != "passed" and i not in pins)
    reasons = ["baseline-passing IDs no longer pass: " + repr(lost)] if lost else []
    details = {}
    for r in new["reports"]:
        if r["outcome"] == "failed":
            details.setdefault(norm_id(r["id"]), []).append(r["detail"])
    for i, pin in sorted(pins.items()):
        if before.get(i) != pin["baseline"] or after.get(i) != pin["candidate"]:
            reasons.append("L38 expected change not as registered: %s (baseline %s, candidate %s)" % (i, before.get(i), after.get(i)))
        elif pin["candidate"] == "failed" and not any(pin["detail_contains"] in d for d in details.get(i, [])):
            reasons.append("L38 expected change has another failure text: " + i)
    return reasons


def recompute_line(runs, label, seal, start, stack=()):
    assert label not in stack, "dependency cycle"
    line = lines()[label]
    inventory, selected = [], None
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
            # L45: a completed child cannot establish acceptance without its parent's post-exit record.
            result["reasons"].append("incomplete: missing parent completion receipt")
            result["passed"], result["actual_killers"] = False, []
            continue
        receipt = read(attempt / "parent_result.json")
        assert not receipt.get("receipt_error"), "; ".join(receipt.get("reasons", ["minimal parent error receipt"]))
        assert receipt["label"] == label and receipt["attempt_dir"] == str(attempt), "parent attempt binding mismatch"
        assert receipt["expected_child_exits"] == line["expected_child_exits"], "parent expected exits differ from gpu_lines.json"
        assert receipt["arm"] == line["arm"] and receipt["suite"] == line["suite"] and receipt["audit"] == start["audit"], "parent line binding mismatch"
        assert receipt["cwd"] == str(TOOL), "parent cwd mismatch"
        assert receipt["run_root"] == str(runs) and receipt["seal_sha256"] == sha(TOOL / SEAL_NAME), "parent run-root/seal mismatch"
        assert type(receipt["raw_exit"]) is int and type(receipt["exit"]) is int, "exit is not an integer"
        assert receipt["exit"] == (receipt["raw_exit"] if receipt["raw_exit"] >= 0 else 128 - receipt["raw_exit"]), "raw_exit normalisation mismatch"
        assert type(receipt["timeout"]) is bool, "timeout flag missing"
        assert datetime.datetime.fromisoformat(start["date"]) <= datetime.datetime.fromisoformat(receipt["start"]) <= datetime.datetime.fromisoformat(receipt["child_exited"]) <= datetime.datetime.fromisoformat(receipt["recorded"]), "parent completion order invalid"
        cause = forge_outcome(attempt / "forge.log", receipt["exit"])
        assert receipt["forge_outcome"] == cause, "stored forge cause differs from log"
        refused = not_admitted(attempt, receipt["exit"], cause)
        assert receipt["admission"] == ("not_admitted" if refused else "admitted"), "admission mismatch"
        inventory.append(dict(attempt=attempt.name, admission=receipt["admission"], exit=receipt["exit"], forge_outcome=cause))
        if refused:
            continue
        assert selected is None, "multiple admitted attempts"
        selected = attempt
        result = evaluate(runs, attempt, line, receipt["exit"], start["audit"], seal)
        if not start["audit"] and line["role"] == "mutant":
            dependency = recompute_line(runs, "candidate_eng2", seal, start, (*stack, label))
            if not dependency["passed"] or any(dependency["outcomes"].get(i) != "passed" for i in result["actual_killers"]):
                result["reasons"].append("mutations require a passing candidate_eng2 on the killer IDs")
        if not start["audit"] and line.get("control_of"):
            pass  # the control's own pass is its evidence; the mutant line cites it (score_gpu.py)
        if not start["audit"] and line["role"] == "regression" and line["arm"] == "candidate":
            baseline = recompute_line(runs, label.replace("candidate_", "baseline_", 1), seal, start, (*stack, label))
            if not baseline["passed"]:
                result["reasons"].append("missing valid completed baseline regression")
            else:
                result["reasons"] += regression_reasons(read(Path(baseline["selected_attempt"]) / "child_result.json"),
                                                        read(attempt / "child_result.json"))
        for key in ("forge_outcome", "timeout_cause"):
            assert receipt[key] == result[key], "stored parent receipt disagrees with recomputation: " + key
        data = read(attempt / "child_result.json")
        mode, device_errors = classify(data, seal)
        result["mode"] = mode
        result["reasons"].extend(device_errors)
        if (mode == "cpu") != bool(start["audit"]):
            result["reasons"].append("mode binding: START audit flag disagrees with child device evidence")
        result["passed"] = not result["reasons"]
        if not result["passed"]:
            result["actual_killers"] = []
    result.update(selected_attempt=str(selected) if selected else None, attempts=inventory)
    return result
