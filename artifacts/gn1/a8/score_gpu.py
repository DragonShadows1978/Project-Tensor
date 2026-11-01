"""Recompute every verdict from START, seal, child and JUnit; no GPU calls.
Prior art: DeMillo et al. (1978), mutation adequacy; paired regression (GN1).
A2 never treats a stored parent verdict as its source of truth.
"""
import json
import os
import sys
from evidence import TOOL, guard, verify_start, lines, recompute_line


def main():
    # Guard before creating/writing anything: hostile outside roots stay untouched.
    runs = guard(os.environ["RUNS"])
    summary = {}
    errors = []
    try:
        seal, start = verify_start(runs)
        assert not start["audit"], "audit run cannot establish GPU acceptance"
        # A6 D1: recompute controls for target sensitivity, bar root acceptance.
        for label in lines():
            try:
                summary[label] = recompute_line(runs, label, seal, start)
            except Exception as exc:
                inventory = []
                for path in sorted((runs / label).glob("attempt-*")):
                    inventory.append(dict(attempt=path.name, receipt=str(path / "parent_result.json")))
                summary[label] = dict(passed=False, reasons=[str(exc) or type(exc).__name__], attempts=inventory)
    except Exception as exc:
        errors.append(str(exc) or type(exc).__name__)
    # A7 D2/D3: no declaration can promote a CPU line. CUDA runtime API
    # evidence is checked per admitted child; see device_evidence.py.
    if summary and any(row.get("mode") != "device" for row in summary.values()):
        errors.append("not a device run: every line requires positive child device evidence")
    # A8 D3: a root declaration can only demote. A CPU root has one
    # deciding reason: absent positive device evidence. Prior art: A7.
    if summary and all(row.get("mode") == "device" for row in summary.values()) and start["control"]:
        errors.append("control run cannot establish GPU acceptance")
    passed = not errors and bool(summary) and all(row["passed"] for row in summary.values())
    result = dict(passed=passed, errors=errors, lines=summary)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "G5_SUMMARY.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
