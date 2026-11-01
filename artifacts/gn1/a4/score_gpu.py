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
        for label in lines():
            try:
                summary[label] = recompute_line(runs, label, seal, start)
            except (AssertionError, KeyError, OSError, ValueError, TypeError) as exc:
                inventory = []
                for path in sorted((runs / label).glob("attempt-*")):
                    inventory.append(dict(attempt=path.name, receipt=str(path / "parent_result.json")))
                summary[label] = dict(passed=False, reasons=[str(exc) or type(exc).__name__], attempts=inventory)
    except (AssertionError, KeyError, OSError, ValueError, TypeError) as exc:
        errors.append(str(exc) or type(exc).__name__)
    passed = not errors and bool(summary) and all(row["passed"] for row in summary.values())
    result = dict(passed=passed, errors=errors, lines=summary)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "G5_SUMMARY.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
