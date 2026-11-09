#!/usr/bin/env python3
"""Recompute every ENG1 verdict from START, seal, parent, child and JUnit receipts; no GPU calls (adapted from
GN1 A8 score_gpu.py). Prior art: DeMillo et al. (1978), mutation adequacy; paired regression (GN1 D6).
ENG1 changes (ledgered): M09 counts only with a passing C_UBSAN control; the E3 census rows are collected from
candidate_eng1 and reported (not gated); the per-mutant killer IDs are listed.
"""
import json
import os
import sys
from pathlib import Path
from evidence import TOOL, guard, verify_start, lines, recompute_line


def main():
    runs = guard(os.environ["RUNS"])
    summary, errors = {}, []
    start = {}
    try:
        seal, start = verify_start(runs)
        for label in lines():
            try:
                summary[label] = recompute_line(runs, label, seal, start)
            except Exception as exc:
                summary[label] = dict(passed=False, reasons=[str(exc) or type(exc).__name__])
    except Exception as exc:
        errors.append(str(exc) or type(exc).__name__)
    if start.get("audit"):
        errors.append("audit run cannot establish GPU acceptance")
    if summary and any(row.get("mode") != "device" for row in summary.values()):
        errors.append("not a device run: every line requires positive child device evidence")
    if "M09" in summary and not summary.get("C_UBSAN", {}).get("passed"):
        summary["M09"]["passed"] = False
        summary["M09"].setdefault("reasons", []).append("M09 needs a passing C_UBSAN control")
    census = []
    cand = summary.get("candidate_eng1", {}).get("selected_attempt")
    if cand:
        census = [json.loads(p.read_text()) for p in sorted(Path(cand).glob("census_*.json"))]
    passed = not errors and bool(summary) and all(row.get("passed") for row in summary.values())
    result = dict(passed=passed, errors=errors, lines=summary,
                  killers={k: v.get("actual_killers") for k, v in summary.items() if lines()[k]["role"] == "mutant"},
                  e3_census=census)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "G5_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, errors=errors,
                          lines={k: (v.get("passed"), v.get("mode"), v.get("reasons", [])[:3]) for k, v in summary.items()})))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
