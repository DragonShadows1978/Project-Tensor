"""Aggregate only completed parent receipts; no GPU calls.
Prior art: DeMillo, Lipton and Sayward (1978), mutation score; exact-ID paired
regression comparison from the GN1 order. No equivalence proofs fabricated.
"""
import json
import os
from pathlib import Path
import sys

tool = Path(__file__).resolve().parent
runs = Path(os.environ["RUNS"]).resolve()
lines = json.loads((tool / "gpu_lines.json").read_text())
summary = {}
for line in lines:
    file = runs / line["label"] / "parent_result.json"
    data = json.loads(file.read_text()) if file.exists() else None
    summary[line["label"]] = dict(passed=bool(data and data["passed"] and not data["timeout"] and not data["audit"]),
                                  receipt=str(file), actual_killers=(data or {}).get("actual_killers", []))
passed = all(row["passed"] for row in summary.values())
(runs / "G5_SUMMARY.json").write_text(json.dumps(dict(passed=passed, lines=summary), indent=2) + "\n")
print(json.dumps(dict(passed=passed, lines=summary), indent=2))
sys.exit(0 if passed else 1)
