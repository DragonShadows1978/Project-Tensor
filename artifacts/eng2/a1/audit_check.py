#!/usr/bin/env python3
"""ENG2-A1 A-G3 audit-route receipt check for tool/gpu_runs/audit_2 (the ten steps of g4/G4_AUDIT_RECEIPT.json, re-derived
from the files), plus the dry-run receipt's seal binding. Writes a1/AG3_AUDIT_RECEIPT.json.
Prior art: GN1/ENG1 audit receipts (2026); no new algorithm."""
import datetime, hashlib, json, sys
from pathlib import Path
A1 = Path(__file__).resolve().parent
TOOL = A1.parent / "tool"
R = TOOL / "gpu_runs" / "audit_2"
A = R / "candidate_eng2" / "attempt-0001"
r = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = sha(TOOL / "staging_seal.json")
start, child, parent = r(R / "START.json"), r(A / "child_result.json"), r(A / "parent_result.json")
summary = r(R / "G5_SUMMARY.json")
row = (A / "forge.log").read_text().splitlines()[-1].split("\t")
t = datetime.datetime.fromisoformat
steps = [("START bound to the seal", start["seal_sha256"] == seal and start["audit"] is True),
         ("child start marker", (A / "child_started.json").is_file()),
         ("origin check (arm package + candidate engine pin)", r(A / "origin_check.json") == child["origin_check"]),
         ("invocation record", r(A / "invocation.json")["seal_sha256"] == seal),
         ("junit.xml", (A / "junit.xml").is_file()),
         ("child result: pytest exit 0, %d IDs (-k cpu), 0 collection errors" % len(child["collected_ids"]),
          child["pytest_exit"] == 0 and len(child["collected_ids"]) == 157 and not child["collection_errors"]),
         ("runner log row 'ran 0' (stand-in)", row[7:] == ["ran", "0"]),
         ("parent receipt passed, recorded after the child exited",
          parent["passed"] and parent["admission"] == "admitted" and t(parent["child_exited"]) <= t(parent["recorded"])),
         ("re-scoring (score_gpu.py) recomputes candidate_eng2 as passed",
          summary["lines"]["candidate_eng2"]["passed"] is True and summary["lines"]["candidate_eng2"]["mode"] == "cpu"),
         ("score_gpu.py exit 1 with the designed audit error",
          (A1 / "audit_score.exit").read_text().strip() == "score rc=1"
          and "audit run cannot establish GPU acceptance" in summary["errors"])]
dry = r(A1 / "G4_DRY_dry_2.json")
steps.append(("dry run dry_2: all 21 labels ok, every start record bound to the new seal",
              dry["all_ok"] and len(dry["rows"]) == 21 and all(x["start_record"]["seal_sha256"] == seal for x in dry["rows"])))
out = dict(root=str(R.relative_to(A1.parents[2])), seal_sha256=seal, steps=[dict(step=s, ok=bool(ok)) for s, ok in steps],
           all_ok=all(ok for _, ok in steps))
(A1 / "AG3_AUDIT_RECEIPT.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
sys.exit(0 if out["all_ok"] else 1)
