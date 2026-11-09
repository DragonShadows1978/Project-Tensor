#!/usr/bin/env python3
"""GN1-REV6 L41: compare the mirror's G5_mutants.json (A8's real cpu_checks.py,
run through checks_wrap6.py) with A8's on-disk G5_mutants.json, field by field,
and list per mutant whether an edited line is among the executed (reached) lines.
Prior art: REV5 mutant_reach_compare.json (2026); line-coverage reach as in
mutation testing's RIP model (reach/infect/propagate: Offutt and colleagues,
1990s; unverified - lead to check: "RIP model mutation reachability infection
propagation"). Ours: the field list.
"""
import json, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
A8 = OUT.parent / "a8"
MA8 = Path("/tmp/gn1_rev6_m/W/artifacts/gn1/a8")
a = {m["id"]: m for m in json.loads((A8 / "G5_mutants.json").read_text())["mutants"]}
b = {m["id"]: m for m in json.loads((MA8 / "G5_mutants.json").read_text())["mutants"]}
fields = ["id", "witness", "edited_lines", "reached_lines", "reached", "loads", "differs", "cpu_killed",
          "cpu_assertion_killed", "cpu_refusal_killed", "equivalence", "passed"]
diff = sorted((k, f) for k in set(a) | set(b) for f in fields if a.get(k, {}).get(f) != b.get(k, {}).get(f))
rows = {k: dict(edited=m["edited_lines"], reached=m["reached_lines"], edit_site_reached=bool(set(m["edited_lines"]) & set(m["reached_lines"])),
                default_binding=bool(m.get("default_binding")), killed=m["cpu_killed"], equivalence=bool(m["equivalence"]), passed=m["passed"]) for k, m in sorted(b.items())}
res = dict(a8_passed=json.loads((A8 / "G5_mutants.json").read_text())["passed"],
           rev6_passed=json.loads((MA8 / "G5_mutants.json").read_text())["passed"],
           mutants=len(b), identical_fields=fields, differing=diff,
           reached=sum(r["passed"] and (r["edit_site_reached"] or r["default_binding"]) for r in rows.values()),
           edit_site_in_reached=sum(r["edit_site_reached"] for r in rows.values()),
           not_edit_site=[k for k, r in rows.items() if not r["edit_site_reached"]],
           cpu_killed=sum(r["killed"] for r in rows.values()),
           equivalent=[k for k, r in rows.items() if r["equivalence"]], per_mutant=rows)
(OUT / "mutant_reach_compare.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "per_mutant"}, indent=1))
