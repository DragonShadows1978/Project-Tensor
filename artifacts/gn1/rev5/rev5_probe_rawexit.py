#!/usr/bin/env python3
"""GN1-REV5 probe: does A5's scorer recompute the D3-normalised exit from raw_exit?
Honest candidate_focus line via the stand-in runner in the mirror, then the
reviewer edits ITS OWN scratch receipt: (a) raw_exit -> -9 with exit kept 0;
(b) raw_exit removed. Scorer re-run each time. Prior art: fault injection
(Meszaros 2007); no prior art known to me for the specific check.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostile_rev5 as H  # noqa: E402

H.BASE.mkdir(parents=True)
H.make_tools()
out = []
for name, edit in (("rawexit_minus9_exit0", lambda r: r.update(raw_exit=-9)),
                   ("rawexit_removed", lambda r: r.pop("raw_exit"))):
    runs = H.start("PROBE_" + name)
    p = H.parent(runs, "candidate_focus", [*H.SR(), *H.CANON, "--", *H.INNER])
    before = H.score(runs, "candidate_focus")
    rec = runs / "candidate_focus" / "attempt-0001" / "parent_result.json"
    data = json.loads(rec.read_text())
    edit(data)
    rec.write_text(json.dumps(data, indent=2) + "\n")
    after = H.score(runs, "candidate_focus")
    out.append(dict(case=name, parent_exit=p["exit"], line_before=(before.get("line") or {}).get("passed"),
                    line_after=(after.get("line") or {}).get("passed"), reasons_after=(after.get("line") or {}).get("reasons"),
                    stored_exit=data["exit"], stored_raw_exit=data.get("raw_exit", "<absent>")))
    print(out[-1])
(Path(__file__).resolve().parent / "rev5_probe_rawexit.json").write_text(json.dumps(out, indent=1) + "\n")
