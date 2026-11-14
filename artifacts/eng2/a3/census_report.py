#!/usr/bin/env python3
"""ENG2-A3 R2: print the census tables and the full site lists from a3/CONSUMER_CENSUS.json (read only; stdout).
Usage: census_report.py [tables|newly|unresolved_changed|unresolved_unchanged|files]
Prior art: none needed (a JSON pretty-printer); no prior art known to me."""
import collections
import json
from pathlib import Path
import sys

A3 = Path(__file__).resolve().parent
d = json.loads((A3 / "CONSUMER_CENSUS.json").read_text())
what = sys.argv[1] if len(sys.argv) > 1 else "tables"
if what == "tables":
    for m in d["repos"]:
        print("%(repo)s branch %(branch)s HEAD %(head)s: tracked .py %(tracked_py)d; listed %(listed)d; cross-file %(cross_file)d; "
              "supplementary %(supplementary)d; WorkTree dropped %(dropped_worktree)d" % m)
    print("\n| Repository | Entries | Class | Sites | Live | Receipt |\n|---|---|---|---:|---:|---:|")
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for c in d["counts"]:
        a = agg[(c["repo"], c["entry_kind"], c["cls"])]
        a[0] += c["total"]; a[1] += c["live"]; a[2] += c["receipt"]
    for k in sorted(agg):
        print("| %s | %s | %s | %d | %d | %d |" % (*k, *agg[k]))
    print("\n| Repository | Group | Role | Files | Sites |\n|---|---|---|---:|---:|")
    fc = collections.Counter((f["repo"], f["group"], f["role"]) for f in d["files"])
    sc = collections.Counter((f["repo"], f["group"], f["role"]) for f in d["files"] for _ in range(f["sites"]))
    for k in sorted(fc):
        print("| %s | %s | %s | %d | %d |" % (*k, fc[k], sc[k]))
    print("\n| Repository | Entry | Sites |\n|---|---|---:|")
    for k, v in sorted(collections.Counter((r["repo"], r["entry"]) for r in d["rows"]).items()):
        print("| %s | %s | %d |" % (*k, v))
    print("\nparse failures:", d["parse_failures"])
elif what == "files":
    for f in d["files"]:
        if f["group"] != "import" or f["cross_file_aliases"]:
            print(f["repo"], f["file"], f["group"], f["role"], "sites", f["sites"], "aliases", f["cross_file_aliases"])
else:
    rows = d[what]
    if what == "unresolved_unchanged":
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["repo"], r["file"], r["role"], r["entry"])].append(r["line"])
        for (repo, f, role, entry), lines in sorted(by.items()):
            print("%s %s:%s (%s; %s)" % (repo, f, ",".join(map(str, sorted(lines))), role, entry))
    else:
        for r in rows:
            print("%s %s:%d | %s | %s%s | %s | arg %s | %s | %s" % (
                r["repo"], r["file"], r["line"], r["role"], r["group"], " (in string code)" if r["in_string_code"] else "",
                r["entry"], r["arg"], r["cls"], "; ".join(r["values"][:4])))
    print(len(rows), what)
