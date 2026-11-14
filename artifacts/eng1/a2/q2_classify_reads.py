#!/usr/bin/env python3
"""ENG1-REV2 Q2: classify the dynamically attributed reads (q2_readers_{A,B,C}.json) and compare with A1's 16.
A test is a source-path reader here if, outside imports/traceback rendering/temp-tree walks, it reads (or dlopens)
a worktree file that is not a receipt under artifacts/ and not an order under orders/. Prior art: none beyond PEP 578.
"""
import json
from pathlib import Path
REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
a1 = {r["test_id"] for r in json.loads((ROOT / "artifacts/eng1/tool/source_reading_tests.json").read_text())}
eng1_changed = {"tensor_cuda/src/conv.cu", "tensor_cuda/src/kernels.cu", "tensor_cuda/src/ops.cpp",
                "tensor_cuda/tests/test_terrain_render.py"}
found, outcomes, blocked = {}, {}, []
for g in "ABC":
    d = json.loads((REV2 / f"q2_readers_{g}.json").read_text())
    outcomes.update(d["outcomes"]); blocked += d["blocked"]
    for test, rows in d["reads_by_test"].items():
        for path, opener, kind in rows:
            if kind not in ("data", "dlopen") or not (ROOT / path).exists() or (ROOT / path).is_dir():
                continue
            if path.startswith(("artifacts/", "orders/")) or "__pycache__" in path:
                continue
            found.setdefault(test.split("[", 1)[0], set()).add((path, opener, kind))
mine = set(found) - {"<collection>", "<between>"}
rows = []
for t in sorted(mine | a1):
    paths = sorted(found.get(t, ()))
    rows.append(dict(test_id=t, in_a1_list=t in a1, observed_source_reads=[p for p, _, _ in paths][:40],
                     n_observed=len(paths), reads_eng1_changed_file=sorted({p for p, _, _ in paths} & eng1_changed),
                     outcome={k.split("|")[1]: v for k, v in outcomes.items() if k.split("[", 1)[0].split("|")[0] == t}))
res = dict(a1_count=len(a1), observed_readers=len(mine), in_both=len(mine & a1), observed_not_in_a1=sorted(mine - a1),
           a1_not_observed=sorted(a1 - mine), collection_reads=sorted(found.get("<collection>", ())),
           rows=rows, blocked_events=len(blocked))
(REV2 / "q2_classify_reads.json").write_text(json.dumps(res, indent=1, default=list) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=list)[:6000])
for r in rows:
    print(r["in_a1_list"], r["test_id"], r["n_observed"], r["reads_eng1_changed_file"], r["observed_source_reads"][:4], r["outcome"])
