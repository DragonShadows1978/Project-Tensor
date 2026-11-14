#!/usr/bin/env python3
"""ENG1-REV2 Q2 static complement (my own grep, AST-scoped): in every regression test file, list the test functions
whose body names a by-path source-reading helper or reads a repo path directly. Names searched: verify_sources,
load_runtime, preserved, current_fingerprint, compatible, fingerprint, read_text, read_bytes, sha, sha256, open,
plus string literals naming tensor_cuda/src, tensor_cuda/include, scripts/ or .cu/.cpp/.cuh/.sh files.
It is a superset screen (receipt reads also match); q2_classify_reads.json is the dynamic discriminator.
Prior art: Python ast (PSF). Ours: the screen only."""
import ast, json
from pathlib import Path
REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
files = json.loads((ROOT / "artifacts/eng1/tool/regression_files.json").read_text())
NAMES = {"verify_sources", "load_runtime", "preserved", "current_fingerprint", "compatible", "fingerprint",
         "read_text", "read_bytes", "sha", "sha256", "open"}
SRC = ("tensor_cuda/src", "tensor_cuda/include", "scripts/", ".cu", ".cpp", ".cuh", ".sh")
a1 = {r["test_id"] for r in json.loads((ROOT / "artifacts/eng1/tool/source_reading_tests.json").read_text())}
dyn = json.loads((REV2 / "q2_classify_reads.json").read_text())
dyn_ids = {r["test_id"] for r in dyn["rows"] if r["n_observed"]}
out = []
for f in files:
    tree = ast.parse((ROOT / f).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            hits = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Attribute) and n.attr in NAMES:
                    hits.add(n.attr)
                elif isinstance(n, ast.Name) and n.id in NAMES:
                    hits.add(n.id)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str) and any(s in n.value for s in SRC) \
                        and len(n.value) < 200:
                    hits.add("literal:" + n.value[:60])
            if hits:
                tid = f + "::" + node.name
                out.append(dict(test_id=tid, hits=sorted(hits), in_a1=tid in a1, observed_dynamic=tid in dyn_ids))
res = dict(screened=len(out), not_in_a1_and_not_observed=[r for r in out if not r["in_a1"] and not r["observed_dynamic"]],
           rows=out)
(REV2 / "q2_static_scan.json").write_text(json.dumps(res, indent=1) + "\n")
print("screened", len(out), "unexplained", len(res["not_in_a1_and_not_observed"]))
for r in res["not_in_a1_and_not_observed"]:
    print(r["test_id"], r["hits"])
