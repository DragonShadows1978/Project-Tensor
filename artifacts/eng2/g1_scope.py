#!/usr/bin/env python3
"""ENG2 G1 (seat form, no git): compare worktree_before.json with an after-map. Allowed to differ: the two product
files, the new test file, the built module installed in the package, tensor_cuda/build-eng2/**, artifacts/eng2/**,
logs/eng2_*. Everything else must be byte-identical and nothing may be removed.
Prior art: manifest diffing (BSD mtree(8); unverified -- lead to check). Ours: the allow-list (the order's writable list).
Usage: g1_scope.py AFTER.json OUT.json
"""
import json, sys
from pathlib import Path
before = json.loads(Path(Path(__file__).parent / "worktree_before.json").read_text())["files"]
after = json.loads(Path(sys.argv[1]).read_text())["files"]
MODIFY = {"tensor_cuda/tensor_cuda/__init__.py", "tensor_cuda/tensor_cuda/nn.py"}
NEW = {"tensor_cuda/tests/test_eng2.py", "tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so"}
PREFIX = ("tensor_cuda/build-eng2/", "artifacts/eng2/", "logs/eng2_")
changed = sorted(k for k in before if k in after and before[k] != after[k])
removed = sorted(k for k in before if k not in after)
added = sorted(k for k in after if k not in before)
bad_changed = [k for k in changed if k not in MODIFY and not k.startswith(PREFIX)]
bad_added = [k for k in added if k not in NEW and not k.startswith(PREFIX)]
added_by_area = {}
for k in added:
    area = next((p for p in PREFIX if k.startswith(p)), k)
    added_by_area[area] = added_by_area.get(area, 0) + 1
rec = dict(before_files=len(before), after_files=len(after), changed=changed, removed=removed,
           added_outside_prefixes=[k for k in added if not k.startswith(PREFIX)], added_by_area=added_by_area,
           identical_outside_allowed=len(before) - len(changed) - len(removed),
           violations=dict(changed=bad_changed, added=bad_added, removed=removed),
           passed=not bad_changed and not bad_added and not removed and set(changed) >= MODIFY)
Path(sys.argv[2]).write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps({k: rec[k] for k in ("before_files", "after_files", "changed", "removed", "added_outside_prefixes",
                                     "added_by_area", "violations", "passed")}, indent=1))
