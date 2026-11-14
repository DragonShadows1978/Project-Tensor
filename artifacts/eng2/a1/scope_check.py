#!/usr/bin/env python3
"""ENG2-A1 A-G3 scope (seat form, no git): compare a1/worktree_before_a1.json (taken before any A1 edit) with an
after-map (artifacts/eng2/snapshot.py, unchanged). Allowed by orders/ENG2_A1_REV1_FIXES.md:
  changed: __init__.py (tensor, make, _np_dtype only), functional.py (_causal_mask only), test_eng2.py (byte-prefix
           additions only), tool/evidence.py (matches/literal_brackets + the pin branch), tool/score_gpu.py (the
           cross-arm keys), artifacts/eng2/dtype_census.py, and the staging files that re-sealing rewrites
           (tool/staging_seal.json, tool/arms/**, tool/ids_*.txt, tool/collect_*.json, tool/baseline_expected_failures.json,
           tool/gpu_lines.json, tool/regression_files.json, tool/source_reading_tests.json, tool/STAGED_GPU_COMMANDS.sh);
  added:   artifacts/eng2/a1/**, tool/gpu_runs/dry_2/**, tool/gpu_runs/audit_2/**.
  removed: nothing.
Product and tool hunks are located by AST spans (difflib over the before copy in a1/before/).
Usage: scope_check.py AFTER.json   -> a1/AG3_SCOPE.json
Prior art: manifest diffing (BSD mtree(8); unverified -- lead to check); difflib (Python stdlib). No new algorithm.
"""
import ast
import difflib
import json
from pathlib import Path
import sys

A1 = Path(__file__).resolve().parent
ROOT = A1.parents[2]
before = json.loads((A1 / "worktree_before_a1.json").read_text())["files"]
after = json.loads(Path(sys.argv[1]).read_text())["files"]
T = "artifacts/eng2/tool/"
CODE = {"tensor_cuda/tensor_cuda/__init__.py": {"_np_dtype", "tensor", "make"},
        "tensor_cuda/tensor_cuda/functional.py": {"_causal_mask"},
        T + "evidence.py": {"literal_brackets", "matches", "regression_reasons", "pin_failure_reasons", "PIN_EXCEPTION"},
        T + "score_gpu.py": {"CROSS_ROWS", "keyed_rows", "main"},
        "artifacts/eng2/dtype_census.py": None,
        "tensor_cuda/tests/test_eng2.py": None}
STAGING = {T + n for n in ("staging_seal.json", "baseline_expected_failures.json", "gpu_lines.json", "regression_files.json",
                           "source_reading_tests.json", "STAGED_GPU_COMMANDS.sh")}
staging_ok = lambda k: k in STAGING or k.startswith((T + "arms/", T + "ids_", T + "collect_"))
ADDED_OK = ("artifacts/eng2/a1/", T + "gpu_runs/dry_2/", T + "gpu_runs/audit_2/")
changed = sorted(k for k in before if k in after and before[k] != after[k])
removed = sorted(k for k in before if k not in after)
added = sorted(k for k in after if k not in before)
bad_changed = [k for k in changed if k not in CODE and not staging_ok(k) and not k.startswith("artifacts/eng2/a1/")]
bad_added = [k for k in added if not k.startswith(ADDED_OK)]


def spans(src):
    """name -> (first, last) line of every def/class/top-level assignment, nested defs included."""
    out = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(n.name, (n.lineno, n.end_lineno))
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.col_offset == 0:
            out.setdefault(n.targets[0].id, (n.lineno, n.end_lineno))
    return out


def innermost(sp, line):
    hits = [(b - a, name) for name, (a, b) in sp.items() if a <= line <= b]
    return min(hits)[1] if hits else None


hunks = {}
for path, allowed in CODE.items():
    old = (A1 / "before" / path.replace("/", "__")).read_text()
    new = (ROOT / path).read_text()
    if allowed is None:
        hunks[path] = dict(old_is_prefix=new.startswith(old)) if path.endswith("test_eng2.py") else dict(note="completion")
        continue
    so, sn = spans(old), spans(new)
    owners, unowned = set(), []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old.splitlines(), new.splitlines()).get_opcodes():
        if tag == "equal":
            continue
        # Repair 1 (A-G3, L56(b)): a changed line outside every span is allowed only if it is blank or a comment
        # (the separators and comment headers of added top-level definitions); its text is listed in the receipt.
        for lines_, sp, a, b in ((old.splitlines(), so, i1, i2), (new.splitlines(), sn, j1, j2)):
            for ln in range(a + 1, b + 1):
                owner = innermost(sp, ln)
                text = lines_[ln - 1].strip()
                if owner is None and (not text or text.startswith("#")):
                    unowned.append(text)
                    continue
                owners.add(owner)
    hunks[path] = dict(owners=sorted(map(str, owners)), within_allowed=owners <= allowed, unowned_comment_or_blank=unowned)
arm_changes = {}
for k in changed:
    if k.startswith(T + "arms/"):
        arm, rest = k[len(T + "arms/"):].split("/", 1)
        arm_changes.setdefault(rest, []).append(arm)
staging_changed = sorted(k for k in changed if staging_ok(k) and not k.startswith(T + "arms/"))
staging_unchanged = sorted(k for k in before if staging_ok(k) and not k.startswith(T + "arms/") and k not in changed)
ok = (not bad_changed and not bad_added and not removed and all(h.get("within_allowed", True) for h in hunks.values())
      and hunks["tensor_cuda/tests/test_eng2.py"]["old_is_prefix"])
rec = dict(before_files=len(before), after_files=len(after), changed_outside_a1=[k for k in changed if not k.startswith("artifacts/eng2/a1/") and not k.startswith(T + "arms/")],
           arm_file_changes={k: sorted(v) for k, v in arm_changes.items()}, staging_changed=staging_changed,
           staging_unchanged=staging_unchanged, removed=removed,
           added_by_area={p: sum(k.startswith(p) for k in added) for p in ADDED_OK}, hunks=hunks,
           violations=dict(changed=bad_changed, added=bad_added, removed=removed), passed=ok)
(A1 / "AG3_SCOPE.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec, indent=1))
sys.exit(0 if ok else 1)
