#!/usr/bin/env python3
"""ENG2 P303 caller census: every call of tensor/zeros/ones/randn/rand/empty/parameter that passes a `dtype` keyword,
in the named roots, by Python AST (multi-line calls included). Receivers that are clearly other libraries (np, numpy,
torch, cp, cupy, jnp, tg) are listed separately and not classified.

Usage: caller_census.py OUT.json ROOT [ROOT ...]
Skips directories named WorkTree, Fork, artifacts, .git, node_modules, __pycache__, DELETE_ME. Never follows symlinks.
Classification of the dtype expression:
  supported   - a string constant in {float32, float16, bfloat16, int64, bool, uint8}
  unsupported - any other string constant (the class P303 says does not exist)
  numpy-object- an attribute like np.float32 (a non-string; refused by the ENG2 rule, so listed for the lead)
  expression  - anything else (a variable, attribute or call); listed with its source text for manual tracing
Prior art: Python ast module (PSF); no prior art known to me for the classification, which is ours.
"""
import ast
import json
import os
from pathlib import Path
import sys

NAMES = {"tensor", "zeros", "ones", "randn", "rand", "empty", "parameter"}
OTHER = {"np", "numpy", "torch", "cp", "cupy", "jnp", "tg", "jax", "onp"}
SUPPORTED = {"float32", "float16", "bfloat16", "int64", "bool", "uint8"}
SKIP = {"WorkTree", "Fork", "artifacts", ".git", "node_modules", "__pycache__", "DELETE_ME", ".venv", "venv"}


def receiver(func):
    if isinstance(func, ast.Attribute):
        return ast.unparse(func.value), func.attr
    if isinstance(func, ast.Name):
        return "", func.id
    return None, None


def classify(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "supported" if node.value in SUPPORTED else "unsupported"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
        return "numpy-object"
    return "expression"


def main(out, roots):
    rows, other, errors, nfiles = [], 0, [], 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
            for f in sorted(filenames):
                if not f.endswith(".py"):
                    continue
                p = Path(dirpath) / f
                if p.is_symlink():
                    continue
                nfiles += 1
                try:
                    tree = ast.parse(p.read_text(errors="replace"))
                except SyntaxError as exc:
                    errors.append(dict(file=str(p), error="SyntaxError: " + str(exc)))
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    recv, name = receiver(node.func)
                    if name not in NAMES:
                        continue
                    kws = [k for k in node.keywords if k.arg == "dtype"]
                    if not kws:
                        continue
                    if recv in OTHER or (recv or "").split(".")[0] in OTHER:
                        other += 1
                        continue
                    if name == "parameter" and recv not in ("", "nn", "tc.nn", "tensor_cuda.nn"):
                        continue
                    rows.append(dict(file=str(p), line=node.lineno, call=(recv + "." if recv else "") + name,
                                     dtype=ast.unparse(kws[0].value), cls=classify(kws[0].value)))
    summary = {}
    for r in rows:
        summary[r["cls"]] = summary.get(r["cls"], 0) + 1
    Path(out).write_text(json.dumps(dict(roots=roots, files_parsed=nfiles, other_library_calls=other,
                                         summary=summary, parse_errors=errors, rows=rows), indent=1) + "\n")
    print(json.dumps(dict(files_parsed=nfiles, summary=summary, other_library_calls=other, parse_errors=len(errors))))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
