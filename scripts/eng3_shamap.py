"""ENG3 helper: sha256 map of the engine sources (tensor_cuda/src, tensor_cuda/include, tensor_cuda/tensor_cuda; .so
and __pycache__ excluded) plus the sha256 of a module file, written as one build receipt.

Usage: python3 -B -P scripts/eng3_shamap.py <out.json> <module.so> <label>

Prior art: no prior art known to me (a plain content-hash manifest, as in ENG2's build receipts).
"""
import hashlib
import json
import os
import sys

ROOTS = ("tensor_cuda/src", "tensor_cuda/include", "tensor_cuda/tensor_cuda")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out, module, label = sys.argv[1], sys.argv[2], sys.argv[3]
    files = {}
    for root in ROOTS:
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for n in sorted(names):
                p = os.path.join(d, n)
                if n.endswith(".so") or n.endswith(".pyc") or os.path.islink(p):
                    continue
                files[p] = sha(p)
    combined = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    rec = {"label": label, "module": module, "module_sha256": sha(module), "module_bytes": os.path.getsize(module),
           "source_files": len(files), "source_map_sha256": combined, "sources": files}
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(label, rec["module_sha256"], "sources", len(files), combined)


if __name__ == "__main__":
    main()
