#!/usr/bin/env python3
"""ENG2 L60 census of every dtype-string entry in tensor_cuda/tensor_cuda/ (and the C++ dtype_from_string behind the
C++ entries), run on CPU with CUDA hidden, against one arm's package.

Usage: CUDA_VISIBLE_DEVICES= dtype_census.py PACKAGE_PARENT ARM OUT.json
Every probe uses zero-byte tensors (device "cpu", or "cuda" with zero bytes, which allocates nothing and makes no
CUDA call), so no device is touched. Outcome per (entry, string):
  refused     - an exception (type and message recorded)
  mis-served  - a tensor whose dtype is not the requested one (the silent float32 fallback)
  handled     - a tensor of the requested dtype (for C++ aliases: of the dtype the alias names)
Strings: the order's four ("float64", "double", "int32", an unknown "not_a_dtype"), the C++ aliases
(dtype_from_string, kernels.cu:50-58), the six supported names, and two non-strings (np.float32, np.dtype("float16")).
Prior art: none known to me (a probe table); the entries were found by `grep -n dtype` over the package (ledger).
"""
import json
import os
from pathlib import Path
import sys

package_parent, arm, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA hidden"
sys.path.insert(0, package_parent)
import numpy as np  # noqa: E402
import tensor_cuda as tc  # noqa: E402
assert Path(tc.__file__).resolve().parent.parent == Path(package_parent).resolve()

ORDER_STRINGS = ["float64", "double", "int32", "not_a_dtype"]
ALIASES = {"float": "float32", "f32": "float32", "half": "float16", "f16": "float16", "long": "int64",
           "u8": "uint8", "byte": "uint8", "bf16": "bfloat16", "bfloat": "bfloat16"}
SUPPORTED = ["float32", "float16", "int64", "bool", "uint8", "bfloat16"]
NON_STRINGS = {"np.float32": np.float32, "np.dtype('float16')": np.dtype("float16")}


def z(device="cpu"):
    return tc.tensor(np.zeros((0,), dtype=np.float32), device=device)


ENTRIES = {
    # name: (site, callable(dtype) -> tensor)
    "tc.tensor": ("__init__.py tensor() (:48 at bc02d3a1)", lambda d: tc.tensor(np.zeros((0, 2)), device="cpu", dtype=d)),
    "tc.tensor(device=cuda,0 bytes)": ("__init__.py tensor()", lambda d: tc.tensor(np.zeros((0, 2)), device="cuda", dtype=d)),
    "tc.zeros": ("__init__.py _factory (:60)", lambda d: tc.zeros(0, 2, device="cpu", dtype=d)),
    "tc.ones": ("__init__.py _factory (:60)", lambda d: tc.ones(0, 2, device="cpu", dtype=d)),
    "tc.randn": ("__init__.py _factory (:60)", lambda d: tc.randn(0, 2, device="cpu", dtype=d)),
    "tc.rand": ("__init__.py _factory (:60)", lambda d: tc.rand(0, 2, device="cpu", dtype=d)),
    "tc.nn.parameter": ("nn.py parameter() -> tc.tensor", lambda d: tc.nn.parameter(np.zeros((0,)), device="cpu", dtype=d)),
    "tc.empty": ("__init__.py empty() -> C++ tensor_empty -> dtype_from_string", lambda d: tc.empty(0, dtype=d, device="cpu")),
    "Tensor.astype": ("bindings.cpp astype -> dtype_from_string", lambda d: z().astype(d)),
}
# C++ out_dtype entries: the parse happens while the call's arguments are evaluated, before the kernel. Only
# unsupported strings are probed (a supported string would reach the kernel with CPU operands).
OUT_DTYPE_ENTRIES = {
    "tc.int4_dequant(out_dtype)": lambda d: tc.int4_dequant(z(), z(), z(), 128, d),
    "tc.intn_dequant(out_dtype)": lambda d: tc.intn_dequant(z(), z(), z(), 4, 0, 128, d),
    "tc.kv_int4_unpack(out_dtype)": lambda d: tc.kv_int4_unpack(z(), z(), 32, 0, 0, d),
}


def probe(fn, requested):
    try:
        r = fn(requested)
    except Exception as exc:  # noqa: BLE001 -- the census records every refusal
        return dict(outcome="refused", exception=type(exc).__name__, message=str(exc)[:300])
    expected = ALIASES.get(requested, requested) if isinstance(requested, str) else None
    return dict(outcome="handled" if r.dtype == expected else "mis-served", result_dtype=r.dtype,
                result_shape=list(r.shape))


rows = []
for name, (site, fn) in ENTRIES.items():
    for s in ORDER_STRINGS + list(ALIASES) + SUPPORTED:
        rows.append(dict(entry=name, site=site, requested=s,
                         kind="order" if s in ORDER_STRINGS else "alias" if s in ALIASES else "supported", **probe(fn, s)))
    for label, obj in NON_STRINGS.items():
        rows.append(dict(entry=name, site=site, requested=label, kind="non-string", **probe(fn, obj)))
for name, fn in OUT_DTYPE_ENTRIES.items():
    for s in ORDER_STRINGS:
        rows.append(dict(entry=name, site="bindings.cpp -> dtype_from_string", requested=s, kind="order", **probe(fn, s)))
summary = {}
for r in rows:
    summary.setdefault(r["entry"], {}).setdefault(r["kind"], {}).setdefault(r["outcome"], 0)
    summary[r["entry"]][r["kind"]][r["outcome"]] += 1
out.write_text(json.dumps(dict(arm=arm, package=str(Path(tc.__file__).resolve()), rows=rows, summary=summary),
                          indent=1) + "\n")
for entry, kinds in summary.items():
    print(arm, entry, json.dumps(kinds, sort_keys=True))
