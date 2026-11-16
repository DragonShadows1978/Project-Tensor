"""ENG3 E1 census helper (static reading, CPU only, no engine import).

Lists every pybind11 entry of the engine module (bindings.cpp: Tensor methods, module functions, ApaInt4Workspace
members) with its line, the C++ targets its body names, and whether any target body contains a device check
(`is_cuda`). The NDArray core rows are hand-written by the seat from the L82 grep and are merged in by this script
from artifacts/eng3/core_rows.json. Output: artifacts/eng3/HOST_CENSUS.json.

Usage: python3 -B -P scripts/eng3_census.py

Prior art: no prior art known to me for this exact census; it is a grep/regex call-site inventory (the same method as
ENG2's caller_census.py in the eng2 worktree, which used Python's ast for Python call sites).
"""
import json
import os
import re

SRC = "tensor_cuda/src"
BIND = os.path.join(SRC, "bindings.cpp")
PY_INIT = "tensor_cuda/tensor_cuda/__init__.py"

# Entries whose host-tensor behaviour the seat classified by reading (see HOST_CENSUS.json "classes").
DATA = {
    "tensor": "tensor_from_numpy -> NDArray::from_host (E2)",
    "numpy": "tensor_to_numpy -> NDArray::astype(Float32) for bfloat16 (E2/E3), NDArray::to_host (E2)",
    "astype": "ops::cast -> NDArray::astype (E2/E3); same dtype returns the input (no clone)",
    "half": "ops::cast -> NDArray::astype(Float16) (E2/E3)",
    "float": "ops::cast -> NDArray::astype(Float32) (E2/E3)",
    "backward": "Tensor::backward -> NDArray::full(1.0) seed (E2), Variable::accumulate_grad -> reduce_to -> "
                "NDArray::clone when shapes match (E2); a second accumulation -> ew_binary (compute kernel)",
    "empty": "bindings empty -> NDArray::empty (host: std::malloc only; no CUDA call for device='cpu')",
}
REFUSING = {
    "fill_": "kernels.cu fill_: 'fill_: CUDA arrays only' (pre-existing refusal)",
    "__dlpack__": "bindings.cpp tensor_dlpack: BufferError for a non-CUDA tensor (pre-existing)",
    "__dlpack_device__": "bindings.cpp tensor_dlpack_device: BufferError on CPU (pre-existing)",
    "_from_dlpack_capsule": "bindings.cpp: refuses non-CUDA capsules (BufferError, pre-existing); takes no tc tensor",
}
META = {
    "shape", "ndim", "dtype", "device", "requires_grad", "grad", "data_ptr", "mark_modified", "revision", "reshape",
    "detach", "zero_grad", "is_grad_enabled", "set_grad_enabled", "synchronize", "empty_cache", "set_alloc_pooling",
    "_dlpack_release_stats", "_dlpack_raise_unraisable", "checkpoint", "bp_source_pin", "bp_wall_begin",
    "bp_wall_end", "bp_configure", "bp_begin", "bp_end", "bp_rows", "bp_kernel_4_set_variant",
    "bp_kernel_4_get_variant", "bp_kernel_2_set_variant", "bp_kernel_2_get_variant", "capacity", "valid_rows",
    "nbytes",
}


def read(p):
    with open(p) as f:
        return f.read().split("\n")


def func_bodies():
    """name -> list of (file, start_line, text) for top-level C++ definitions returning NDArray/Tensor/tuple/void."""
    out = {}
    pat = re.compile(r"^(?:static\s+)?(?:inline\s+)?(?:NDArray|Tensor|void|std::tuple<[^>]*>|py::\w+|bool|int64_t)\s+"
                     r"(?:\w+::)?(\w+)\s*\(")
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith((".cu", ".cpp", ".cuh")):
            continue
        lines = read(os.path.join(SRC, fn))
        i = 0
        while i < len(lines):
            m = pat.match(lines[i])
            if m and not lines[i].rstrip().endswith(";"):
                j = i
                while j < len(lines) and not lines[j].startswith("}"):
                    j += 1
                out.setdefault(m.group(1), []).append((os.path.join(SRC, fn), i + 1, "\n".join(lines[i:j + 1])))
                i = j
            i += 1
    return out


def bindings():
    lines = read(BIND)
    start = next(i for i, l in enumerate(lines) if l.startswith("PYBIND11_MODULE"))
    rows = []
    cls = None
    pat = re.compile(r'(m\.def|\.def|\.def_property_readonly|\.def_property)\(\s*"([^"]+)"')
    for i in range(start, len(lines)):
        l = lines[i]
        c = re.search(r'py::class_<(\w+)', l)
        if c:
            cls = c.group(1)
        for m in pat.finditer(l):
            kind = "module" if m.group(1) == "m.def" else ("method:" + (cls or "?"))
            body = "\n".join(lines[i:i + 8])
            nxt = [k for k in range(i + 1, min(i + 40, len(lines))) if pat.search(lines[k])]
            if nxt:
                body = "\n".join(lines[i:nxt[0]])
            targets = sorted(set(re.findall(r"(?:ops::|tc::|&)(\w+)\s*[,(\)]", body)))
            rows.append({"name": m.group(2), "kind": kind, "line": i + 1, "targets": targets})
    return rows


def main():
    bodies = func_bodies()
    rows = bindings()
    for r in rows:
        checks = []
        for t in r["targets"]:
            for (f, ln, text) in bodies.get(t, []):
                for k, tl in enumerate(text.split("\n")):
                    if "is_cuda" in tl:
                        checks.append(f"{f}:{ln + k}")
        r["device_checks_in_targets"] = checks
        n = r["name"]
        if n in DATA:
            r["class"], r["note"] = "E2_data_path", DATA[n]
        elif n in REFUSING:
            r["class"], r["note"] = "refuses_host_already", REFUSING[n]
        elif n in META:
            r["class"], r["note"] = "no_device_work_on_host_storage", "metadata/config; no kernel on the tensor's storage"
        else:
            r["class"] = "compute_launches_kernel"
            r["note"] = ("reaches engine kernels with the tensor's data pointer; with a host tensor they run on "
                         "malloc'd memory (HMM) unless a device check below refuses first")
    with open("artifacts/eng3/core_rows.json") as f:
        core = json.load(f)
    by_class = {}
    for r in rows:
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
    out = {
        "order": "orders/ENG3_HOST_TENSOR_CORE.md (114d6269)",
        "method": "static reading at 2478a4d: L82 grep of kernels.cu 531-760 for the core rows (hand-written); "
                  "regex inventory of bindings.cpp's PYBIND11_MODULE for the Python rows (scripts/eng3_census.py). "
                  "Evidence class: reasoning from source, not measured. device_checks_in_targets looks ONE call level deep "
                  "(the binding body's named targets); deeper refusals (e.g. the APA-SP1 host refusal that "
                  "tests/test_apa_sp1_host.py exercises) are not traced, so a compute row may already refuse a host "
                  "tensor further down. The 172 compute rows are the upper bound of E4's population.",
        "core_members": core,
        "python_entries_summary": by_class,
        "python_entries": rows,
    }
    with open("artifacts/eng3/HOST_CENSUS.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(by_class))
    print("compute entries with a device check in a target:",
          sum(1 for r in rows if r["class"] == "compute_launches_kernel" and r["device_checks_in_targets"]))


if __name__ == "__main__":
    main()
