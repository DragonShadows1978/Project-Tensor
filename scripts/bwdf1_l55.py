"""BWD-F1 L55 (order) and Amendment 1 L55: witnesses of each class, from the real source text of Build B and Build E.

  python3 -B -P scripts/bwdf1_l55.py     -> artifacts/bwdf1/L55.json

Z2: the exact text of contiguous_strides, broadcast_shape and bcast_strides (and Build E's bcast_shape_text) is cut
from each build's kernels.cu (/tmp/bwdf1_src_B, /tmp/bwdf1_src_E) by brace matching, compiled with the host compiler,
and run on one witness per class (0 against 1, 0 against an extent above 1, equal extents, 1 against an extent above
1): the output shape, each operand's broadcast strides, the output strides, and the first and last element offset
the binary kernel reads from each operand (the kernel's own index loop, binary_kernel in kernels.cu, evaluated here).
C2: for one witness per term (rank, non-cat extent, dtype, device, dim range), Build B's cat_nd index arithmetic
(kernels.cu:1513-1535 at fbd335ae7) evaluated in Python on the shapes (source reasoning, nothing run on a device): the
output shape B allocates and, per operand, the largest output element its copy writes against the output's element
count; Build E's refusal is the G1 row's message (artifacts/bwdf1/g1/records_E/g1_copy_ops.json).
Prior art: none needed beyond NumPy's broadcasting rule (the reference); ours: the extraction harness.
"""
import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def cut(src, sig):
    i = src.index(sig)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        depth += {"{": 1, "}": -1}.get(src[k], 0)
        if depth == 0:
            return src[i:k + 1]
    raise ValueError(sig)


HARNESS = r'''
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <initializer_list>
using Shape = std::vector<int64_t>;
%FUNCS%
static std::string txt(const Shape& s) { std::string t = "["; for (size_t i = 0; i < s.size(); ++i) { if (i) t += ", "; t += std::to_string(s[i]); } return t + "]"; }
static void w(const char* label, Shape a, Shape b) {
  try {
    Shape o = broadcast_shape(a, b);
    int n = (int)o.size();
    std::vector<int64_t> as(n), bs(n);
    bcast_strides(a, o, as.data()); bcast_strides(b, o, bs.data());
    Shape os = contiguous_strides(o);
    int64_t numel = 1; for (auto e : o) numel *= e;
    // binary_kernel's loop: offsets read for the first and last output element
    auto offs = [&](int64_t idx, int64_t& ao, int64_t& bo) { int64_t rem = idx; ao = bo = 0;
      for (int d = 0; d < n; ++d) { int64_t c = rem / os[d]; rem -= c * os[d]; ao += c * as[d]; bo += c * bs[d]; } };
    int64_t a0 = 0, b0 = 0, a1 = 0, b1 = 0;
    if (numel > 0) { offs(0, a0, b0); offs(numel - 1, a1, b1); }
    std::printf("{\"class\": \"%s\", \"a\": \"%s\", \"b\": \"%s\", \"out\": \"%s\", \"out_numel\": %lld, \"a_strides\": \"%s\", \"b_strides\": \"%s\", \"out_strides\": \"%s\", \"a_offsets_first_last\": [%lld, %lld], \"b_offsets_first_last\": [%lld, %lld], \"launch\": %s}\n",
                label, txt(a).c_str(), txt(b).c_str(), txt(o).c_str(), (long long)numel, txt(Shape(as.begin(), as.end())).c_str(),
                txt(Shape(bs.begin(), bs.end())).c_str(), txt(os).c_str(), (long long)a0, (long long)a1, (long long)b0, (long long)b1,
                numel > 0 ? "true" : "false");
  } catch (const std::exception& e) {
    std::printf("{\"class\": \"%s\", \"a\": \"%s\", \"b\": \"%s\", \"refused\": \"%s\"}\n", label, txt(a).c_str(), txt(b).c_str(), e.what());
  }
}
int main() {
  w("0 against 1", {4, 0}, {4, 1});
  w("0 against an extent above 1", {4, 0}, {4, 3});
  w("equal extents", {4, 3}, {4, 3});
  w("1 against an extent above 1", {4, 1}, {4, 3});
  return 0;
}
'''


def z2(arm):
    src = (pathlib.Path(f"/tmp/bwdf1_src_{arm}") / "tensor_cuda/src/kernels.cu").read_text()
    funcs = [cut(src, "Shape contiguous_strides(const Shape& shape)")]
    if "std::string bcast_shape_text(" in src:
        funcs.append(cut(src, "std::string bcast_shape_text(const Shape& s)"))
    funcs += [cut(src, "Shape broadcast_shape(const Shape& a, const Shape& b)"),
              cut(src, "void bcast_strides(const Shape& in, const Shape& out, int64_t* dst)")]
    code = HARNESS.replace("%FUNCS%", "\n".join(funcs)).replace("__builtin_mul_overflow", "__builtin_mul_overflow")
    with tempfile.TemporaryDirectory(dir="/tmp/bwdf1_tmp") as d:
        p = pathlib.Path(d)
        (p / "l55.cpp").write_text(code)
        r = subprocess.run([shutil.which("c++") or "g++", "-std=c++17", "-O1", "-o", str(p / "l55"), str(p / "l55.cpp")],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        out = subprocess.run([str(p / "l55")], capture_output=True, text=True, timeout=60)
    return [json.loads(l) for l in out.stdout.splitlines()], [f.splitlines()[0] for f in funcs]


def b_cat(shapes, dim, itemsizes=None):
    """Build B's cat_nd on the shapes (kernels.cu:1513-1535 at fbd335ae7): os = arrs[0].shape, os[dim] = sum."""
    nd = len(shapes[0])
    if dim < 0:
        dim += nd
    if not 0 <= dim < nd:
        return {"b": f"os[{dim}] on a rank-{nd} shape vector: out-of-range std::vector access on the host (undefined)"}
    os_ = list(shapes[0])
    os_[dim] = sum(s[dim] if dim < len(s) else 0 for s in shapes)
    big = np.cumprod([1] + os_[::-1][:-1])[::-1].tolist() if os_ else []
    out_numel = int(np.prod(os_))
    rows, running = [], 0
    for i, s in enumerate(shapes):
        n = int(np.prod(s))
        last = None
        if n:
            coords = np.unravel_index(n - 1, s)
            last = int(sum((int(c) + (running if d == dim else 0)) * big[d] for d, c in enumerate(coords[:nd])))
        rows.append({"operand": i, "shape": list(s), "offset_along_dim": running, "elements": n,
                     "largest_output_element_written": last,
                     "past_output_end": None if last is None else last >= out_numel})
        running += s[dim] if dim < len(s) else 0
    return {"b_output_shape": os_, "b_output_elements": out_numel, "copies": rows}


def main():
    res = {"z2": {}, "c2": []}
    for arm in ("B", "E"):
        rows, sigs = z2(arm)
        res["z2"][arm] = {"functions_cut": sigs, "witnesses": rows}
    g1 = json.loads((ROOT / "artifacts/bwdf1/g1/records_E/g1_copy_ops.json").read_text())["results"]
    for term, rid, shapes, dim, note in (
            ("non-cat extent (Voice: zero first)", "cat-extent-voice-zero-first", [(1, 512, 0), (1, 1, 1)], 1, ""),
            ("non-cat extent (Voice: later one frame longer)", "cat-extent-voice-longer-later",
             [(1, 3, 5), (1, 2, 6), (1, 4, 6)], 1, ""),
            ("rank", "cat-rank", [(2, 3), (2, 3, 1)], 0,
             "B builds operand 1's copy spec with nd = 2 from operand 0: its third extent is ignored"),
            ("dtype", "cat-dtype", [(2, 3), (2, 3)], 0,
             "B dispatches on operand 0's dtype (float32) and reads the float16 operand as 4-byte elements: twice its "
             "bytes, a read past its end"),
            ("device", "cat-device", [(0, 3), (0, 3)], 0,
             "B allocates on operand 0's device and launches the copy with the other operand's pointer whatever its "
             "device (here both are empty, so no copy runs; with elements the kernel reads a host or foreign pointer)"),
            ("dim range", "cat-dim-high", [(2, 3), (2, 3)], 2, "")):
        row = {"term": term, "witness_shapes": [list(s) for s in shapes], "dim": dim, **b_cat(shapes, dim)}
        if note:
            row["b_note"] = note
        row["e_refusal"] = g1[rid].get("message")
        row["e_cuda_calls"] = g1[rid].get("calls")
        res["c2"].append(row)
    (ROOT / "artifacts/bwdf1/L55.json").write_text(json.dumps(res, indent=1))
    for arm in ("B", "E"):
        for w in res["z2"][arm]["witnesses"]:
            print(arm, json.dumps(w))
    for r in res["c2"]:
        print("C2", r["term"], "| B:", r.get("b_output_shape", r.get("b")), [c.get("past_output_end") for c in
                                                                           r.get("copies", [])], "| E:", r["e_refusal"])


if __name__ == "__main__":
    main()
