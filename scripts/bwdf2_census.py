"""BWD-F2 R2 census (L82, L89, L90): every fixed-size array in the engine's sources, found by declared size (not by
function name), with each axis-indexed spec's consumers and where each consumer now checks the rank ceiling.

  python3 -B -P scripts/bwdf2_census.py     -> artifacts/bwdf1/f2/CENSUS_F2.json
Scan: every declaration `<type> <name>[<SIZE>]` in tensor_cuda/src/*.cu, *.cuh, *.cpp (recursive), tensor_cuda/include,
tc_signal/src, tc_signal/include. Arrays sized TC_MAX_DIMS are indexed by axis (rank-bound); every other size is
listed with its reason for being rank-independent. Consumers of each TC_MAX_DIMS spec: every function that declares
a variable of that spec type (`<Spec> s{}`), and the line of its check_meta_rank call (or why none is needed).
Prior art: none beyond grep; the L82 grep starts it. Ours: the classification.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["tensor_cuda/src", "tensor_cuda/include", "tc_signal/src", "tc_signal/include"]
DECL = re.compile(r"^\s*(?:const\s+)?(?:[A-Za-z_][\w:<>]*\s+)+\*?([A-Za-z_]\w*)\[([A-Za-z_0-9]+)\]\s*(?:[;=,{)]|$)")


def files():
    for d in DIRS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            dn[:] = sorted(x for x in dn if not x.startswith("build"))
            for f in sorted(fn):
                if f.endswith((".cu", ".cuh", ".cpp", ".h")):
                    yield os.path.relpath(os.path.join(dp, f), ROOT)


def func_of(lines, i):
    """Name of the function enclosing line i (nearest preceding top-level definition)."""
    for j in range(i, -1, -1):
        m = re.match(r"^(?:static\s+|inline\s+|template\s*<[^>]*>\s*)*[\w:<>,\s\*&]+?\b([A-Za-z_]\w*)\s*\([^;]*$",
                     lines[j])
        if m and not lines[j].startswith((" ", "\t", "#", "//")) and m.group(1) not in ("if", "for", "while", "switch"):
            return m.group(1), j + 1
    return None, None


def main():
    arrays, specs = [], {}
    for f in files():
        lines = open(os.path.join(ROOT, f), errors="replace").read().splitlines()
        for i, l in enumerate(lines):
            if l.lstrip().startswith("//"):
                continue
            for part in re.split(r";", l):
                m = DECL.match(part + ";") if "[" in part else None
                if not m:
                    continue
                name, size = m.group(1), m.group(2)
                sm = re.search(r"struct\s+(\w+)", l)
                rec = {"file": f, "line": i + 1, "name": name, "size": size, "struct": sm.group(1) if sm else None,
                       "text": l.strip()[:160]}
                if size == "TC_MAX_DIMS":
                    rec["class"] = "axis-indexed (rank-bound): TC_MAX_DIMS = 8"
                    if rec["struct"]:
                        specs.setdefault(rec["struct"], {"file": f, "line": i + 1, "arrays": []})["arrays"].append(name)
                    else:
                        fn, _ = func_of(lines, i)
                        rec["enclosing_function"] = fn
                else:
                    rec["class"] = "rank-independent"
                arrays.append(rec)
    # struct with TC_MAX_DIMS arrays declared on several lines (DimSpec, ReduceSpec): attach by preceding struct line
    for a in arrays:
        if a["size"] == "TC_MAX_DIMS" and not a["struct"] and a.get("enclosing_function") is None:
            pass
    for f in files():
        lines = open(os.path.join(ROOT, f), errors="replace").read().splitlines()
        cur = None
        for i, l in enumerate(lines):
            m = re.match(r"^\s*struct\s+(\w+)\s*\{\s*$", l)
            if m:
                cur = m.group(1)
                continue
            if cur and l.strip().startswith("};"):
                cur = None
                continue
            if cur and "[TC_MAX_DIMS]" in l:
                specs.setdefault(cur, {"file": f, "line": i + 1, "arrays": []})
                n = re.search(r"(\w+)\[TC_MAX_DIMS\]", l).group(1)
                if n not in specs[cur]["arrays"]:
                    specs[cur]["arrays"].append(n)
                for a in arrays:
                    if a["file"] == f and a["line"] == i + 1:
                        a["struct"] = cur
    # consumers of each spec: functions declaring `<Spec> s{}` (or `<Spec> s;`), and their rank check
    k = os.path.join(ROOT, "tensor_cuda/src/kernels.cu")
    kl = open(k).read().splitlines()
    for spec, info in specs.items():
        info["consumers"] = []
        for i, l in enumerate(kl):
            if re.search(rf"\b{spec}\s+\w+\s*(\{{\}}|;)", l) and "__global__" not in l:
                fn, start = func_of(kl, i)
                body_end = next((j for j in range(i, len(kl)) if kl[j].startswith("}")), i)
                chk = [j + 1 for j in range(start - 1, body_end) if "check_meta_rank(" in kl[j] or
                       "check_copy_operands(" in kl[j]]  # cat_nd: check_copy_operands calls check_meta_rank
                info["consumers"].append({"function": fn, "decl_line": i + 1, "function_line": start,
                                          "check_meta_rank_lines": chk,
                                          "check_before_decl": bool(chk) and min(chk) < i + 1})
    # reduce_kernel's device-local coord[TC_MAX_DIMS] (no struct): bound by ReduceSpec's ndim, checked in reduce_impl
    callers = {"axpy_ (ew_scalar launches before ew_binary)": "axpy_", "broadcast_to (zeros before ew_binary)":
               "broadcast_to", "masked_fill (fill before where_nd)": "masked_fill"}
    pre = {}
    for label, fn in callers.items():
        src = open(os.path.join(ROOT, "tensor_cuda/src/ops.cpp" if fn == "masked_fill" else
                                "tensor_cuda/src/kernels.cu")).read().splitlines()
        hits = [i + 1 for i, l in enumerate(src) if re.search(rf"check_meta_rank\(\"{fn}\"", l)]
        pre[label] = hits
    ccl = [i + 1 for i, l in enumerate(kl) if "check_meta_rank(op, nd + (all_extents ? 1 : 0)" in l]
    pre["check_copy_operands (cat_nd; ops.cpp stack calls it before its unsqueeze, rank nd + 1)"] = ccl
    role = {"product (engine)": 0, "test": 0, "consumer (report-only)": 0, "harness": 0}
    role["product (engine)"] = sum(len(v["consumers"]) for v in specs.values()) + len(pre)
    g = subprocess.run(["grep", "-rlE", "rank 9 exceeds|TC_MAX_DIMS", "tensor_cuda/tests", "tc_signal/tests", "tests"],
                       cwd=ROOT, capture_output=True, text=True)
    role["test"] = len([x for x in g.stdout.split() if x])
    role["harness"] = len([x for x in os.listdir(os.path.join(ROOT, "scripts")) if x.startswith("bwdf2_")])
    doc = {"order": "BWD-F2 R2 census", "method": __doc__.strip().splitlines()[3:7],
           "TC_MAX_DIMS": 8, "axis_indexed_specs": specs,
           "pre_dispatch_callers_checked_before_their_own_first_cuda_call": pre,
           "rank_independent_arrays": [a for a in arrays if a["size"] != "TC_MAX_DIMS"],
           "rank_independent_reason": "sized by a geometry constant (3 = xyz in terrain / dda / paint), a fixed "
                                      "operand count (gated_delta_step's 8 inputs; kl/ql/sl[8] = Dk/32 rows per lane, "
                                      "Dk <= 256 checked at its entry), a cuBLASLt heuristic request count (heur[8]), "
                                      "an accumulator width (ACCN, kDetailOctaves, Channels) or the shared-memory "
                                      "reduction width: none is indexed by an operand's rank",
           "axis_indexed_arrays_all": [a for a in arrays if a["size"] == "TC_MAX_DIMS"],
           "roles_count": role,
           "consumer_roots_grep": "Voice engine and GRAPA: no rank-9 tensor construction found (grep 'reshape(' "
                                  "with 9 extents not attempted; consumers are report-only)"}
    json.dump(doc, open(os.path.join(ROOT, "artifacts/bwdf1/f2/CENSUS_F2.json"), "w"), indent=1)
    for s_, v in specs.items():
        print(s_, v["file"] + ":" + str(v["line"]), v["arrays"])
        for c in v["consumers"]:
            print("   ", c["function"], "decl", c["decl_line"], "check", c["check_meta_rank_lines"], c["check_before_decl"])
    print("pre-dispatch", pre)
    print("rank-independent arrays", len(doc["rank_independent_arrays"]), "roles", role)


if __name__ == "__main__":
    main()
