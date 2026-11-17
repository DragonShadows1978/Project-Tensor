"""ENG3 G3 comparison: per test ID, Build B vs Build E, under the registered intended-change list
(artifacts/eng3/G3_INTENDED.json); and the device-path byte check (artifacts/eng3/devpath/).

  python3 -B -P scripts/eng3_g3_compare.py outcomes <out.json> <B outcomes.json>[,<more>] <E outcomes.json>[,<more>]
  python3 -B -P scripts/eng3_g3_compare.py devpath <out.json> <ref.npz> <cand.npz>

Outcomes: several outcomes.json per build (the GPU halves) are merged; an ID present in only one build is a difference
(except test_eng3_host.py, excluded from both). Devpath: every key of the reference must exist in the candidate with
identical bytes (after L19: each side nonzero and non-constant, except the zeros/ and full/ keys, which are constant by
definition, and bool rows, which are 0/1); keys listed as waived (bool .numpy() on Bh/B, A2.2) are reported, not gated.

Prior art: no prior art known to me beyond a per-key diff.
"""
import json
import sys

import numpy as np


def load(paths):
    ids, coll = {}, {}
    fails = {}
    for p in paths.split(","):
        d = json.load(open(p))
        ids.update(d["outcomes"])
        coll.update(d["collection_errors"])
        fails.update(d.get("failures", {}))
    return ids, coll, fails


def outcomes(out, bpaths, epaths):
    reg = json.load(open("artifacts/eng3/G3_INTENDED.json"))
    pins = set(reg["intended"][0]["files"])
    b, bc, bf = load(bpaths)
    e, ec, ef = load(epaths)
    rows, regress, intended = [], [], []
    for k in sorted(set(b) | set(e)):
        ob, oe = b.get(k, "absent"), e.get(k, "absent")
        if ob == oe:
            continue
        f = k.split("::")[0]
        row = {"id": k, "B": ob, "E": oe}
        etext = ef.get(k, "")
        btext = bf.get(k, "")
        if f in pins and ob == "passed" and oe in ("failed", "error") and (
                "kernels.cu" in etext or "bindings.cpp" in etext):
            row["class"] = "S_source_pin"
            intended.append(row)
        elif ob in ("failed", "error") and oe == "passed" and any(
                s in btext for s in ("CUDA error at from_host", "CUDA error at to_host", "CUDA error at astype")):
            row["class"] = "H_host_path"
            intended.append(row)
        elif ob in ("failed", "error") and oe == "passed" and "bool" in k:
            row["class"] = "BOOL_numpy"
            intended.append(row)
        else:
            row["B_text"] = btext[-400:]
            row["E_text"] = etext[-400:]
            regress.append(row)
        rows.append(row)
    coll_diff = sorted(set(bc) ^ set(ec))
    res = {"B": bpaths, "E": epaths, "n_B": len(b), "n_E": len(e), "identical": len(set(b) & set(e)) - len(rows),
           "differences": len(rows), "intended": intended, "regressions": regress,
           "collection_errors_B": sorted(bc), "collection_errors_E": sorted(ec), "collection_error_diff": coll_diff,
           "PASS": not regress and not coll_diff}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_B", "n_E", "identical", "differences", "PASS")}),
          "intended", len(intended), "regressions", len(regress))
    for r in regress[:20]:
        print("REGRESSION", r["id"], r["B"], "->", r["E"])


def devpath(out, ref, cand):
    r, c = np.load(ref), np.load(cand)
    rows, bad = [], []
    for k in sorted(r.files):
        key = k.replace("|", "/")
        # A2.2 waives bool .numpy() only: the two keys that read a bool tensor through .numpy() (the raw-byte bool keys,
        # read through _core_to + ctypes, stay gated). v1 of this tool listed only the first key (ledgered).
        waived = key in ("bool_numpy_waived/bool", "from_host+numpy/bool")
        if k not in c.files:
            bad.append({"key": key, "why": "missing in candidate"})
            continue
        a, b = r[k], c[k]
        const_ok = key.startswith(("zeros/", "full/")) or "bool" in key
        for side, v in (("ref", a), ("cand", b)):
            if not const_ok and (v.size < 2 or not v.any() or not (v != v[0]).any()):
                bad.append({"key": key, "why": f"L19: {side} zero or constant"})
        same = a.shape == b.shape and bool((a == b).all())
        rows.append({"key": key, "bytes": int(a.size), "identical": same, "waived": waived})
        if not same and not waived:
            bad.append({"key": key, "why": "bytes differ", "n_diff": int((a != b).sum()) if a.shape == b.shape else -1})
    res = {"ref": ref, "cand": cand, "keys": len(rows), "identical": sum(x["identical"] for x in rows),
           "waived": [x for x in rows if x["waived"]], "failures": bad, "rows": rows, "PASS": not bad}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("keys", "identical", "PASS")}), "failures", bad[:10],
          "waived", [(x["key"], x["identical"]) for x in res["waived"]])


if __name__ == "__main__":
    if sys.argv[1] == "outcomes":
        outcomes(*sys.argv[2:5])
    else:
        devpath(*sys.argv[2:5])
