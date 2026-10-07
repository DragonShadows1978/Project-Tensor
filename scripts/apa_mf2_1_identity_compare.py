"""APA-MF2.1 G3(b) comparison (v2: v1 counted +-inf reduce outputs of rows that contain +-inf as non-finite failures; compare.json kept as that receipt) of the identity arms (scripts/apa_mf2_1_identity.py).

  before vs after       full MF2 -> MF2.1 change; every key byte-equal EXCEPT
                        real/*/apa_p0.15 (item 3 changes the selection by design) and
                        design/* (rows beyond the old sentinel; expected to change)
  engine_iso vs after   the ENGINE (sentinel) change alone, same tree package: every key
                        byte-equal except design/*  (this is the p=0.15 identity arm)
  before vs engine_iso  the PACKAGE change alone, same MF2 engine: equal except apa_p0.15
Also: every after-arm tensor finite (except design rows of all -inf by construction).
Prior art: none (mine).
"""
import json, sys, collections
D = sys.argv[1]
A = {a: json.load(open(f"{D}/{a}.json")) for a in ("before", "after", "engine_iso")}
for a, d in A.items():
    print(a, d["meta"]["engine_sha256"][:12], d["meta"]["functional_sha256"][:12], d["meta"]["quant_sha256"][:12])
def cat(k):
    c = k.split("/")[0]
    if c == "real":
        c += "/" + k.split("/")[-1] if "sdpa_grad" not in k else "/sdpa_grad"
    return c
res = {}
for x, y, expect_diff in (("before", "after", lambda k: k.startswith("design") or k.endswith("apa_p0.15")),
                          ("engine_iso", "after", lambda k: k.startswith("design")),
                          ("before", "engine_iso", lambda k: k.endswith("apa_p0.15"))):
    rx, ry = A[x]["rec"], A[y]["rec"]
    assert rx.keys() == ry.keys()
    tab = collections.defaultdict(lambda: {"n": 0, "equal": 0})
    bad = []
    for k in rx:
        c = cat(k)
        tab[c]["n"] += 1
        eq = rx[k]["sha256"] == ry[k]["sha256"]
        tab[c]["equal"] += eq
        if not eq and not expect_diff(k):
            bad.append(k)
    res[f"{x}_vs_{y}"] = {"by_class": dict(tab), "unexpected_differences": bad}
    print(f"{x} vs {y}:", json.dumps(dict(tab)), "UNEXPECTED:", len(bad), bad[:5])
# v2: reduce rows built with +-inf entries ("mixed") legitimately reduce to +-inf (max of a
# row holding +inf IS +inf); finiteness is asserted for every other output.
nonfinite = [k for k, v in A["after"]["rec"].items() if not v["finite"] and not k.startswith("design")
             and not (k.startswith("reduce/") and "/mixed/" in k)]
res["mixed_reduce_nonfinite_by_construction"] = sum(1 for k, v in A["after"]["rec"].items()
                                                    if k.startswith("reduce/") and "/mixed/" in k and not v["finite"])
res["after_nonfinite_outside_design"] = nonfinite
res["design_rows"] = {k: {a: A[a]["rec"][k]["sha256"][:16] for a in A} for k in A["after"]["rec"] if k.startswith("design")}
res["G3b_pass"] = all(not v["unexpected_differences"] for k, v in res.items() if k.endswith(("_after", "_iso"))) and not nonfinite
print("after nonfinite outside design:", nonfinite)
print("G3b_pass", res["G3b_pass"])
json.dump(res, open(f"{D}/compare_v2.json", "x"), indent=1)
