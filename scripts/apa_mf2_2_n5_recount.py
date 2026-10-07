"""APA-MF2.2 N5 erratum: recount the MF2.1 identity records (artifacts/apa_mf2_1/identity).
Counts, per comparison: total keys, equal, differing, the 3 design keys, non-design equal, and
non-design keys whose outputs are finite on both arms (strict-finite). Mine."""
import json
D = "artifacts/apa_mf2_1/identity/"
arms = {a: json.load(open(D + a + ".json"))["rec"] for a in ("before", "after", "engine_iso")}
cmp2 = json.load(open(D + "compare_v2.json"))
design = set()
for pair in ("before_vs_after", "engine_iso_vs_after"):
    pass
out = {}
for x, y in (("before", "after"), ("engine_iso", "after"), ("before", "engine_iso")):
    a, b = arms[x], arms[y]
    assert a.keys() == b.keys()
    eq = [k for k in a if a[k]["sha256"] == b[k]["sha256"]]
    ne = [k for k in a if a[k]["sha256"] != b[k]["sha256"]]
    finite_both = [k for k in a if a[k].get("finite") and b[k].get("finite")]
    out[f"{x}_vs_{y}"] = {"total": len(a), "equal": len(eq), "differ": len(ne), "differ_keys": ne,
                          "finite_both": len(finite_both),
                          "finite_both_equal": sum(a[k]["sha256"] == b[k]["sha256"] for k in finite_both),
                          "nonfinite_either": len(a) - len(finite_both)}
classes = cmp2["before_vs_after"]["by_class"]
out["compare_v2_by_class_total"] = sum(v["n"] for v in classes.values())
out["compare_v2_design_n"] = classes["design"]["n"]
out["compare_v2_nondesign_total"] = out["compare_v2_by_class_total"] - classes["design"]["n"]
out["claimed"] = "artifacts/apa_mf2_1/verdict.json G3: '481/481 finite-input keys'"
r = out["before_vs_after"]
out["erratum"] = (f"MF2.1 identity before vs after: {r['total']} keys total; {r['equal']} equal; {r['differ']} differ "
                  f"(the {len(r['differ_keys'])} design keys); non-design {r['total'] - r['differ']} all equal; "
                  f"{r['finite_both']} keys finite on both arms ({r['finite_both_equal']} equal); "
                  f"{r['nonfinite_either']} keys have a non-finite output on an arm (by construction: +-inf reduce rows "
                  f"and design rows). '481' matches none of these counts.")
print(json.dumps(out, indent=1))
json.dump(out, open("artifacts/apa_mf2_2/n5_recount.json", "x"), indent=1)
