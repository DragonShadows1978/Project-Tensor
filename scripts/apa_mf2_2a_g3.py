"""APA-MF2.2a G3: after vs MF2.2 on the MF2.2 identity census + N1 rows (same harness,
scripts/apa_mf2_2_identity.py, separate processes). Baselines: the ORIGINAL MF2.2 engine's recorded
arm (bc7ed2a9, artifacts/apa_mf2_2/identity/arm_mf2_2_20261007T231830Z.json) and a fresh arm of the
MF2.2-source rebuild. Mine."""
import json, sys
after, rebuild, out = sys.argv[1], sys.argv[2], sys.argv[3]
A = json.load(open(after)); R = json.load(open(rebuild))
O = json.load(open("artifacts/apa_mf2_2/identity/arm_mf2_2_20261007T231830Z.json"))
res = {"after": A["provenance"], "rebuild": R["provenance"], "original_mf2_2": O["provenance"]}
for name, B in (("original_mf2_2", O), ("rebuild_mf2_2", R)):
    for t in ("identity", "n1"):
        assert A[t].keys() == B[t].keys(), (name, t)
        d = [k for k in A[t] if A[t][k]["sha256"] != B[t][k]["sha256"]]
        res[f"{t}_vs_{name}"] = {"keys": len(A[t]), "equal": len(A[t]) - len(d), "differ_keys": d[:20]}
w = {k: v for k, v in A["n1"].items() if k.startswith("witness/")}
res["n1_after_all_finite"] = all(v["finite"] for v in A["n1"].values())
res["n1_witnesses_exactly_one"] = all(v["finite"] and v["max_err"] == 0.0 for v in w.values()) and len(w) == 18
res["observations_after"] = {k: v for k, v in A["observations"].items()}
res["G3_pass"] = (res["identity_vs_original_mf2_2"]["equal"] == res["identity_vs_original_mf2_2"]["keys"]
                  and res["n1_vs_original_mf2_2"]["equal"] == res["n1_vs_original_mf2_2"]["keys"]
                  and res["n1_after_all_finite"] and res["n1_witnesses_exactly_one"])
json.dump(res, open(out, "x"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("observations_after",)}, indent=1))
print("observations_after:", {k: (v.get("finite"), v.get("values", [None])[:1]) if isinstance(v, dict) else v for k, v in res["observations_after"].items()})
