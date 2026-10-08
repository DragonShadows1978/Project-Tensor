"""APA-MF2.2b G2/G3 from census arms (Amendment-1 harness). Mine.
Blocked-row cells = cells whose input has at least one row with every (visible) score -inf:
all_blocked / all_masked (any S), neg_inf_first at S=1, single_visible on causal paths at Lq=3, S>=3
(rows 0-1 see only blocked keys)."""
import collections, json, os, sys
A1 = "artifacts/apa_mf2_2a/census/"; A2 = "artifacts/apa_mf2_2b/census/"
S1, S2 = "20261007T2327Z", "20261007T2352Z"
ARM = {"fd32b6d": A1 + f"arm_fd32b6d_{S1}.json", "mf2": A1 + f"arm_mf2_{S1}.json", "mf2_1": A1 + f"arm_mf2_1_{S1}.json",
       "mf2_2": A1 + f"arm_mf2_2_{S1}.json", "after1": A1 + f"arm_after_{S1}.json",
       "after1f": A2 + f"arm_after1f_{S2}.json", "after2": A2 + f"arm_after2_{os.environ.get('AFTER2_STAMP', S2)}.json"}
A = {a: json.load(open(p)) for a, p in ARM.items()}
C = {a: r["cells"] for a, r in A.items()}
keys = list(C["fd32b6d"])
assert all(list(C[a]) == keys for a in C)
PRIM = ("eager_softmax", "causal_softmax")
CAUSAL = ("sdpa_infer_causal", "sdpa_infer_causal_fused", "sdpa_grad_causal", "apa_quant_p1.0_causalTrue",
          "apa_quant_p0.15_causalTrue", "causal_softmax")
def st(c): return "err" if "error" in c else ("F" if all(c[n]["finite"] for n in c) else "N")
def blocked_cell(k):
    p = k.split("/"); path, row = p[0], p[-1]
    S = int([x for x in p if x.startswith("S")][0][1:]); Lq = int([x for x in p if x.startswith("Lq")][0][2:])
    if row in ("all_blocked", "all_masked"): return True
    if row == "neg_inf_first" and S == 1: return True
    if row == "single_visible" and path in CAUSAL and Lq == 3 and S >= 3: return True
    return False
same = lambda a, b, k: json.dumps(C[a][k], sort_keys=True) == json.dumps(C[b][k], sort_keys=True)  # noqa: E731
res = {"arms": {a: {"engine_sha256": A[a]["provenance"]["engine_sha256"], "package": A[a]["provenance"]["package"]} for a in A}}
res["after1f_equals_after1_recorded"] = sum(same("after1f", "after1", k) for k in keys)
nb = [k for k in keys if not blocked_cell(k)]; bl = [k for k in keys if blocked_cell(k)]
res["cells"] = {"total": len(keys), "blocked_cells": len(bl), "non_blocked_cells": len(nb)}
newnan = [k for k in keys if st(C["fd32b6d"][k]) == "F" and st(C["after2"][k]) == "N"]
res["G2a_fd32b6d_finite_after2_nonfinite"] = {"count": len(newnan), "keys": newnan}
bl_nan = [k for k in bl if st(C["after2"][k]) == "N"]
res["G2b_blocked_cells_nonfinite_after2"] = {"count": len(bl_nan),
    "attention_paths": [k for k in bl_nan if k.split("/")[0] not in PRIM],
    "primitive_paths": collections.Counter(k.split("/")[0] + "/" + k.split("/")[-1] for k in bl_nan if k.split("/")[0] in PRIM)}
res["G2b_attention_blocked_cells"] = {"cells": sum(1 for k in bl if k.split("/")[0] not in PRIM and st(C["after2"][k]) != "err"),
                                      "finite_after2": sum(1 for k in bl if k.split("/")[0] not in PRIM and st(C["after2"][k]) == "F"),
                                      "blocked_rows_zero_by_first_value": None}
diff_nb = [k for k in nb if not same("after2", "after1", k)]
diff_nbf = [k for k in nb if not same("after2", "after1f", k)]
res["G3_non_blocked_after2_eq_after1"] = {"cells": len(nb), "differ_vs_recorded": len(diff_nb), "differ_vs_fresh": len(diff_nbf), "keys": diff_nb[:20]}
other_nan = [k for k in keys if st(C["after2"][k]) == "N" and k not in bl_nan]
res["after2_nonfinite_not_blocked"] = {"count": len(other_nan), "fd32b6d_also_nonfinite": sum(st(C["fd32b6d"][k]) == "N" for k in other_nan),
                                       "by_path_row": collections.Counter(k.split("/")[0] + "/" + k.split("/")[1] + "/" + k.split("/")[-1] for k in other_nan)}
res["G2_attention_pass"] = len(newnan) == 0 and not res["G2b_blocked_cells_nonfinite_after2"]["attention_paths"]
res["G2_literal_pass"] = len(newnan) == 0 and len(bl_nan) == 0
res["G3_census_pass"] = len(diff_nb) == 0 and len(diff_nbf) == 0
json.dump(res, open(sys.argv[1], "x"), indent=1)
print(json.dumps(res, indent=1, default=str)[:4000])
