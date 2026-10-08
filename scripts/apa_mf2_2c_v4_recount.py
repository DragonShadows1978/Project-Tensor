"""APA-MF2.2c V4: recount the Amendment-2 'after2 == after1 on 1230 non-blocked census cells' claim by
record class (unsupported-operand error / non-finite / finite tensor bytes). Mine."""
import json
from importlib.machinery import SourceFileLoader
cmp_ = SourceFileLoader("c", "scripts/apa_mf2_2b_compare.py")  # reuse its blocked-cell rule without running it
src = open("scripts/apa_mf2_2b_compare.py").read()
ns = {}
exec(src.split("same = lambda")[0].split("A = {a: json.load")[0], ns)        # constants only
CAUSAL = ("sdpa_infer_causal", "sdpa_infer_causal_fused", "sdpa_grad_causal", "apa_quant_p1.0_causalTrue",
          "apa_quant_p0.15_causalTrue", "causal_softmax")
def blocked_cell(k):
    p = k.split("/"); path, row = p[0], p[-1]
    S = int([x for x in p if x.startswith("S")][0][1:]); Lq = int([x for x in p if x.startswith("Lq")][0][2:])
    return row in ("all_blocked", "all_masked") or (row == "neg_inf_first" and S == 1) or \
        (row == "single_visible" and path in CAUSAL and Lq == 3 and S >= 3)
a2 = json.load(open("artifacts/apa_mf2_2b/census/arm_after2_20261008T0000Z.json"))["cells"]
a1 = json.load(open("artifacts/apa_mf2_2a/census/arm_after_20261007T2327Z.json"))["cells"]
nb = [k for k in a2 if not blocked_cell(k)]
cls = {"unsupported_operand_error": 0, "nonfinite_record": 0, "finite_tensor_bytes_equal": 0, "differ": 0}
for k in nb:
    eq = json.dumps(a2[k], sort_keys=True) == json.dumps(a1[k], sort_keys=True)
    if not eq: cls["differ"] += 1
    elif "error" in a2[k]: cls["unsupported_operand_error"] += 1
    elif not all(c["finite"] for c in a2[k].values()): cls["nonfinite_record"] += 1
    else: cls["finite_tensor_bytes_equal"] += 1
res = {"claim": "artifacts/apa_mf2_2b/verdict.json G3: 'after2 == after1 bytes on all 1230 census cells without an all -inf row'",
       "non_blocked_records": len(nb), **cls,
       "corrected_wording": f"after2 == after1 on all {len(nb)} non-blocked census records: {cls['finite_tensor_bytes_equal']} finite tensor records byte-equal, "
                            f"{cls['nonfinite_record']} pre-existing non-finite gradient records equal (canonical NaN), "
                            f"{cls['unsupported_operand_error']} records are the same unsupported-operand error (bf16 into the fp16/fp32-only fused kernels) and carry no tensor bytes"}
json.dump(res, open("artifacts/apa_mf2_2c/v4_recount.json", "x"), indent=1); print(json.dumps(res, indent=1))
