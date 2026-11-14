#!/usr/bin/env python3
"""ENG2 L38: write L38_PINS.json, the trace of every read of a norm class's output and every dtype-string result in
the inherited test files that line (c) runs (tool/regression_files.json), made BEFORE line (c)'s expected outcomes are
frozen. The trace was done by reading each call site and the variables it flows from (parametrize lists, helper
defaults, local assignments), not by grepping assert lines; the call sites themselves come from the AST census
(P303_caller_census.json), which sees multi-line calls. Reads of norm outputs come from the five files that construct
a norm class (grep of `(nn|tc.nn).<Class>(` over the test tree, plus nn.py's internal users).
Prior art: none known to me (a hand trace recorded as data).
"""
import json
from pathlib import Path

E = Path(__file__).resolve().parent
reg = json.loads((E / "tool/regression_files.json").read_text())
census = json.loads((E / "P303_caller_census.json").read_text())
root = str(E.parents[1]) + "/"
files = reg["regnorm"] + reg["regdtype"]

# dtype values per call site, traced by hand (file -> {line: (variable source, values)})
TRACE = {
    "test_apa_int4_sdpa_noncausal.py": {132: ("dts = 'float16' if q.dtype == np.float16 else 'float32' (:121)", ["float16", "float32"])},
    "test_apa_mf1_quant.py": {83: ("CASES (:44)", ["float32", "bfloat16"])},
    "test_apa_mf2_1_definition.py": {85: ("DTYPES (:35) via CELLS", ["float32", "bfloat16", "float16"]),
                                     155: ("DTYPES (:152)", ["float32", "bfloat16", "float16"]),
                                     156: ("DTYPES (:152)", ["float32", "bfloat16", "float16"]),
                                     170: ("count_witness default 'float32' / parametrize ['float32','float16'] (:173)", ["float32", "float16"])},
    "test_apa_mf2_1_gaps.py": {39: ("_f4_operands default 'float16' (:35)", ["float16"])},
    "test_apa_mf2_1_sentinel.py": {37: ("parametrize float32/bfloat16 (:52,:95)", ["float32", "bfloat16"]),
                                   97: ("parametrize float32/bfloat16 (:95)", ["float32", "bfloat16"]),
                                   137: ("parametrize (:132)", ["float32", "bfloat16", "float16"])},
    "test_apa_mf2_1_short.py": {50: ("DTYPES (:27)", ["float32", "bfloat16", "float16"])},
    "test_apa_mf2_2_n1.py": {50: ("_t(a, dt): dt from parametrize ['float32','float16'] / (:58)", ["float32", "float16"])},
    "test_apa_mf2_2_n2.py": {150: ("parametrize (:141)", ["float32", "bfloat16"])},
    "test_apa_mf2_2a_blocked.py": {71: ("parametrize (:59)", ["float32", "float16"])},
    "test_apa_mf2_2b_blocked.py": {33: ("DTS (:28)", ["float32", "float16", "bfloat16"])},
    "test_apa_mf2_2c_longkeys.py": {136: ("parametrize (:128)", ["float32", "bfloat16"])},
    "test_apa_mf2_mask.py": {93: ("DTYPES (:43)", ["float32", "bfloat16", "float16"]),
                             159: ("DTYPES (:43)", ["float32", "bfloat16", "float16"]),
                             182: ("DTYPES (:43)", ["float32", "bfloat16", "float16"])},
    "test_apa_mf2_rule.py": {149: ("parametrize (:146)", ["float32", "bfloat16", "float16"]),
                             150: ("parametrize (:146)", ["float32", "bfloat16", "float16"])},
    "test_apa_selective_int4.py": {126: ("_run callers pass 'float32' (:151,:203,:213,:214)", ["float32"]),
                                  127: ("same", ["float32"]), 128: ("same", ["float32"])},
    "test_apa_selective_splitk.py": {k: ("DTYPES (:96)", ["float16", "bfloat16", "float32"]) for k in (45, 46, 47, 48, 68, 69, 70, 71, 72)},
    "test_argmax_last_axis.py": {29: ("DTYPES (:11)", ["float32", "float16", "bfloat16"])},
    "test_dlpack.py": {214: ("tc.empty (C++ dtype_from_string; unchanged by ENG2)", ["n/a: C++ path"]),
                       559: ("tc.empty", ["n/a: C++ path"]), 578: ("tc.empty 'float99' (refusal test, C++)", ["float99"]),
                       670: ("tc.empty", ["n/a: C++ path"]), 952: ("tc.empty", ["n/a: C++ path"]),
                       958: ("tc.empty", ["n/a: C++ path"]), 2218: ("tc.empty", ["n/a: C++ path"])},
    "test_eng1.py": {70: ("_leaf(values, dtype): E2_DTYPES incl. 'float64' (test_e2_pow_grad_bits :266), E2_ENGINE_DTYPE "
                          "values, E3_DTYPES, 'float32' (E5)", ["float16", "bfloat16", "float32", "float64"]),
                     76: ("_const(values, dtype): E2_DTYPES incl. 'float64' (test_e2_pow_grad :204 probe; bits :275), "
                          "E3_DTYPES, 'float32'", ["float16", "bfloat16", "float32", "float64"])},
    "test_groupnorm_gn1.py": {k: ("DTYPES = tuple(TOLERANCES) (:30); :382 ('int64','bool','uint8')",
                                  ["float32", "float16", "bfloat16", "int64", "bool", "uint8"])
                              for k in (113, 187, 199, 219, 283, 303, 387, 389)},
    "test_hy3d_engine_ops.py": {69: ("dtype = 'float16' if q.dtype == np.float16 else 'float32' (:63)", ["float16", "float32"]),
                                70: ("same", ["float16", "float32"]),
                                200: ("'float16' if dtype is np.float16 else 'float32'", ["float16", "float32"])},
    "test_qtile_attention.py": {k: ("dts = 'float16' if q.dtype == np.float16 else 'float32' (:50,:68)", ["float16", "float32"])
                                for k in (55, 56, 79, 80)},
}
SUPPORTED = {"float32", "float16", "int64", "bool", "uint8", "bfloat16"}

rows = []
for r in census["rows"]:
    rel = r["file"].replace(root, "")
    if rel not in files or r["call"].split(".")[0] not in ("tc", "TC", "tensor_cuda"):
        continue
    base = rel.rsplit("/", 1)[1]
    if r["cls"] == "supported":
        values, source = [r["dtype"].strip("'\"")], "literal"
    else:
        source, values = TRACE[base][r["line"]]
    c_path = r["call"].endswith(".empty")
    unsupported = [v for v in values if v not in SUPPORTED and not c_path and not v.startswith("n/a")]
    rows.append(dict(site=f"{rel}:{r['line']}", call=r["call"], dtype_expr=r["dtype"], source=source, values=values,
                     path="C++ dtype_from_string (unchanged)" if c_path else "Python _np_dtype (ENG2 F1)",
                     changes=bool(unsupported), unsupported_values=unsupported))

NORM_READS = [
    dict(site="tensor_cuda/tests/test_attention_phase5.py:51-54", cls="RMSNorm", input_dtype="float32",
         read="np.isfinite(out) on the output", expectation="unchanged: float32 path, same op sequence (P302)"),
    dict(site="tensor_cuda/tests/test_attention_phase5.py:44-47", cls="LayerNorm (inside nn.TransformerEncoderLayer)",
         input_dtype="float32", read="output shape", expectation="unchanged: float32 path"),
    dict(site="tensor_cuda/tests/test_nn_phase3.py:17-22", cls="LayerNorm", input_dtype="float32",
         read="row mean ~0 (atol 1e-3) and std ~1 (atol 1e-2)", expectation="unchanged: float32 path"),
    dict(site="tensor_cuda/tests/test_ext_phase7.py:35-40", cls="BatchNorm1D, GroupNorm, InstanceNorm2D",
         input_dtype="float32", read="output shapes", expectation="unchanged: float32 path; GroupNorm untouched"),
    dict(site="tensor_cuda/tests/test_groupnorm_gn1.py (all)", cls="GroupNorm", input_dtype="float32/float16/bfloat16 "
         "and int64/bool/uint8 refusals", read="values vs float64 references, refusal messages",
         expectation="unchanged: GroupNorm is not edited; tc.tensor dtypes are all supported"),
    dict(site="tensor_cuda/tests/test_eng1.py::test_e3_census (40 IDs)", cls="all five",
         input_dtype="float16/bfloat16", read="census row values written to ENG1_CENSUS_DIR; the test asserts only "
         "shape, dtype, finite reference (measurement only)",
         expectation="outcome unchanged (passes on both arms); ROW VALUES expected to change for LayerNorm, RMSNorm, "
                     "BatchNorm1D and InstanceNorm2D (float32 statistics; N1_EMULATION.json predicts every row within "
                     "E3's tolerance on the candidate); GroupNorm rows unchanged"),
]

EXPECTED = {}
for test in ("test_e2_pow_grad", "test_e2_pow_grad_bits"):
    for p in ("p-1.5", "p-0.5", "p+0.5", "p+2", "p+3"):
        EXPECTED[f"tensor_cuda/tests/test_eng1.py::{test}[float64-{p}]"] = dict(
            baseline="passed", candidate="failed",
            detail_contains="TypeError: unsupported dtype 'float64'; supported dtypes: float32, float16, int64, bool, uint8, bfloat16",
            why="test_eng1.py builds its 'float64' rows with tc.tensor(..., dtype='float64') (:70/:76 via E2_DTYPES), "
                "which relied on the silent float32 fallback that F1 removes; inherited tests are not edited (L38)",
            receipts=["artifacts/eng2/g2/l38_float64_cpu.txt (candidate: 10 TypeError failures on CPU)",
                      "ENG1 G5 candidate_eng1 passed these IDs on the engine that is ENG2's baseline"])
assert all(i in (E / "tool/ids_regnorm.txt").read_text().split() for i in EXPECTED)

out = dict(order="orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md", line_c_files=files, norm_output_reads=NORM_READS,
           dtype_string_sites=rows, sites_that_change=[r for r in rows if r["changes"]],
           expected_changes=EXPECTED,
           note="A pin listed in expected_changes is not a regression: tool/evidence.py regression_reasons skips it and "
                "instead requires the registered baseline and candidate outcomes and the failure text.")
(E / "L38_PINS.json").write_text(json.dumps(out, indent=1) + "\n")
print(len(rows), "dtype sites;", sum(r["changes"] for r in rows), "that change;", len(EXPECTED), "expected changes")
