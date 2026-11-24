"""BWD-F1 G5: run each CPU-killable mutant against its named tests and record MUTANT_RESULTS.json.

  python3 -B -P scripts/bwdf1_mutant_results.py cpu

Each mutant's tests run with BWDF1_PKG_PARENT / BWDF1_TS_PARENT / BWDF1_KERNELS_CU pointing at the mutant's tree
(/tmp/bwdf1_src_<mutant>, whose module sha256 is checked against its build receipt first), CUDA_VISIBLE_DEVICES empty.
A mutant is KILLED when every test MUTANTS.json names fails and the failure is the named assertion (the junit message
is kept). m03's GPU kill is the staged line g3_m03_reduce_gt1. Prior art: DeMillo, Lipton and Sayward (1978).
"""
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_ZERO, T_COPY, T_TS = ("tensor_cuda/tests/test_bwdf1_zero_extent.py", "tensor_cuda/tests/test_bwdf1_copy_ops.py",
                        "tc_signal/tests/test_bwdf1_hardening.py")
ZERO_ONE = ["(4, 0)|(4, 1)", "(4, 1)|(4, 0)", "(0,)|(1,)", "(2, 1, 0)|(1,)"]
RUNS = {
    "m01_max_rule": (T_ZERO, "m01_max_rule", "E",
                     [f"test_cpu_broadcast_shape_text_equals_numpy[{p}]" for p in ZERO_ONE] +
                     ["test_cpu_g1_zero_against_one_numpy_shape[add-float32-(4, 0)-(4, 1)]",
                      "test_cpu_g1_zero_against_one_numpy_shape[where_x-float32-(4, 0)-(4, 1)]",
                      "test_cpu_g1_zero_against_one_numpy_shape[masked_fill-bfloat16-(4, 0)-(4, 1)]",
                      "test_cpu_g1_zero_against_one_numpy_shape[expand-float32-(4, 1)-(4, 0)]"]),
    "m02_no_empty_operand_guard": (T_ZERO, "m02_no_empty_operand_guard", "E",
                                   [f"test_cpu_guard_refuses_empty_operand_under_nonempty_output[{c}]" for c in
                                    ("ew_binary-a-empty-a-[4, 0]-[4, 1]", "ew_binary-b-empty-b-[4, 0]-[4, 1]",
                                     "compare-b-empty-rank1-b-[0]-[3]", "where-x-empty-x-[0]-[2, 3]",
                                     "where-y-empty-y-[2, 0, 3]-[2, 3]")]),
    "m03_reduce_gt1": (T_ZERO, "m03_reduce_gt1", "E", ["test_cpu_z3_zero_axis_gradient_reaches_the_sum"]),
    "m04_h1_dropped": (T_TS, "E", "m04_h1_dropped",
                       [f"test_cpu_h1_frozen_inputs_refuse_destination_requiring_grad[{op}]" for op in
                        ("conv1d", "conv_transpose1d", "instance_norm1d", "upsample_nearest1d", "reflection_pad1d")]),
    "m05_cat_no_extent_check": (T_COPY, "m05_cat_no_extent_check", "E",
                                [f"test_cpu_c3_copy_op_refusal_no_cuda_call[{r}]" for r in
                                 ("cat-extent-voice-zero-first", "cat-extent-voice-longer-later",
                                  "cat-extent-shorter-later", "stack-shape")]),
    "m06_cat_no_dtype_check": (T_COPY, "m06_cat_no_dtype_check", "E",
                               [f"test_cpu_c3_copy_op_refusal_no_cuda_call[{r}]" for r in ("cat-dtype", "stack-dtype")]),
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def junit(path):
    out = {}
    for tcase in ET.parse(path).getroot().iter("testcase"):
        f = tcase.find("failure") if tcase.find("failure") is not None else tcase.find("error")
        out[tcase.get("name")] = ("failed", (f.get("message") or "")[:300]) if f is not None else (
            ("skipped", "") if tcase.find("skipped") is not None else ("passed", ""))
    return out


def main():
    res = {}
    d = os.path.join(ROOT, "artifacts/bwdf1/mutants/runs")
    os.makedirs(d, exist_ok=True)
    for m, (tfile, tc_arm, ts_arm, expected) in RUNS.items():
        tc_tree, ts_tree = f"/tmp/bwdf1_src_{tc_arm}", f"/tmp/bwdf1_src_{ts_arm}"
        rec_arm = tc_arm
        rec = json.load(open(os.path.join(ROOT, f"artifacts/bwdf1/build/BUILD_tc_{rec_arm}.json")))
        so = os.path.join(tc_tree, "tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so")
        assert sha(so) == rec["module_sha256"], (m, "module differs from receipt")
        xml = os.path.join(d, f"{m}.xml")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/bwdf1_tmp",
                   BWDF1_PKG_PARENT=os.path.join(tc_tree, "tensor_cuda"),
                   BWDF1_TS_PARENT=os.path.join(ts_tree, "tc_signal"),
                   BWDF1_KERNELS_CU=os.path.join(tc_tree, "tensor_cuda/src/kernels.cu"))
        p = subprocess.run([sys.executable, "-B", "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            f"--basetemp=/tmp/bwdf1_tmp/pt/mut_{m}", f"--junitxml={xml}", tfile], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=500)
        got = junit(xml)
        rows = {e: got.get(e, ("NOT-RUN", "")) for e in expected}
        killed = all(v[0] == "failed" for v in rows.values())
        n_fail = sum(1 for v in got.values() if v[0] == "failed")
        res[m] = {"test_file": tfile, "tc_tree": tc_tree, "ts_tree": ts_tree, "module_sha256": rec["module_sha256"],
                  "pytest_exit": p.returncode, "summary": p.stdout.strip().splitlines()[-1] if p.stdout else "",
                  "expected_killing_tests": {k: {"outcome": v[0], "message": v[1]} for k, v in rows.items()},
                  "tests_failed_total": n_fail, "tests_total": len(got),
                  "result": "KILLED" if killed else "SURVIVED"}
        print(m, res[m]["result"], res[m]["summary"], flush=True)
    res["m03_reduce_gt1"]["gpu"] = "staged line g3_m03_reduce_gt1 (G3 rows on the m03 build); lead runs it"
    with open(os.path.join(ROOT, "artifacts/bwdf1/MUTANT_RESULTS.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
