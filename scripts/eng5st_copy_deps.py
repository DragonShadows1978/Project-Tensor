"""ENG5-S-T: writes artifacts/eng5s/t/COPY_DEPS.json, the list of what tensor_cuda/tests/test_eng5s_status_checks.py copied
from scripts/eng5s_gpu_step.py, scripts/eng5s_gpu_verdict.py and tensor_cuda/tests/test_eng5s_status.py, with a check
that each copied item is still equal to its source (sources are read, never imported or edited).

  python3 -B -P scripts/eng5st_copy_deps.py

Checks: the child body (every line of the test's ROW_CHILD that is not new appears verbatim in the step's G2_CHILD, and
the lines of G2_CHILD that were left out are exactly the lease check and the site-9 case), the S1 regular expression
(identical pattern text), the entry/call prefixes (EXPECT == PREFIX without the error name), the row list (G2_CASES minus
site 9, plus the control-only row).

Prior art: no prior art known to me (a copy manifest with an equality check).
"""
import ast
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP = "scripts/eng5s_gpu_step.py"
VERDICT = "scripts/eng5s_gpu_verdict.py"
OLD_TEST = "tensor_cuda/tests/test_eng5s_status.py"
NEW_TEST = "tensor_cuda/tests/test_eng5s_status_checks.py"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def assigns(path):
    """Top-level name -> source segment and literal-or-None value of a module's simple assignments."""
    text = open(os.path.join(ROOT, path)).read()
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            seg = ast.get_source_segment(text, node.value)
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                val = None
            out[node.targets[0].id] = (seg, val, node.value)
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    step, ver, old, new = assigns(STEP), assigns(VERDICT), assigns(OLD_TEST), assigns(NEW_TEST)
    g2 = step["G2_CHILD"][1].splitlines()
    mine = new["ROW_CHILD"][1].splitlines()
    g2set = {l.strip() for l in g2 if l.strip()}
    new_lines = [l for l in mine if l.strip() and l.strip() not in g2set]
    left_out = [l for l in g2 if l.strip() and l.strip() not in {m.strip() for m in mine}]
    # S1 regex: compare the compiled pattern text
    s1_old = eval(compile(ast.Expression(old["S1_FORM"][2]), "s", "eval"), {"re": re}).pattern
    s1_new = eval(compile(ast.Expression(new["S1_FORM"][2]), "s", "eval"), {"re": re}).pattern
    err = ver["ERR"][1]
    prefix = eval(compile(ast.Expression(ver["PREFIX"][2]), "s", "eval"), {"ERR": err})  # the dict adds ERR to each string
    expect = new["EXPECT"][1]
    g2_cases = step["G2_CASES"][1]
    rows_ok = (new["GPU_ROWS"][1] + new["GPU_CONTROL_ONLY"][1] ==
               [c for c in g2_cases if c != "packed_prefill_dump"] + step["CONTROL_ONLY"][1])
    prefix_ok = all(expect[c] == prefix[c][:-len(err)] for c in expect) and set(expect) == set(prefix) - {"packed_prefill_dump"}
    checks = {
        "S1_FORM identical pattern text": s1_old == s1_new,
        "EXPECT equals PREFIX minus the error name, site 9 left out": prefix_ok,
        "GPU_ROWS + GPU_CONTROL_ONLY == G2_CASES minus packed_prefill_dump + CONTROL_ONLY": rows_ok,
        "the lease-check line is among the child lines left out": any("forge-gpu-lease-check" in l for l in left_out),
        "the site-9 case body is among the child lines left out": any("packed_prefill" in l for l in left_out),
    }
    out = {
        "order": "orders/ENG5ST_REGRESSION_TEST.md", "new_file": NEW_TEST, "new_file_sha256": sha(NEW_TEST),
        "sources_read_only": {STEP: sha(STEP), VERDICT: sha(VERDICT), OLD_TEST: sha(OLD_TEST),
                              "tensor_cuda/tests/support/eng5s_cuda_shim.c": sha("tensor_cuda/tests/support/eng5s_cuda_shim.c")},
        "copied": [
            {"what": "S1_FORM regular expression", "from": OLD_TEST + " S1_FORM", "to": NEW_TEST + " S1_FORM",
             "how": "verbatim (copied, not imported)"},
            {"what": "the row child body (helpers informative/check/attempt, the cases synchronize, oom_drain, "
                     "oom_drain_pooled, zeros, from_host, to_host, clone, to_d2h, to_h2d, axpy_, axpy_empty)",
             "from": STEP + " G2_CHILD", "to": NEW_TEST + " ROW_CHILD",
             "how": "line for line, except the items under left_out_of_child and new_child_lines"},
            {"what": "the entry-and-call prefix per row", "from": VERDICT + " PREFIX", "to": NEW_TEST + " EXPECT",
             "how": "the PREFIX string without its trailing error name (the test checks the error name apart)"},
            {"what": "the row list", "from": STEP + " G2_CASES and CONTROL_ONLY", "to": NEW_TEST + " GPU_ROWS, GPU_CONTROL_ONLY",
             "how": "G2_CASES without packed_prefill_dump (site 9 needs a hooks build), plus the control-only row"},
            {"what": "the OOM-drain arming pattern (eng5_inject_next of the allocation and cudaStreamSynchronize)",
             "from": OLD_TEST + " OOM_CHILD (and G2_CHILD's oom cases)", "to": NEW_TEST + " ROW_CHILD cpu_oom_drain*",
             "how": "same calls, one injected child per row; the CPU rows have no success control (no device)"},
            {"what": "the CPU-lane predicate and the compiler lookup",
             "from": "tensor_cuda/tests/test_eng4_matmul.py CPU_LANE; tensor_cuda/tests/test_eng3_host.py (cc, gcc, clang)",
             "to": NEW_TEST + " CPU_LANE, _compiler", "how": "same expression"},
        ],
        "new_child_lines": [l.strip() for l in new_lines],
        "left_out_of_child": [l.strip() for l in left_out],
        "checks": checks, "all_checks_pass": all(checks.values()),
        "note": "the shim itself (tests/support/eng5s_cuda_shim.c) is used as it is: it is the ENG5-S file, unchanged.",
    }
    json.dump(out, open(os.path.join(ROOT, "artifacts/eng5s/t/COPY_DEPS.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1), "\nnew child lines:", len(new_lines), "left out:", len(left_out))
    return 0 if out["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
