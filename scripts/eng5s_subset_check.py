"""ENG5-S faithfulness receipt: this round's status code is a subset of ENG5-F2's (eng5 branch tip, read-only).

  python3 -B -P scripts/eng5s_subset_check.py <eng5 worktree> <out.json>

For kernels.cu, apa_spo/packed_prefill.cuh and include/tc/core.h it extracts every check_cuda_status(...) statement
(whitespace-normalized, up to its closing `;`) and the helper's definition and declaration, in this tree and in the
eng5 tree, and reports: each of this tree's statements found verbatim in ENG5-F2's; ENG5-F2's statements absent here
(expected: none of the nine sites); the helper's body equal. It also lists every line this round adds that is not a
comment, so a reviewer sees the code delta without the annotations.

Prior art: no prior art known to me (a text comparison).
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["tensor_cuda/src/kernels.cu", "tensor_cuda/src/apa_spo/packed_prefill.cuh", "tensor_cuda/include/tc/core.h"]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def statements(text):
    out = []
    for m in re.finditer(r"check_cuda_status\(", text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith("//") or text[m.start() - 5:m.start()] == "void ":
            continue
        end = text.index(";", m.start())
        out.append(norm(text[m.start():end + 1]))
    return out


def helper(text):
    i = text.find("void check_cuda_status(int status, const char* entry, const char* call) {")
    return norm(text[i:text.index("\n}\n", i) + 2]) if i >= 0 else None


def main():
    eng5, out = sys.argv[1], sys.argv[2]
    mine, theirs = [], []
    res = {"files": {}}
    for f in FILES:
        a = open(os.path.join(ROOT, f)).read()
        b = open(os.path.join(eng5, f)).read()
        sa, sb = statements(a), statements(b)
        mine += sa
        theirs += sb
        res["files"][f] = {"statements_here": sa, "here_not_in_eng5_f2": [s for s in sa if s not in sb],
                           "eng5_f2_not_here": [s for s in sb if s not in sa]}
    k_here = open(os.path.join(ROOT, FILES[0])).read()
    k_eng5 = open(os.path.join(eng5, FILES[0])).read()
    res["helper_equal"] = helper(k_here) == helper(k_eng5) and helper(k_here) is not None
    decl = "void check_cuda_status(int status, const char* entry, const char* call);"
    res["declaration_in_both"] = decl in open(os.path.join(ROOT, FILES[2])).read() and \
        decl in open(os.path.join(eng5, FILES[2])).read()
    res["statement_count_here"] = len(mine)
    res["every_statement_here_in_eng5_f2"] = all(s in theirs for s in mine)
    d = subprocess.run(["git", "-C", ROOT, "diff", "-U0", "fbd335ae7", "--"] + FILES, capture_output=True,
                       text=True).stdout
    added = [l[1:] for l in d.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in d.splitlines() if l.startswith("-") and not l.startswith("---")]
    res["added_code_lines"] = [l for l in added if l.strip() and not l.strip().startswith("//")]
    res["removed_lines"] = removed
    res["pass"] = res["helper_equal"] and res["declaration_in_both"] and res["every_statement_here_in_eng5_f2"] \
        and len(mine) == 9
    json.dump(res, open(out, "w"), indent=1)
    print("statements", len(mine), "all in ENG5-F2", res["every_statement_here_in_eng5_f2"], "helper equal",
          res["helper_equal"], "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
