"""APA-MF1 mutation run: registered mutants M1..M13, one at a time.

Each mutant rewrites ONE line of a tree file, runs the MF1 suite (pytest -x on
the three MF1 test files, default APA_MF1_RULE=B, GPU lock + timeout per run),
then restores the original bytes and verifies the sha256 (receipted). Killed =
pytest exit code 1 (>= 1 test failure); exit codes other than 0/1 are 'error'.

Prior art: mutation testing (DeMillo, Lipton & Sayward 1978; HOUSE_RULES s8 /
AtlasForge mutation_testing). The line-swap harness is mine.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts/apa_mf1/mutation")
Q = "tensor_cuda/tensor_cuda/quant.py"
C = "tensor_gpu_v2/_core.py"
T = "apa_cuda/tests/test_parity.py"
TRIU = "tc.tensor(np.triu(np.ones((L, S), np.float32), {k}), device=dev)"
TL = "xp.arange(S).reshape(1, -1) > xp.arange(i, end_i).reshape(-1, 1)"
MUTANTS = {
    "M1": (Q, 181, "masked_fill(causal_add.float(), 0.0)", "causal_add.float()", TRIU.format(k="1")),
    "M2": (Q, 181, "masked_fill(causal_add.float(), 0.0)", "causal_add.float()", TRIU.format(k="S - L")),
    "M3": (Q, 181, "masked_fill(causal_add.float(), 0.0)", "causal_add.float()", TRIU.format(k="1 + L - S")),
    "M4": (C, None, "rows = xp.arange(row_start, row_end).reshape(-1, 1) + (S - L)", "+ (S - L)", "+ (S - L - 1)"),
    "M5": (C, None, "rows = xp.arange(row_start, row_end).reshape(-1, 1) + (S - L)", "+ (S - L)", "+ (L - S)"),
    "M6": (C, 1677, "causal_mask = _causal_mask_bool(xp, L, S)", "_causal_mask_bool(xp, L, S)", "xp.triu(xp.ones((L, S), dtype=xp.bool_), k=1)"),
    "M7": (C, 2258, "cmask = _causal_mask_bool(xp, L, S)", "_causal_mask_bool(xp, L, S)", "xp.triu(xp.ones((L, S), dtype=xp.bool_), k=1)"),
    "M8": (C, 2154, "causal_mask = _causal_mask_bool(xp, L, S, i, end_i)", "_causal_mask_bool(xp, L, S, i, end_i)", TL),
    "M9": (Q, 191, "np.arange(S - L + 1, S + 1, dtype=np.float32)", "np.arange(S - L + 1, S + 1,", "np.arange(S - L, S,"),
    "M10": (Q, 191, "np.arange(S - L + 1, S + 1, dtype=np.float32)", "np.arange(S - L + 1, S + 1,", "np.arange(S - L + 2, S + 2,"),
    "M11": (Q, 194, "var = ((absr_f * absr_f).sum([-1], True) * inv - mean * mean).relu()",
            "var = ((absr_f * absr_f).sum([-1], True) * inv - mean * mean).relu()",
            "var = (((absr_f * absr_f).sum([-1], True) * inv - mean * mean) * tc.tensor((cnt / np.maximum(cnt - 1.0, 1.0))[:, None], device=dev)).relu()"),
    "M12": (T, 110, "k=1 + (S - L))", "k=1 + (S - L))", "k=1)"),
    "M13": (C, 1837, "causal_mask = _causal_mask_bool(xp, L, S, i, end_i)", "_causal_mask_bool(xp, L, S, i, end_i)", TL),
}
SUITE = ["tensor_cuda/tests/test_apa_mf1_quant.py", "apa_cuda/tests/test_apa_mf1_tgv2.py",
         "apa_cuda/tests/test_apa_mf1_parity_ref.py"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run_one(mid):
    path, lineno, expect, old, new = MUTANTS[mid]
    fp = os.path.join(ROOT, path)
    orig = open(fp, "rb").read()
    lines = orig.decode().split("\n")
    if lineno is None:
        hits = [i for i, l in enumerate(lines) if expect in l]
        assert len(hits) == 1, (mid, hits)
        idx = hits[0]
    else:
        idx = lineno - 1
    assert expect in lines[idx] and lines[idx].count(old) == 1, (mid, lines[idx])
    mutated_line = lines[idx].replace(old, new)
    lines[idx] = mutated_line
    t0 = time.time()
    try:
        with open(fp, "w") as f:
            f.write("\n".join(lines))
        cmd = ["flock", "-w", "900", "/tmp/forge-gpu.lock", "timeout", "600", sys.executable, "-m", "pytest",
               *SUITE, "-x", "-q", "-p", "no:cacheprovider"]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finally:
        with open(fp, "wb") as f:
            f.write(orig)
    restored = sha(open(fp, "rb").read()) == sha(orig)
    tail = [l for l in p.stdout.splitlines() if l.strip()][-3:]
    status = "killed" if p.returncode == 1 else ("survived" if p.returncode == 0 else "error")
    rec = dict(mutant=mid, file=path, line=idx + 1, mutated_line=mutated_line.strip(), returncode=p.returncode,
               status=status, restored_sha_ok=restored, orig_sha256=sha(orig), t_s=time.time() - t0, pytest_tail=tail)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{mid}.json"), "x") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec), flush=True)
    assert restored, f"{mid}: restore failed"


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        run_one(mid)
