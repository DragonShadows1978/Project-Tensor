"""ENG4 L60 GPU boundary: every refusal class with CUDA operands, one child process per case (a case that faults the
device or crashes on Build B cannot affect the next), for one build. Run inside the GPU line boundary_<arm>.

Usage: PYTHONPATH=<package parent> python3 -B -P scripts/eng4_boundary_gpu.py <arm> <out.json>
       python3 -B -P scripts/eng4_boundary_gpu.py --case <index>      (the child)

Operands: M3 classes use seeded nonzero float32 CUDA tensors (seed 4402 + case index); the device class uses one host
and one CUDA operand, both nonzero; the size classes use zero-element CUDA operands (nothing large is allocated).
Each row records the call, the child's exit code, and the forward's result (shape, sha256 of the output bytes) or its
exception, plus a synchronize.

Prior art: no prior art known to me (a witness table; the cases are this order's M3/M4 classes).
"""
import hashlib
import json
import subprocess
import sys

import numpy as np

BIG = 2 ** 31
CASES = [
    ("m3_eq_rank_smaller_b_batch_ENG-MM1", (4, 2, 3), "cuda", (2, 3, 4), "cuda", False, "nonzero"),
    ("m3_eq_rank_smaller_b_batch_trans", (4, 2, 3), "cuda", (2, 4, 3), "cuda", True, "nonzero"),
    ("m3_eq_rank_larger_b_batch", (2, 2, 3), "cuda", (4, 3, 4), "cuda", False, "nonzero"),
    ("m3_eq_rank_larger_b_batch_a_ones", (1, 2, 3), "cuda", (2, 3, 4), "cuda", False, "nonzero"),
    ("m3_eq_count_diff_shape", (2, 3, 2, 3), "cuda", (3, 2, 3, 4), "cuda", False, "nonzero"),
    ("m3_eq_rank_ones_b_under_larger_a", (2, 2, 3), "cuda", (1, 3, 4), "cuda", False, "nonzero"),
    ("m3_eq_rank_ones_b_under_larger_a_rank4", (2, 3, 2, 3), "cuda", (1, 1, 3, 4), "cuda", False, "nonzero"),
    ("m3_uneq_rank_a2_b3_batched", (2, 3), "cuda", (2, 3, 4), "cuda", False, "nonzero"),
    ("m3_uneq_rank_a4_b3_batched", (2, 1, 2, 3), "cuda", (2, 3, 4), "cuda", False, "nonzero"),
    ("m3_uneq_rank_a3_b4_partial_ones", (2, 2, 3), "cuda", (1, 2, 3, 4), "cuda", False, "nonzero"),
    ("m3_uneq_rank_a3_b4_partial_ones_trans", (2, 2, 3), "cuda", (1, 2, 4, 3), "cuda", True, "nonzero"),
    ("m4_device_cuda_a_host_b", (3, 5), "cuda", (5, 4), "cpu", False, "nonzero"),
    ("m4_device_host_a_cuda_b", (3, 5), "cpu", (5, 4), "cuda", False, "nonzero"),
    ("m4_M_over_intmax", (BIG, 0), "cuda", (0, 0), "cuda", False, "zero"),
    ("m4_N_over_intmax", (0, 0), "cuda", (0, BIG), "cuda", False, "zero"),
    ("m4_N_over_intmax_trans", (0, 0), "cuda", (BIG, 0), "cuda", True, "zero"),
    ("m4_K_over_intmax", (0, BIG), "cuda", (BIG, 0), "cuda", False, "zero"),
    ("m4_K_over_intmax_trans", (0, BIG), "cuda", (0, BIG), "cuda", True, "zero"),
    ("m4_batch_over_intmax_b2d", (BIG, 0, 0), "cuda", (0, 0), "cuda", False, "zero"),
    ("m4_batch_over_intmax_eq_rank", (2 ** 16, 2 ** 15, 0, 0), "cuda", (2 ** 16, 2 ** 15, 0, 0), "cuda", False,
     "zero"),
]


def child(i):
    import tensor_cuda as tc
    cid, sa, da, sb, db, tb, kind = CASES[i]
    rng = np.random.default_rng(4402 + i)
    mk = (lambda s: rng.standard_normal(s).astype(np.float32)) if kind == "nonzero" else (
        lambda s: np.zeros(s, np.float32))
    a, b = tc.tensor(mk(sa), device=da), tc.tensor(mk(sb), device=db)
    row = {}
    try:
        r = tc.matmul(a, b, trans_b=tb)
        host = r.numpy()
        tc.synchronize()
        row["returned"] = {"shape": list(host.shape), "device": str(r.device),
                           "sha256": hashlib.sha256(host.tobytes()).hexdigest()}
    except Exception as e:  # noqa: BLE001 (recorded)
        row["raised"] = {"type": type(e).__name__, "message": str(e)}
    print("ENG4-ROW " + json.dumps(row), flush=True)


def main():
    arm, out = sys.argv[1], sys.argv[2]
    rows = []
    for i, (cid, sa, da, sb, db, tb, kind) in enumerate(CASES):
        p = subprocess.run([sys.executable, "-B", "-P", __file__, "--case", str(i)], capture_output=True, text=True,
                           timeout=120)
        row = {"class": cid, "a_shape": list(sa), "a_device": da, "b_shape": list(sb), "b_device": db,
               "trans_b": tb, "operands": kind, "child_exit": p.returncode, "stderr_tail": p.stderr[-1500:]}
        for line in p.stdout.splitlines():
            if line.startswith("ENG4-ROW "):
                row.update(json.loads(line[len("ENG4-ROW "):]))
        rows.append(row)
        print(cid, p.returncode, row.get("raised") or row.get("returned"), flush=True)
    import tensor_cuda as tc
    rec = {"arm": arm, "module": tc._C.__file__,
           "module_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(), "rows": rows}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--case":
        child(int(sys.argv[2]))
    else:
        main()
