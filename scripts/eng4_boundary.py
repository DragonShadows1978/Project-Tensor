"""ENG4 boundary probe: run every M3/M4 refusal-class witness (and the accepted-class controls) through the public
Python API with host operands, in THIS process, and record what the build does.

Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=<package parent> python3 -B -P scripts/eng4_boundary.py <label> <out.json>

- Every call is `tc.matmul(a, b, trans_b=...)` on `tc.tensor(np.zeros(shape, float32), device=...)` operands. The
  size classes (M4) use zero-element operands, so nothing large is allocated (order G1).
- The device class builds its CUDA operand with zero elements: Storage(0 bytes) and from_host(0 bytes) return before
  any CUDA call (kernels.cu Storage::Storage, NDArray::from_host), so the class runs with no visible GPU.
- Output rows: class id, operand shapes/devices, trans_b, then either {"raised": type, "message"} or {"returned":
  shape}. The module path and sha256 are recorded so the row says which build produced it.

Used for the L99 seat check (Build B: does the ENG-MM1 call reach the cuBLAS call?), BOUNDARY.json (L60: Build B's
behaviour per refusal class) and an E cross-check of G1.

Prior art: NumPy's matmul shape rule (NumPy docs, numpy.matmul) for the accepted-class controls; no other prior art
known to me (a plain witness table).
"""
import hashlib
import json
import os
import sys

import numpy as np

BIG = 2 ** 31  # INT_MAX + 1
MAXI = 2 ** 31 - 1

# (class id, gate class, A shape, B shape, trans_b, A device, B device)
CASES = [
    # M3 refusals
    ("m3_eq_rank_smaller_b_batch_ENG-MM1", "M3", (4, 2, 3), (2, 3, 4), False, "cpu", "cpu"),
    ("m3_eq_rank_smaller_b_batch_trans", "M3", (4, 2, 3), (2, 4, 3), True, "cpu", "cpu"),
    ("m3_eq_rank_larger_b_batch", "M3", (2, 2, 3), (4, 3, 4), False, "cpu", "cpu"),
    ("m3_eq_rank_larger_b_batch_a_ones", "M3", (1, 2, 3), (2, 3, 4), False, "cpu", "cpu"),
    ("m3_eq_count_diff_shape", "M3", (2, 3, 2, 3), (3, 2, 3, 4), False, "cpu", "cpu"),
    ("m3_eq_rank_ones_b_under_larger_a", "M3", (2, 2, 3), (1, 3, 4), False, "cpu", "cpu"),
    ("m3_eq_rank_ones_b_under_larger_a_rank4", "M3", (2, 3, 2, 3), (1, 1, 3, 4), False, "cpu", "cpu"),
    ("m3_uneq_rank_a2_b3_batched", "M3", (2, 3), (2, 3, 4), False, "cpu", "cpu"),
    ("m3_uneq_rank_a4_b3_batched", "M3", (2, 1, 2, 3), (2, 3, 4), False, "cpu", "cpu"),
    ("m3_uneq_rank_a3_b4_partial_ones", "M3", (2, 2, 3), (1, 2, 3, 4), False, "cpu", "cpu"),
    ("m3_uneq_rank_a3_b4_partial_ones_trans", "M3", (2, 2, 3), (1, 2, 4, 3), True, "cpu", "cpu"),
    # M4 device class: one host and one CUDA operand (zero-element CUDA operand, no CUDA call)
    ("m4_device_host_a_cuda_b", "M4-device", (2, 0), (0, 3), False, "cpu", "cuda"),
    ("m4_device_cuda_a_host_b", "M4-device", (2, 0), (0, 3), False, "cuda", "cpu"),
    # M4 size classes: zero-element operands
    ("m4_M_over_intmax", "M4-size", (BIG, 0), (0, 0), False, "cpu", "cpu"),
    ("m4_N_over_intmax", "M4-size", (0, 0), (0, BIG), False, "cpu", "cpu"),
    ("m4_N_over_intmax_trans", "M4-size", (0, 0), (BIG, 0), True, "cpu", "cpu"),
    ("m4_K_over_intmax", "M4-size", (0, BIG), (BIG, 0), False, "cpu", "cpu"),
    ("m4_K_over_intmax_trans", "M4-size", (0, BIG), (0, BIG), True, "cpu", "cpu"),
    ("m4_batch_over_intmax_b2d", "M4-size", (BIG, 0, 0), (0, 0), False, "cpu", "cpu"),
    ("m4_batch_over_intmax_eq_rank", "M4-size", (2 ** 16, 2 ** 15, 0, 0), (2 ** 16, 2 ** 15, 0, 0), False,
     "cpu", "cpu"),
    # accepted controls (the predicate accepts; the forward then goes on to its cuBLAS call as before)
    ("ok_a_b2d", "accept-a", (4, 2, 3), (3, 4), False, "cpu", "cpu"),
    ("ok_b_equal_batch", "accept-b", (4, 2, 3), (4, 3, 4), False, "cpu", "cpu"),
    ("ok_c_ones_b_rank4_a3", "accept-c", (2, 2, 3), (1, 1, 3, 4), False, "cpu", "cpu"),
    ("ok_c_ones_b_rank3_a2", "accept-c", (2, 3), (1, 3, 4), False, "cpu", "cpu"),
    ("ok_M_eq_intmax", "accept-a", (MAXI, 0), (0, 0), False, "cpu", "cpu"),
    ("ok_N_eq_intmax", "accept-a", (0, 0), (0, MAXI), False, "cpu", "cpu"),
    ("ok_K_eq_intmax", "accept-a", (0, MAXI), (MAXI, 0), False, "cpu", "cpu"),
    ("ok_batch_eq_intmax", "accept-a", (MAXI, 0, 0), (0, 0), False, "cpu", "cpu"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    label, out = sys.argv[1], sys.argv[2]
    import tensor_cuda as tc
    rows = []
    for cid, gate, sa, sb, tb, da, db in CASES:
        row = {"class": cid, "gate": gate, "a_shape": list(sa), "b_shape": list(sb), "trans_b": tb,
               "a_device": da, "b_device": db,
               "call": f"tc.matmul(tc.tensor(np.zeros({sa}, np.float32), device={da!r}), "
                       f"tc.tensor(np.zeros({sb}, np.float32), device={db!r}), trans_b={tb})"}
        try:
            a = tc.tensor(np.zeros(sa, np.float32), device=da)
            b = tc.tensor(np.zeros(sb, np.float32), device=db)
        except Exception as e:  # noqa: BLE001  (recorded, not hidden)
            row["operand_error"] = {"type": type(e).__name__, "message": str(e)}
            rows.append(row)
            continue
        try:
            r = tc.matmul(a, b, trans_b=tb)
            row["returned"] = {"shape": list(r.shape), "device": str(r.device)}
        except Exception as e:  # noqa: BLE001
            row["raised"] = {"type": type(e).__name__, "message": str(e)}
        rows.append(row)
        print(cid, row.get("raised") or row.get("returned"), flush=True)
    rec = {"label": label, "module": tc._C.__file__, "module_sha256": sha(tc._C.__file__),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "rows": rows}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
