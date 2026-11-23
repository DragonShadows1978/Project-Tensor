"""ENG4-F2 bwd0: the backward of a batch-0 forward with a shared, nonempty B (Y2's pairs), for one build.

  python3 -B -P scripts/eng4_f2_bwd0.py run <outdir>          (GPU line bwd0_<arm>; one child process per case)
  python3 -B -P scripts/eng4_f2_bwd0.py --case <i> <device>   (the child; prints one ENG4-ROW json line)
  python3 -B -P scripts/eng4_f2_bwd0.py verdict <outdir> <B|F2>  (CPU, after the line: VERDICT.json)

This script never reads, sets or clears CUDA_VISIBLE_DEVICES; the GPU line's environment decides the device.

Cases: classes a and c with A's batch 0 (a zero leading extent) and B shared: A (0,2,3) x B (3,4); A (0,2,3) x
B (1,1,3,4); A (2,0,2,3) x B (3,4); A (0,3,2,3) x B (1,3,4); trans_b both ways; float32, float16, bfloat16 (24
cases). B holds seeded nonzero values (seed 4406 + case index); A is empty. Each child runs out = tc.matmul(a, b,
trans_b); out.backward(), and records the forward's output shape and either the backward's exception or both
gradients (shape, dtype, the bytes of .numpy() as hex; bfloat16 comes back as its exact float32 widening).

Verdict (L09): F2 mode: every case must not raise; A's gradient has A's shape and no elements; B's gradient has B's
dtype and shape (after those checks) and every element is a positive zero (by value, on the unsigned view). B mode:
recorded only (the order asserts nothing about Build B beyond this record, L81).

Prior art: the gradient of a broadcast operand is the sum over the broadcast axes, zero over an empty axis
(reverse-mode autodiff; PyTorch's sum_to_size); NumPy's empty sum. Ours: the witness table.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

CLASSES = [("a", (0, 2, 3), (3, 4)), ("c", (0, 2, 3), (1, 1, 3, 4)), ("a_rank4", (2, 0, 2, 3), (3, 4)),
           ("c_rank4_b3", (0, 3, 2, 3), (1, 3, 4))]
DTYPES = ("float32", "float16", "bfloat16")


def cases():
    out = []
    for dtype in DTYPES:
        for cls, sa, sb in CLASSES:
            for tb in (False, True):
                b = sb[:-2] + (sb[-1], sb[-2]) if tb else sb
                out.append({"id": f"{cls}-{dtype}-{'T' if tb else 'N'}", "class": cls, "dtype": dtype, "trans_b": tb,
                            "a": list(sa), "b": list(b)})
    for i, c in enumerate(out):
        c["seed"] = 4406 + i
    return out


def record_case(tc, c, device):
    """Run one case on `device` with the imported tensor_cuda `tc`; return its record (no exception escapes)."""
    rng = np.random.default_rng(c["seed"])
    b_np = rng.standard_normal(c["b"]).astype(np.float32)
    row = {"id": c["id"], "device": device, "b_input_sha256": hashlib.sha256(b_np.tobytes()).hexdigest()}
    try:
        a = tc.tensor(np.zeros(c["a"], np.float32), dtype=c["dtype"], device=device, requires_grad=True)
        b = tc.tensor(b_np, dtype=c["dtype"], device=device, requires_grad=True)
        out = tc.matmul(a, b, trans_b=c["trans_b"])
        row["out_shape"] = list(out.shape)
    except Exception as e:  # noqa: BLE001 (recorded)
        row["forward_raised"] = {"type": type(e).__name__, "message": str(e)}
        return row
    try:
        out.backward()
    except Exception as e:  # noqa: BLE001 (recorded)
        row["backward_raised"] = {"type": type(e).__name__, "message": str(e)}
        return row
    for name, t in (("a", a), ("b", b)):
        if t.grad is None:
            row[f"{name}_grad"] = None
            continue
        g = np.ascontiguousarray(t.grad.numpy())
        row[f"{name}_grad"] = {"shape": list(t.grad.shape), "dtype": t.grad.dtype, "host_dtype": str(g.dtype),
                               "elements": int(g.size), "hex": g.tobytes().hex(),
                               "sha256": hashlib.sha256(g.tobytes()).hexdigest()}
    return row


def child(i, device):
    import tensor_cuda as tc
    row = record_case(tc, cases()[i], device)
    try:
        tc.synchronize()
    except Exception as e:  # noqa: BLE001
        row["synchronize_raised"] = str(e)
    print("ENG4-ROW " + json.dumps(row), flush=True)


def run(outdir):
    os.makedirs(outdir)  # write-once
    rows = []
    for i, c in enumerate(cases()):
        p = subprocess.run([sys.executable, "-B", "-P", os.path.abspath(__file__), "--case", str(i), "cuda"],
                           capture_output=True, text=True, timeout=120)
        row = dict(c, child_exit=p.returncode, stderr_tail=p.stderr[-1500:])
        for line in p.stdout.splitlines():
            if line.startswith("ENG4-ROW "):
                row.update(json.loads(line[len("ENG4-ROW "):]))
        rows.append(row)
        print(c["id"], p.returncode, row.get("backward_raised") or ("grads" if "b_grad" in row else row.get(
            "forward_raised")), flush=True)
    import tensor_cuda as tc
    rec = {"module": tc._C.__file__, "module_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
           "rows": rows}
    with open(os.path.join(outdir, "bwd0.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return 0


def judge_row(r):
    """F2 rule for one record; returns (verdict, reason)."""
    if r.get("forward_raised") or r.get("backward_raised") or r.get("child_exit", 0) != 0:
        return "FAIL", f"raised or died: {r.get('forward_raised') or r.get('backward_raised') or r.get('child_exit')}"
    ga, gb = r.get("a_grad"), r.get("b_grad")
    if ga is None or ga["shape"] != r["a"] or ga["elements"] != 0:
        return "FAIL", "A's gradient is not A's empty gradient"
    if gb is None:
        return "FAIL", "B got no gradient"
    if gb["dtype"] != r["dtype"] or gb["shape"] != r["b"]:
        return "FAIL", f"B's gradient dtype/shape {gb['dtype']} {gb['shape']}"
    raw = bytes.fromhex(gb["hex"])
    if gb["elements"] == 0 or len(raw) == 0:
        return "FAIL", "B's gradient is empty"
    width = np.uint16 if gb["host_dtype"] == "float16" else np.uint32
    if np.any(np.frombuffer(raw, dtype=width)):
        return "FAIL", "B's gradient is not all positive zeros"
    return "PASS", "zeros of B's shape and dtype; A's gradient empty"


def verdict(outdir, mode):
    rec = json.load(open(os.path.join(outdir, "bwd0.json")))
    rows = []
    for r in rec["rows"]:
        if mode == "B":
            rows.append({"id": r["id"], "verdict": "RECORDED",
                         "result": r.get("backward_raised") or r.get("forward_raised") or
                         {"b_grad": {k: v for k, v in (r.get("b_grad") or {}).items() if k != "hex"}}})
        else:
            v, why = judge_row(r)
            rows.append({"id": r["id"], "verdict": v, "reason": why})
    overall = "RECORDED" if mode == "B" else ("PASS" if rows and all(x["verdict"] == "PASS" for x in rows)
                                              else "FAIL")
    out = {"mode": mode, "module_sha256": rec["module_sha256"], "cases": len(rows), "verdict": overall, "rows": rows}
    with open(os.path.join(outdir, "VERDICT.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("bwd0", mode, overall, len(rows))
    return 0 if overall in ("PASS", "RECORDED") else 1


if __name__ == "__main__":
    if sys.argv[1] == "--case":
        child(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "run":
        sys.exit(run(sys.argv[2]))
    elif sys.argv[1] == "verdict":
        sys.exit(verdict(sys.argv[2], sys.argv[3]))
