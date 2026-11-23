"""ENG4-F3 K4 witness: a pending, non-sticky CUDA error in the last-error slot, made through the public API, then a
zero fill. Run inside the GPU line pending_F2_F3 (one child process per build and variant).

  python3 -B -P scripts/eng4_f3_pending.py run <outdir>                  (the line: children for F2 and F3)
  python3 -B -P scripts/eng4_f3_pending.py --child <variant>             (one child; prints one ENG4-ROW json line)
  python3 -B -P scripts/eng4_f3_pending.py verdict <outdir>              (CPU, after the line: VERDICT.json)

This script never reads, sets or clears CUDA_VISIBLE_DEVICES; the line's environment decides the device, and each
child's PYTHONPATH (set by the line) selects the build's frozen package.

How the error is made pending (reading at 9b7eb1796): a K = 0 matmul of zero-element CUDA operands A (2^19, 0) and
B (0, 2^19) needs a 2^38-element float32 output (1 TiB). Building the operands makes no CUDA call (zero bytes). The
output allocation (kernels.cu Storage::Storage) fails, clears the slot once (cudaGetLastError), retries, fails again
and throws "cudaMalloc failed: out of memory" WITHOUT consuming the retry's slot entry, so cudaErrorMemoryAllocation
(non-sticky) stays pending. tensor_cuda.empty would not do: it consumes the entry (bindings.cpp tensor_empty,
P1T-FIX4). A request this size fails at once and takes no device memory.
Variants (each in a fresh process, so the slot starts empty):
  forward : [B operand for nothing] -> OOM -> small K = 0 matmul A (2,0) x B (0,3) -> tc.synchronize() (drain) ->
            read the result. F2 (fill + cuda_check_last) raises the stale error as "CUDA error at matmul zero fill";
            F3 (zeros_checked) returns zeros, and the stale error surfaces at the drain ("CUDA error at sync").
  gradient: B (3,4) nonzero CUDA, requires_grad (from_host consumes the slot before the OOM) -> OOM -> A (0,2,3)
            zero-element -> forward (empty, no CUDA call) -> backward -> drain -> read B's gradient. F2 raises "CUDA
            error at matmul backward zero gradient"; F3 gives B zeros and the stale error surfaces at the drain.

Prior art: the CUDA Runtime API's per-call return status and its separate, per-thread last-error slot
(cudaGetLastError resets it; CUDA 12.6 Runtime API). Ours: the public-API route to a pending error.
"""
import json
import os
import subprocess
import sys

import numpy as np

PKG = {"F2": "tensor_cuda/build-eng4/F2/pkg", "F3": "tensor_cuda/build-eng4/F3/pkg"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def attempt(fn):
    try:
        return {"returned": fn()}
    except Exception as e:  # noqa: BLE001 (recorded)
        return {"raised": {"type": type(e).__name__, "message": str(e)}}


def child(variant):
    import tensor_cuda as tc
    row = {"variant": variant}
    if variant == "gradient":
        b = tc.tensor(np.arange(1, 13, dtype=np.float32).reshape(3, 4), requires_grad=True)
    big_a, big_b = tc.tensor(np.zeros((2 ** 19, 0), np.float32)), tc.tensor(np.zeros((0, 2 ** 19), np.float32))
    row["make_pending"] = attempt(lambda: list(tc.matmul(big_a, big_b).shape))
    if variant == "forward":
        a, s = tc.tensor(np.zeros((2, 0), np.float32)), tc.tensor(np.zeros((0, 3), np.float32))
        holder = {}

        def fwd():
            holder["r"] = tc.matmul(a, s)
            return list(holder["r"].shape)
        row["k0_matmul"] = attempt(fwd)
        row["drain"] = attempt(lambda: tc.synchronize())
        if "r" in holder:
            row["readback"] = attempt(lambda: holder["r"].numpy().view(np.uint32).tolist())
    else:
        a = tc.tensor(np.zeros((0, 2, 3), np.float32), requires_grad=True)
        out = tc.matmul(a, b)

        def bwd():
            out.backward()
            return list(out.shape)
        row["backward"] = attempt(bwd)
        row["drain"] = attempt(lambda: tc.synchronize())
        row["readback"] = attempt(lambda: None if b.grad is None else
                                  {"shape": list(b.grad.shape), "u32": b.grad.numpy().view(np.uint32).tolist()})
    try:
        tc.synchronize()
        row["final_slot_clean"] = True
    except Exception as e:  # noqa: BLE001
        row["final_slot_clean"] = str(e)
    print("ENG4-ROW " + json.dumps(row), flush=True)


def run(outdir):
    os.makedirs(outdir)  # write-once
    rows = []
    for arm in ("F2", "F3"):
        for variant in ("forward", "gradient"):
            env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, PKG[arm]))  # CUDA_VISIBLE_DEVICES untouched
            p = subprocess.run([sys.executable, "-B", "-P", os.path.abspath(__file__), "--child", variant],
                               capture_output=True, text=True, timeout=180, env=env, cwd=ROOT)
            row = {"arm": arm, "variant": variant, "child_exit": p.returncode, "stderr_tail": p.stderr[-1500:]}
            for line in p.stdout.splitlines():
                if line.startswith("ENG4-ROW "):
                    row.update(json.loads(line[len("ENG4-ROW "):]))
            rows.append(row)
            print(arm, variant, p.returncode, flush=True)
    json.dump({"rows": rows}, open(os.path.join(outdir, "pending.json"), "w"), indent=1)
    return 0


def judge(row):
    """Expected outcome per arm and variant; returns (verdict, reason)."""
    made = row.get("make_pending", {}).get("raised", {}).get("message", "")
    if "cudaMalloc failed" not in made:
        return "FAIL", f"the pending error was not made: {row.get('make_pending')}"
    key, label = ("k0_matmul", "matmul zero fill") if row["variant"] == "forward" else \
        ("backward", "matmul backward zero gradient")
    step = row.get(key, {})
    if row["arm"] == "F2":
        msg = step.get("raised", {}).get("message", "")
        ok = label in msg
        return ("PASS" if ok else "FAIL"), f"F2 must raise the stale error under '{label}': {step}"
    if "raised" in step:
        return "FAIL", f"F3 raised at the fill: {step['raised']}"
    rb = row.get("readback", {}).get("returned")
    if row["variant"] == "forward":
        ok = isinstance(rb, list) and len(rb) == 2 and len(rb[0]) == 3 and not any(any(r) for r in rb)
    else:
        ok = isinstance(rb, dict) and rb["shape"] == [3, 4] and not any(any(r) for r in rb["u32"])
    return ("PASS" if ok else "FAIL"), f"F3 returns positive zeros; drain: {row.get('drain')}"


def verdict(outdir):
    rec = json.load(open(os.path.join(outdir, "pending.json")))
    rows = [dict(arm=r["arm"], variant=r["variant"], verdict=judge(r)[0], reason=judge(r)[1]) for r in rec["rows"]]
    gate = [r for r in rows if r["variant"] == "forward"]
    out = {"gate": "K4 forward: F2 raises 'matmul zero fill', F3 returns zeros",
           "verdict": "PASS" if len(gate) == 2 and all(r["verdict"] == "PASS" for r in gate) else "FAIL",
           "gradient_variant": "recorded with the same rule (F2 raises 'matmul backward zero gradient', F3 zeros)",
           "rows": rows}
    json.dump(out, open(os.path.join(outdir, "VERDICT.json"), "w"), indent=1)
    print("K4", out["verdict"])
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2])
    elif sys.argv[1] == "run":
        sys.exit(run(sys.argv[2]))
    elif sys.argv[1] == "verdict":
        sys.exit(verdict(sys.argv[2]))
