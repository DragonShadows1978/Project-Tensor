"""ENG3 G3 device-path dump (GPU): the device paths of every E2 member on one shape per dtype, as raw storage bytes,
for the installed build. Compared across builds by scripts/eng3_g3_compare.py (Build Bh vs Build E for every member;
Build B vs Build E for the members Build B's public API reaches).

  python3 -B -P scripts/eng3_devpath.py <label> hooks|public

hooks  (Bh, E): from_host, to_host, zeros, full, clone, to (device->device, host->device, device->host), astype (all 18
       A1.4 pairs), read as raw bytes through _core_to(t, "cpu") + ctypes (independent of .numpy()); also .numpy().
public (B, E): from_host + .numpy() (non-bool), device astype through .numpy(), backward() on a device leaf (full seed +
       clone in accumulate_grad).
Output: artifacts/eng3/devpath/<label>.npz (+ .json with key list and module sha256).

Prior art: no prior art known to me (a byte dump of each member's output).
"""
import ctypes
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda"))
import numpy as np  # noqa: E402
import tensor_cuda as tc  # noqa: E402

FLOATS = ("float32", "float16", "bfloat16")
ALL = FLOATS + ("uint8", "int64", "bool")
ITEM = {"float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1, "int64": 8, "bool": 1}
SHAPE = (3, 5)
rng = np.random.default_rng(20261116)


def data(dt):
    n = 15
    if dt in ("float32", "bfloat16"):
        v = rng.standard_normal(n).astype(np.float32) * 100
        v[:6] = [0.0, -0.0, 65520.0, 1e-30, 3.0e38, 0.5]
        return v.reshape(SHAPE)
    if dt == "float16":
        v = (rng.standard_normal(n) * 100).astype(np.float16)
        v[:4] = [0.0, -0.0, 65504.0, 6e-8]
        return v.reshape(SHAPE)
    if dt == "uint8":
        return rng.integers(0, 256, n).astype(np.uint8).reshape(SHAPE)
    if dt == "int64":
        return rng.integers(-2**62, 2**62, n).astype(np.int64).reshape(SHAPE)
    return (rng.integers(0, 2, n) == 1).reshape(SHAPE)


def raw(t):
    h = t if t.device == "cpu" else tc._C._core_to(t, "cpu")
    n = int(np.prod(h.shape)) * ITEM[h.dtype]
    return np.frombuffer(ctypes.string_at(h.data_ptr(), n), np.uint8).copy()


def main():
    label, mode = sys.argv[1], sys.argv[2]
    out = {}
    src = {dt: data(dt) for dt in ALL}
    for dt in ALL:
        if mode == "public" and dt == "bool":
            continue  # bool .numpy() is the A2.2 waived row
        d = tc.tensor(src[dt], device="cuda", dtype=dt)
        out[f"from_host+numpy/{dt}"] = np.ascontiguousarray(d.numpy()).view(np.uint8).ravel()
        for d2 in FLOATS + ("uint8",):
            if dt in ("int64", "bool") and d2 != dt:
                continue
            o = d.astype(d2)
            out[f"astype_numpy/{dt}->{d2}"] = np.ascontiguousarray(o.numpy()).view(np.uint8).ravel()
    if mode == "public":
        for dt in FLOATS:
            x = tc.tensor(src[dt], device="cuda", dtype=dt).detach()
            x.requires_grad = True
            x.backward()
            out[f"backward_full_clone/{dt}"] = np.ascontiguousarray(x.grad.numpy()).view(np.uint8).ravel()
    else:
        C = tc._C
        for dt in ALL:
            d = tc.tensor(src[dt], device="cuda", dtype=dt)
            h = tc.tensor(src[dt], device="cpu", dtype=dt) if dt != "bfloat16" else C._core_to(d, "cpu")
            out[f"from_host_raw/{dt}"] = raw(d)
            out[f"zeros/{dt}"] = raw(C._core_zeros(list(SHAPE), dt, "cuda"))
            if dt in FLOATS:
                for v in (1.5, -0.0, 65520.0, float("nan"), 1e-8, 3.0e38):
                    out[f"full/{dt}/{v}"] = raw(C._core_full(list(SHAPE), v, dt, "cuda"))
            out[f"clone/{dt}"] = raw(C._core_clone(d))
            out[f"to_d2d/{dt}"] = raw(C._core_to(d, "cuda"))
            out[f"to_h2d/{dt}"] = raw(C._core_to(h, "cuda"))
            out[f"to_d2h/{dt}"] = raw(C._core_to(d, "cpu"))
            for d2 in FLOATS + ("uint8",):
                if dt in ("int64", "bool") and d2 != dt:
                    continue
                out[f"astype_raw/{dt}->{d2}"] = raw(d.astype(d2))
            if dt == "bool":
                out["bool_numpy_waived/bool"] = np.ascontiguousarray(d.numpy()).view(np.uint8).ravel()
    tc.synchronize()
    os.makedirs(os.path.join(ROOT, "artifacts/eng3/devpath"), exist_ok=True)
    path = os.path.join(ROOT, "artifacts/eng3/devpath", label + ".npz")
    np.savez(path, **{k.replace("/", "|"): v for k, v in out.items()})
    msha = hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest()
    json.dump({"label": label, "mode": mode, "module_sha256": msha, "keys": sorted(out)},
              open(path[:-4] + ".json", "w"), indent=1)
    print("devpath", label, mode, len(out), "keys; module", msha)


if __name__ == "__main__":
    main()
