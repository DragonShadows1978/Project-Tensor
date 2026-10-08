"""APA-MF2.2c G4 diagnosis: the review's guards/*/NaN_softmax_policy case with per-row values,
plus the softmax primitive on the same rows (no guard). Mine."""
import numpy as np, tensor_cuda as tc, hashlib
from tensor_cuda import functional as F
print("engine", hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest()[:12], "functional", hashlib.sha256(open(F.__file__, "rb").read()).hexdigest()[:12])
for dt in ("float32", "float16", "bfloat16"):
    data = np.full((1, 1, 3, 9), -np.inf, np.float32)
    data[..., 1, 0] = np.nan; data[..., 2, 0] = 0
    s = tc.tensor(data, dtype=dt)
    with tc.no_grad():
        w = F._guarded_softmax(s, lambda x: x.softmax(-1)).float().numpy()
        p = s.softmax(-1).float().numpy()
        blk = F._blocked_rows(s).float().numpy()
    print(dt, "blocked", blk.ravel().tolist())
    for r in range(3):
        print(f"  row{r} guarded={w[0,0,r,:3].tolist()}  primitive={p[0,0,r,:3].tolist()}")
    print("  probe asserts: row0==0:", bool((w[..., 0, :] == 0).all()), " row1 all NaN:", bool(np.isnan(w[..., 1, :]).all()),
          " row2 one-hot:", bool(np.array_equal(w[..., 2, :], np.array([1] + [0] * 8, np.float32))))
# shape check: the probe compares got[...,2,:] (shape (1,1,9)) with a (9,) array via np.array_equal
for dt in ("float32", "float16", "bfloat16"):
    data = np.full((1, 1, 3, 9), -np.inf, np.float32); data[..., 1, 0] = np.nan; data[..., 2, 0] = 0
    with tc.no_grad():
        w = F._guarded_softmax(tc.tensor(data, dtype=dt), lambda x: x.softmax(-1)).float().numpy()
    exp = np.array([1] + [0] * 8, np.float32)
    print(dt, "row2", w[..., 2, :].tolist(), "shape", w[..., 2, :].shape, "vs", exp.shape,
          "| array_equal as written:", np.array_equal(w[..., 2, :], exp), "| raveled:", np.array_equal(w[..., 2, :].ravel(), exp),
          "| numpy array_equal((1,1,9) ones, (9,) ones):", np.array_equal(np.ones((1, 1, 9)), np.ones(9)))
