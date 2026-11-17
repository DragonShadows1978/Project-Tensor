"""ENG3 L99 seat check (CPU only, CUDA hidden): does tc.tensor([1.0], device="cpu") fail on Build B? Also records
which of G1's named entries exist on the public Python API (introspection of the built module).
Usage: CUDA_VISIBLE_DEVICES= python3 -B -P scripts/eng3_seat_check.py
Prior art: no prior art known to me (a direct reproduction call)."""
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath("tensor_cuda"))
import numpy as np  # noqa: E402
import tensor_cuda as tc  # noqa: E402

print("CUDA_VISIBLE_DEVICES=%r" % os.environ.get("CUDA_VISIBLE_DEVICES"))
print("module:", tc._C.__file__)
for label, fn in [
    ("tc.tensor([1.0], device='cpu')", lambda: tc.tensor([1.0], device="cpu")),
    ("tc.zeros(3, device='cpu')", lambda: tc.zeros(3, device="cpu")),
    ("tc.tensor(np.zeros((0,1)), device='cpu') [empty]", lambda: tc.tensor(np.zeros((0, 1)), device="cpu")),
    ("tc.empty(3, device='cpu')", lambda: tc.empty(3, device="cpu")),
]:
    try:
        t = fn()
        print("OK   ", label, "->", t.shape, t.dtype, t.device)
    except Exception as e:  # report the refusal text verbatim
        print("RAISE", label, "->", type(e).__name__ + ":", e)
print("public API: hasattr(tc, 'full') =", hasattr(tc, "full"))
print("public API: hasattr(tc.Tensor, 'clone') =", hasattr(tc.Tensor, "clone"))
print("public API: hasattr(tc.Tensor, 'to') =", hasattr(tc.Tensor, "to"))
print("public API: hasattr(tc.Tensor, 'cpu') =", hasattr(tc.Tensor, "cpu"), " 'cuda' =", hasattr(tc.Tensor, "cuda"))
print("public API: tc names containing full/clone/copy/to_device:",
      sorted(n for n in dir(tc) if any(k in n.lower() for k in ("full", "clone", "copy", "to_device"))))
print("public API: Tensor names containing full/clone/copy/to:",
      sorted(n for n in dir(tc.Tensor) if any(k in n.lower() for k in ("full", "clone", "copy")) or n in ("to",)))
