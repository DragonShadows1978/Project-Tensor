"""ENG3 L60: the E4 refusal (fill_ on a host tensor) on the installed build, CUDA hidden; prints a JSON row.
Prior art: no prior art known to me."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.abspath("tensor_cuda"))
import numpy as np
import tensor_cuda as tc
row = {"module_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
       "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
for label, mk in (("empty host tensor", lambda: tc.tensor(np.zeros((0,), np.float32), device="cpu")),
                  ("3-element host tensor", lambda: tc.tensor([1.0, 2.0, 3.0], device="cpu"))):
    try:
        t = mk()
    except Exception as e:
        row[label] = f"tensor creation raised {type(e).__name__}: {e}"
        continue
    try:
        t.fill_(5.0)
        row[label] = "fill_ returned (no refusal)"
    except Exception as e:
        row[label] = f"fill_ raised {type(e).__name__}: {e}; values after: {t.numpy().tolist()}"
print(json.dumps(row))
