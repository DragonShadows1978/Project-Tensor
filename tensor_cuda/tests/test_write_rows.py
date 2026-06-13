"""write_rows gate: in-place ring writes must land exactly (in-order
and wrapped), reject autograd, and leave untouched rows untouched."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


def run():
    buf = tc.tensor(np.zeros((2, 8, 4), np.float32))
    src = tc.tensor(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    with tc.no_grad():
        tc.write_rows(buf, src, 0)
        b = buf.numpy()
        assert np.array_equal(b[:, :3], src.numpy())
        assert b[:, 3:].sum() == 0
        tc.write_rows(buf, src, 6)                   # wrap: rows 6,7,0
        b = buf.numpy()
        assert np.array_equal(b[:, 6], src.numpy()[:, 0])
        assert np.array_equal(b[:, 7], src.numpy()[:, 1])
        assert np.array_equal(b[:, 0], src.numpy()[:, 2])
        assert np.array_equal(b[:, 1:3], src.numpy()[:, 1:])  # old rows kept

        # bf16 path + single-row append (the decode shape)
        rb = tc.tensor(np.zeros((1, 8, 1024, 64), np.float32)).astype("bfloat16")
        row = tc.tensor(np.ones((1, 8, 1, 64), np.float32)).astype("bfloat16")
        tc.write_rows(rb, row, 1023)
        assert float(rb.float().numpy()[:, :, 1023].sum()) == 8 * 64
        assert float(rb.float().numpy()[:, :, :1023].sum()) == 0
    try:
        tc.write_rows(buf, src, 0)
        raise AssertionError("grad guard missing")
    except RuntimeError:
        pass
    print("WRITE_ROWS GATE: PASS")


if __name__ == "__main__":
    run()
