import numpy as np

import tensor_cuda as tc


def _run(use_checkpoint):
    x_np = np.array([[-1.0, 0.5, 2.0], [3.0, -0.25, 1.5]], dtype=np.float32)
    w_np = np.array([[0.25, -2.0, 0.75], [1.0, 0.5, -1.5]], dtype=np.float32)
    x = tc.tensor(x_np, requires_grad=True)
    w = tc.tensor(w_np, requires_grad=True)

    def block(z):
        return ((z * w).gelu() + z * 0.25).sum([1], True)

    y = tc.checkpoint(block, x) if use_checkpoint else block(x)
    loss = (y * y).sum()
    loss.backward()
    return loss.numpy(), x.grad.numpy(), w.grad.numpy()


def test_checkpoint_replay_matches_plain_autograd():
    plain_loss, plain_xg, plain_wg = _run(False)
    ckpt_loss, ckpt_xg, ckpt_wg = _run(True)
    assert np.allclose(ckpt_loss, plain_loss, atol=1e-6)
    assert np.allclose(ckpt_xg, plain_xg, atol=1e-6)
    assert np.allclose(ckpt_wg, plain_wg, atol=1e-6)
