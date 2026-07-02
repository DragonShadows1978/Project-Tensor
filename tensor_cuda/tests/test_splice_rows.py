"""splice_rows gate: fused functional cache surgery must match slice+cat."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


def _rope_tables(t, d):
    inv = 1.0 / (1e4 ** (np.arange(0, d, 2, np.float32) / d))
    ang = np.arange(t, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def run():
    old_np = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    ins_np = (np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
              + 1000.0)
    old = tc.tensor(old_np)
    ins = tc.tensor(ins_np)
    with tc.no_grad():
        out = tc.splice_rows(old, ins, 1, 2, 5)
    exp = np.concatenate([old_np[:, :2], ins_np, old_np[:, 5:]], axis=1)
    assert np.array_equal(out.numpy(), exp)

    kpe_np = np.arange(1 * 1 * 8 * 2, dtype=np.float32).reshape(1, 1, 8, 2)
    kpe_ins_np = np.full((1, 1, 2, 2), 22.0, dtype=np.float32)
    kpe = tc.tensor(kpe_np)
    kpe_ins = tc.tensor(kpe_ins_np)
    with tc.no_grad():
        kpe_out = tc.splice_rows(kpe, kpe_ins, 2, 2, 5)
    kpe_exp = np.concatenate([kpe_np[:, :, :2], kpe_ins_np,
                              kpe_np[:, :, 5:]], axis=2)
    assert np.array_equal(kpe_out.numpy(), kpe_exp)

    with tc.no_grad():
        ev = tc.evict_rows(old, 1, 5, 2)
    ev_exp = np.concatenate([old_np[:, :5], old_np[:, 7:]], axis=1)
    assert np.array_equal(ev.numpy(), ev_exp)

    uold_np = (np.arange(1 * 1 * 8 * 4, dtype=np.uint8).reshape(1, 1, 8, 4)
               + 1)
    uins_np = np.full((1, 1, 3, 4), 251, dtype=np.uint8)
    uold = tc.tensor(uold_np, dtype="uint8")
    uins = tc.tensor(uins_np, dtype="uint8")
    with tc.no_grad():
        uout = tc.splice_rows(uold, uins, 2, 2, 5)
    uexp = np.concatenate([uold_np[:, :, :2], uins_np,
                           uold_np[:, :, 5:]], axis=2)
    assert np.array_equal(uout.numpy(), uexp)

    try:
        tc.splice_rows(old, ins, 1, 2, 5)
        raise AssertionError("grad guard missing")
    except RuntimeError:
        pass

    cs_np, sn_np = _rope_tables(16, 64)
    cs, sn = tc.tensor(cs_np), tc.tensor(sn_np)
    raw0_np = np.arange(1 * 8 * 5, dtype=np.float32).reshape(1, 8, 5)
    raw1_np = raw0_np + 200.0
    rins0_np = np.full((1, 2, 5), 7.0, dtype=np.float32)
    rins1_np = np.full((1, 2, 5), 9.0, dtype=np.float32)
    rope0_np = np.random.default_rng(3).standard_normal(
        (1, 1, 8, 64)).astype(np.float32)
    rope1_np = rope0_np * 0.5
    pins0_np = np.random.default_rng(4).standard_normal(
        (1, 1, 2, 64)).astype(np.float32)
    pins1_np = pins0_np * -0.25
    with tc.no_grad():
        raw0, raw1 = tc.tensor(raw0_np), tc.tensor(raw1_np)
        rins0, rins1 = tc.tensor(rins0_np), tc.tensor(rins1_np)
        rope0, rope1 = tc.tensor(rope0_np), tc.tensor(rope1_np)
        pins0, pins1 = tc.tensor(pins0_np), tc.tensor(pins1_np)
        raw_sw, rope_sw = tc.swap_row_pairs_with_rope(
            [raw0, raw1], [rope0, rope1], [rins0, rins1], [pins0, pins1],
            cs, sn, 1, 2, 2, 5, 2)
        pin0_rot = tc.rope_apply(pins0, cs, sn, 2)
        pin1_rot = tc.rope_apply(pins1, cs, sn, 2)
        raw_ev, rope_ev = tc.evict_row_pairs(
            [raw0, raw1], [rope0, rope1], 1, 2, 5, 2)
        tx_raw, tx_rope, tx_mount = tc.arena_row_pair_transaction(
            [raw0, raw1], [rope0, rope1], [rins0, rins1], [pins0, pins1],
            cs, sn, 1, 2, 2, 3, 4)
        ev_tx_raw, ev_tx_rope, ev_tx_mount = tc.arena_row_pair_transaction(
            [raw0, raw1], [rope0, rope1], [], [], cs, sn, 1, 2, 2, 3, 4)
    np.testing.assert_array_equal(
        raw_sw[0].numpy(),
        np.concatenate([raw0_np[:, :2], rins0_np, raw0_np[:, 5:]], axis=1))
    np.testing.assert_array_equal(
        raw_sw[1].numpy(),
        np.concatenate([raw1_np[:, :2], rins1_np, raw1_np[:, 5:]], axis=1))
    np.testing.assert_allclose(
        rope_sw[0].numpy(),
        np.concatenate([rope0_np[:, :, :2], pin0_rot.numpy(),
                        rope0_np[:, :, 5:]], axis=2),
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        rope_sw[1].numpy(),
        np.concatenate([rope1_np[:, :, :2], pin1_rot.numpy(),
                        rope1_np[:, :, 5:]], axis=2),
        rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        raw_ev[0].numpy(), np.concatenate([raw0_np[:, :5],
                                           raw0_np[:, 7:]], axis=1))
    np.testing.assert_array_equal(
        raw_ev[1].numpy(), np.concatenate([raw1_np[:, :5],
                                           raw1_np[:, 7:]], axis=1))
    np.testing.assert_array_equal(
        rope_ev[0].numpy(), np.concatenate([rope0_np[:, :, :5],
                                            rope0_np[:, :, 7:]], axis=2))
    np.testing.assert_array_equal(
        rope_ev[1].numpy(), np.concatenate([rope1_np[:, :, :5],
                                            rope1_np[:, :, 7:]], axis=2))
    assert tx_mount == 2
    np.testing.assert_array_equal(tx_raw[0].numpy(), raw_sw[0].numpy())
    np.testing.assert_array_equal(tx_raw[1].numpy(), raw_sw[1].numpy())
    np.testing.assert_array_equal(tx_rope[0].numpy(), rope_sw[0].numpy())
    np.testing.assert_array_equal(tx_rope[1].numpy(), rope_sw[1].numpy())
    assert ev_tx_mount == 0
    np.testing.assert_array_equal(
        ev_tx_raw[0].numpy(), np.concatenate([raw0_np[:, :2],
                                              raw0_np[:, 5:]], axis=1))
    np.testing.assert_array_equal(
        ev_tx_raw[1].numpy(), np.concatenate([raw1_np[:, :2],
                                              raw1_np[:, 5:]], axis=1))
    np.testing.assert_array_equal(
        ev_tx_rope[0].numpy(), np.concatenate([rope0_np[:, :, :2],
                                               rope0_np[:, :, 5:]], axis=2))
    np.testing.assert_array_equal(
        ev_tx_rope[1].numpy(), np.concatenate([rope1_np[:, :, :2],
                                               rope1_np[:, :, 5:]], axis=2))
    try:
        with tc.no_grad():
            tc.arena_row_pair_transaction(
                [raw0], [rope0], [rins0], [pins0], cs, sn, 1, 2, 2, 3, 1)
        raise AssertionError("arena width guard missing")
    except RuntimeError:
        pass
    try:
        tc.swap_row_pairs_with_rope(
            [raw0], [rope0], [rins0], [pins0], cs, sn, 1, 2, 2, 5, 2)
        raise AssertionError("pair swap grad guard missing")
    except RuntimeError:
        pass
    try:
        tc.evict_row_pairs([raw0], [rope0], 1, 2, 5, 2)
        raise AssertionError("pair evict grad guard missing")
    except RuntimeError:
        pass
    try:
        tc.arena_row_pair_transaction(
            [raw0], [rope0], [rins0], [pins0], cs, sn, 1, 2, 2, 3, 4)
        raise AssertionError("arena transaction grad guard missing")
    except RuntimeError:
        pass
    print("SPLICE_ROWS GATE: PASS")


if __name__ == "__main__":
    run()
