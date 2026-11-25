"""BWD-F3 P2 case table (imported by scripts/bwdf3_step_p2.py inside the lease, and by scripts/bwdf3_p2_drive.py on
the CPU for the L48 geometry drive). No engine import and no CUDA call at import time: run_row() receives the engine
module and, optionally, the operand factories (the CPU drive substitutes host tensors for CUDA tensors with data).
See scripts/bwdf3_step_p2.py for the modes. Prior art: see scripts/bwdf3_step_p2.py; ours: the cases.
"""
import bwdf3_gpu_common as G

P61 = 2 ** 61
MODES = ("e1", "e2", "e3", "count")


def rows_for(mode):
    """(row id, kind): kind 'refuse' (must raise before any work on Build F3), 'accept' (returns), 'control'."""
    if mode == "e1":
        return [("e1-control-rope_apply-covering", "control"), ("e1-control-export-covering", "control"),
                ("e1-export-reading-short-sin", "refuse"), ("e1-export-reading-negative-pos0", "refuse"),
                ("e1-export-reading-short-cos", "refuse"), ("e1-export_row_pairs-reading-short-sin", "refuse"),
                ("e1-swap-reading-short-sin", "refuse"), ("e1-arena-reading-short-sin", "refuse"),
                ("e1-rope_apply-empty-cuda-short-sin", "accept"), ("e1-export-empty-cuda-short-sin", "accept"),
                ("e1-swap-empty-cuda-short-sin", "accept")]
    if mode == "e2":
        return [("e2-rope_apply-sin-T-D-1", "accept"), ("e2-rope_apply-cos-T-D-1-1", "accept"),
                ("e2-export-sin-T-D-1", "accept"), ("e2-swap-tables-T-D-1", "accept"),
                ("e2-refused-sin-T-D-2", "refuse")]
    if mode == "e3":
        return [("e3-control-sum-last", "control"), ("e3-control-sum-two-distinct", "control"),
                ("e3-control-mean", "control"), ("e3-control-argmax", "control"),
                ("e3-sum-duplicate", "refuse"), ("e3-sum-out-of-range", "refuse"), ("e3-mean-duplicate", "refuse"),
                ("e3-softmax-out-of-range", "refuse"), ("e3-argmax-out-of-range", "refuse")]
    return [("control-swap-valid-pairs", "control"), ("control-sum-valid", "control"),
            ("e4-swap-pair1-bound-after-work", "refuse"), ("e4-arena-layer1-bound-after-work", "refuse"),
            ("e3-sum-duplicate-count", "refuse"), ("e1-rope_apply-empty-cuda-count", "accept")]


def run_row(tc, mode, rid, T=None, E=None):
    """Returns (operands builder, call). T / E: operand factories (default: CUDA tensors with data / empty)."""
    import numpy as np

    def T_cuda(shape, dt="float32", salt=0):
        if dt == "uint8":
            return tc.tensor((np.arange(int(np.prod(shape))) % 251).astype(np.uint8).reshape(shape), device="cuda",
                             dtype=dt)
        return tc.tensor(G.data(shape, dt, salt), device="cuda", dtype=dt)

    def E_cuda(shape, dt="float32"):
        assert int(np.prod(shape)) == 0, shape
        return tc.tensor(np.zeros(shape, np.uint8 if dt == "uint8" else np.float32), device="cuda", dtype=dt)

    T = T or T_cuda
    E = E or E_cuda

    def rb(t):
        return np.ascontiguousarray(t.astype("float32").numpy())

    def tup(r):
        return {f"o{i}_{j}": rb(x) for i, part in enumerate(r) for j, x in enumerate(part)}

    NG = tc.no_grad
    if mode == "e1":
        tab = lambda: (T((8, 2), salt=1), T((8, 2), salt=2))
        if rid == "e1-control-rope_apply-covering":
            return lambda: (T((3, 2, 4)), T((6, 4), salt=1), T((6, 4), salt=2)), \
                lambda o: {"out": rb(tc.rope_apply(o[0], o[1], o[2], 1))}
        if rid == "e1-control-export-covering":
            return lambda: (T((1, 4, 2), salt=3),) + tab(), \
                lambda o: {"out": rb(tc.export_rope_rows(o[0], o[1], o[2], 1, 1, 2, 3))}
        if rid == "e1-export-reading-short-sin":
            return lambda: (T((1, 4, 2), salt=3), T((8, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: tc.export_rope_rows(o[0], o[1], o[2], 1, 0, 2, 0)
        if rid == "e1-export-reading-negative-pos0":
            return lambda: (T((1, 4, 2), salt=3),) + tab(), lambda o: tc.export_rope_rows(o[0], o[1], o[2], 1, 0, 1, -1)
        if rid == "e1-export-reading-short-cos":
            return lambda: (T((1, 4, 2), salt=3), T((1, 2), salt=1), T((8, 2), salt=2)), \
                lambda o: tc.export_rope_rows(o[0], o[1], o[2], 1, 0, 2, 0)
        if rid == "e1-export_row_pairs-reading-short-sin":
            return lambda: ([T((1, 4, 2), salt=4)], [T((1, 4, 2), salt=5)], T((8, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: tc.export_row_pairs(o[0], o[1], o[2], o[3], 1, 1, [0], [0], 2, 0)
        if rid == "e1-swap-reading-short-sin":
            return lambda: ([T((1, 4, 2), salt=4)], [T((1, 4, 2), salt=5)], [T((1, 2, 2), salt=6)],
                            [T((1, 2, 2), salt=7)], T((2, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 0, 0)
        if rid == "e1-arena-reading-short-sin":
            return lambda: ([T((1, 4, 2), salt=4)], [T((1, 4, 2), salt=5)], [T((1, 2, 2), salt=6)],
                            [T((1, 2, 2), salt=7)], T((2, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: tc.arena_row_pair_transaction(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 0, 4)
        if rid == "e1-rope_apply-empty-cuda-short-sin":
            return lambda: (E((0, 2, 2)), T((2, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: {"shape": np.array(tc.rope_apply(o[0], o[1], o[2], 0).shape, np.float32)}
        if rid == "e1-export-empty-cuda-short-sin":
            return lambda: (E((0, 4, 2)), T((2, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: {"shape": np.array(tc.export_rope_rows(o[0], o[1], o[2], 1, 0, 2, 0).shape, np.float32)}
        if rid == "e1-swap-empty-cuda-short-sin":
            return lambda: ([E((0, 2, 2))], [E((0, 2, 2))], [E((0, 2, 2))], [E((0, 2, 2))], T((2, 2), salt=1),
                            T((1, 2), salt=2)), \
                lambda o: {"n": np.array([len(tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1,
                                                                          0, 0, 0)[1])], np.float32)}
    if mode == "e2":
        # Each accepted row returns the result with the widened table(s) ("out") and the same call with [T, D] tables
        # ("ref"), in the same process and build: identical kernel, identical storage, so bitwise equal.
        if rid == "e2-rope_apply-sin-T-D-1":
            return lambda: (T((3, 2, 4)), T((6, 4), salt=1), T((6, 4), salt=2)), \
                lambda o: {"out": rb(tc.rope_apply(o[0], o[1], o[2].reshape([6, 4, 1]), 1)),
                           "ref": rb(tc.rope_apply(o[0], o[1], o[2], 1))}
        if rid == "e2-rope_apply-cos-T-D-1-1":
            return lambda: (T((3, 2, 4)), T((6, 4), salt=1), T((6, 4), salt=2)), \
                lambda o: {"out": rb(tc.rope_apply(o[0], o[1].reshape([6, 4, 1, 1]), o[2], 1)),
                           "ref": rb(tc.rope_apply(o[0], o[1], o[2], 1))}
        if rid == "e2-export-sin-T-D-1":
            return lambda: (T((2, 5, 4), salt=3), T((8, 4), salt=1), T((8, 4), salt=2)), \
                lambda o: {"out": rb(tc.export_rope_rows(o[0], o[1], o[2].reshape([8, 4, 1]), 1, 1, 3, 2)),
                           "ref": rb(tc.export_rope_rows(o[0], o[1], o[2], 1, 1, 3, 2))}
        if rid == "e2-swap-tables-T-D-1":
            def call(o):
                a = tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4].reshape([8, 2, 1]),
                                                o[5].reshape([8, 2, 1, 1]), 1, 1, 1, 2, 1)
                b = tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 1, 2, 1)
                return {"out": np.concatenate([v.reshape(-1) for v in tup(a).values()]),
                        "ref": np.concatenate([v.reshape(-1) for v in tup(b).values()])}
            return lambda: ([T((1, 4, 2), salt=4)], [T((1, 4, 2), salt=5)], [T((1, 2, 2), salt=6)],
                            [T((1, 2, 2), salt=7)], T((8, 2), salt=1), T((8, 2), salt=2)), call
        if rid == "e2-refused-sin-T-D-2":
            return lambda: (T((1, 2, 4)), T((4, 4), salt=1), T((4, 4, 2), salt=2)), \
                lambda o: tc.rope_apply(o[0], o[1], o[2], 0)
    if mode == "e3":
        if rid == "e3-control-sum-last":
            return lambda: (T((3, 5)),), lambda o: {"out": rb(o[0].sum([-1]))}
        if rid == "e3-control-sum-two-distinct":
            return lambda: (T((2, 3, 4)),), lambda o: {"out": rb(o[0].sum([2, 0]))}
        if rid == "e3-control-mean":
            return lambda: (T((2, 3)),), lambda o: {"out": rb(o[0].mean([0, 1]))}
        if rid == "e3-control-argmax":
            return lambda: (T((3, 7)),), lambda o: {"out": o[0].argmax(-1).numpy().astype(np.float32)}
        if rid == "e3-sum-duplicate":
            return lambda: (T((2,)),), lambda o: {"out": rb(o[0].sum([0, 0]))}
        if rid == "e3-sum-out-of-range":
            return lambda: (T((2,)),), lambda o: {"out": rb(o[0].sum([8]))}
        if rid == "e3-mean-duplicate":
            return lambda: (T((2, 3)),), lambda o: {"out": rb(o[0].mean([0, 0]))}
        if rid == "e3-softmax-out-of-range":
            return lambda: (T((2, 3)),), lambda o: {"out": rb(o[0].softmax(-3))}
        if rid == "e3-argmax-out-of-range":
            return lambda: (T((2, 3)),), lambda o: {"out": o[0].argmax(2).numpy().astype(np.float32)}
    if mode == "count":
        if rid == "control-swap-valid-pairs":
            return lambda: ([T((1, 4, 2), "uint8"), T((1, 4, 2), "uint8")], [E((1, 4, 0)), E((1, 4, 0))],
                            [T((1, 1, 2), "uint8"), T((1, 1, 2), "uint8")], [E((1, 1, 0)), E((1, 1, 0))],
                            E((1, 0)), E((1, 0))), \
                lambda o: tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 0, 0)
        if rid == "control-sum-valid":
            return lambda: (T((2, 3)),), lambda o: o[0].sum([-1])
        if rid == "e4-swap-pair1-bound-after-work":
            # pair 0 allocates and launches (a [1, 5, 2] uint8 splice); pair 1's output [0, 2^62, 2] uint8 is 2^63
            # bytes counting the zero as 1, which the NDArray constructor refuses.
            return lambda: ([T((1, 4, 2), "uint8"), E((0, P61, 2), "uint8")], [E((1, 4, 0)), E((1, 4, 0))],
                            [T((1, 1, 2), "uint8"), E((0, P61, 2), "uint8")], [E((1, 1, 0)), E((1, 1, 0))],
                            E((1, 0)), E((1, 0))), \
                lambda o: tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 0, 0)
        if rid == "e4-arena-layer1-bound-after-work":
            # every layer mounts 1 token (the arena's rule); layer 0 allocates and launches, layer 1's cache
            # [0, 2^62 - 1, 2] uint8 fits but its output [0, 2^62, 2] does not.
            return lambda: ([T((1, 4, 2), "uint8"), E((0, 2 * P61 - 1, 2), "uint8")], [E((1, 4, 0)), E((1, 4, 0))],
                            [T((1, 1, 2), "uint8"), E((0, 1, 2), "uint8")], [E((1, 1, 0)), E((1, 1, 0))],
                            E((1, 0)), E((1, 0))), \
                lambda o: tc.arena_row_pair_transaction(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 0, 4)
        if rid == "e3-sum-duplicate-count":
            return lambda: (T((2,)),), lambda o: o[0].sum([0, 0])
        if rid == "e1-rope_apply-empty-cuda-count":
            return lambda: (E((0, 2, 2)), T((2, 2), salt=1), T((1, 2), salt=2)), \
                lambda o: tc.rope_apply(o[0], o[1], o[2], 0)
    raise ValueError((mode, rid))
