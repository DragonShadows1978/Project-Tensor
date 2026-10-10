"""APA-REGT2: packed identity-V (pIdV) leak regression tests -- detector logic on numpy kernels (CPU) and the real
engine (GPU) at APA-LD2's small edge geometries and at the production layout (B 2, Hq 16, KVH 8, D 128, S 8192).
Author-run tests: a baseline, not blind verification.

pIdV (scripts/apa_pidv.py, ported from APA-LD3 @ 24f8b11): V in {0, 1}; cut-indicator columns (one per row boundary
of the true causal visibility, in every slab) and interleaved slab-colour columns; every output entry must be
EXACTLY zero or EXACTLY nonzero as the true visibility and head map say. No reference run, no repeat run, no
determinism assumption, no tolerance.

Engine cells (each registered with an in-test positive control, so no PASS is vacuous):
  * edge cells: APA-LD2 / LD3 Group E hosts with S <= 1025 (QW (64,1025), QW (3,1025), QW / MC3 (3,33), G4G (3,33),
    G4G (3,1025)) for sdpa, the sdpa fused causal softmax, apa_quant, apa_selective (fused), apa_int4 and apa_sp; plus
    one fp32 cell for each kernel APA-LD3 ran only in 16-bit (R01-R03). apa_sp is fp32 only (its registered margin
    table covers fp32 at scale 1/sqrt(D)). Control: the Python offby1 mutant (one extra future key) must ALARM.
  * production-layout cells: sdpa and apa_selective at S = 8192, L in {1, 64}, bf16 and fp32; L = 1 runs
    apa_selective's split-K decode path, L = 64 its fused kernel. Control: the Python bstride mutant (batch stride
    off by one) must ALARM.
  * S = 98,304 (apa_selective bf16, L = 64): behind TC_SLOW_TESTS=1. Measured on main: about 26 s and 10.7 GB peak
    host RSS with its control, which does not fit the 30 s default-suite budget.

Registered known-bad builds these tests must fail on (APA-REGT2 G3): LD1 CU1 (GQA tile head map), CU3 (wrong batch
V), CU4 (V batch stride off by one); LD2 CU5 (one extra future key), CU6 (single tile-edge row); LD3 I1 (interior
tile reads KV head +1), I2 (interior tile reads the other batch, split-K), I3 (I1 on one slab only).

NOT covered: in-slab reordering of visible keys (pIdV is blind to it BY DESIGN: a row that reads the right slab but
permutes keys inside its true visible set gives the same zero/nonzero pattern; pinned by
test_pidv_blind_to_in_slab_reordering_of_visible_keys); numerics, selection and statistics paths; backward passes;
fp16 tiny-weight underflow beyond the registered cells; the GRM gemma4 sliding path (production tests do not import
GRM).

Availability guard (APA-REGT1 review nit F01): skip ONLY on a recognised no-device condition -- the extension cannot
be imported (ImportError naming tensor_cuda / _tensor_cuda), or the engine reports "no CUDA-capable device". Any
other exception from the probe propagates and the test FAILS.

Prior art: Dorfman (1943) group testing and non-adaptive combinatorial group testing (Du & Hwang): set-indicator
columns are group tests; identity-V (APA-LD1 / LD2): the exact zero/nonzero oracle; DeMillo, Lipton & Sayward (1978)
mutation testing: the known-bad builds and in-test mutants; Dao (2023) FlashAttention-2: bottom-right causal.
Unverified -- lead to check. Mine (APA-LD3): the pIdV design. Mine (APA-REGT2): the port and the cell tables.
"""
import importlib
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_ld1_pdwl as P  # noqa: E402
import apa_ld2_pdwl as P2  # noqa: E402
import apa_pidv as M  # noqa: E402

NO_DEVICE = "no CUDA-capable device"
EXTENSION_MODULES = frozenset({"tensor_cuda", "_tensor_cuda", "tensor_cuda._tensor_cuda"})
SLOW = os.environ.get("TC_SLOW_TESTS") == "1"

# (cell id, family, dtype, geo, L, S). A-ids: APA-LD2 / LD3 Group E hosts (several A cells share a host: A02 = A20,
# A06 = A22, A09 = A23 = A26 = A30). R-ids: APA-REGT2 fp32 cells for kernels LD3 ran only in 16-bit.
EDGE_CELLS = [
    ("A01", "sdpa", "float32", "QW", 64, 1025),
    ("A02", "sdpa", "bfloat16", "QW", 64, 1025),
    ("A03", "sdpa", "float16", "QW", 64, 1025),
    ("A04", "sdpa", "bfloat16", "MC3", 3, 33),
    ("A05", "sdpa_fused_cs", "bfloat16", "QW", 64, 1025),
    ("R01", "sdpa_fused_cs", "float32", "QW", 64, 1025),
    ("A06", "apa_quant", "bfloat16", "QW", 64, 1025),
    ("A07", "apa_quant", "float16", "QW", 3, 1025),
    ("R02", "apa_quant", "float32", "QW", 64, 1025),
    ("A08", "apa_sel", "float32", "QW", 64, 1025),
    ("A09", "apa_sel", "bfloat16", "QW", 64, 1025),
    ("A10", "apa_sel", "float16", "G4G", 3, 33),
    ("A24", "apa_sel", "float32", "G4G", 3, 1025),
    ("A27", "apa_sel", "float32", "QW", 3, 33),
    ("A31", "apa_sel", "float16", "G4G", 3, 1025),
    ("A13", "apa_int4", "bfloat16", "QW", 64, 1025),
    ("A25", "apa_int4", "float16", "QW", 64, 1025),
    ("R03", "apa_int4", "float32", "QW", 64, 1025),
    ("A15", "apa_sp", "float32", "QW", 64, 1025),
]
# (cell id, family, dtype, L): production layout QW at S = 8192. W / I ids: APA-LD3 Group W / I hosts.
PROD_S = 8192
PROD_CELLS = [
    ("W16", "sdpa", "bfloat16", 64),
    ("W18", "sdpa", "bfloat16", 1),
    ("W20", "sdpa", "float32", 64),
    ("W21", "apa_sel", "bfloat16", 64),     # fused kernel; host of W01 / W06 / W11 / I01 / I09 (CU1 CU3 CU4 I1 I3)
    ("W23", "apa_sel", "bfloat16", 1),      # split-K decode; host of I05 (I2)
    ("W25", "apa_sel", "float32", 64),      # fused kernel; host of W03 / W08 / W13 / I03 / I11
    ("I07", "apa_sel", "float32", 1),       # split-K decode, fp32 (I2)
]
SLOW_CELL = ("I02", "apa_sel", "bfloat16", 64, 98304)   # host of W02 / W07 / W12 / I02 / I10


# ---------------------------------------------------------------- availability guard (narrow; F01)
def _import_engine(import_module=importlib.import_module):
    """Import tensor_cuda; skip only when the extension itself cannot be imported."""
    try:
        return import_module("tensor_cuda")
    except ImportError as exc:
        if exc.name in EXTENSION_MODULES:
            pytest.skip(f"tensor_cuda extension not importable: {exc}")
        raise


def _require_device(tc):
    """tensor_cuda imports without a CUDA device and fails at the first allocation; skip only on the engine's
    no-device error, let every other failure propagate."""
    try:
        tc.zeros((1,), dtype="float32").numpy()
    except RuntimeError as exc:
        if NO_DEVICE in str(exc):
            pytest.skip(f"CUDA unavailable: {exc}")
        raise


def _engine():
    tc = _import_engine()
    _require_device(tc)
    from tensor_cuda import functional as F
    from tensor_cuda import quant as Q
    M.bind(tc, F, Q)
    return M


@pytest.fixture(scope="module")
def margins_dir(tmp_path_factory):
    """caller-owned directory for apa_sp's synthetic margin table (pytest's tmp area, never the repo)."""
    return tmp_path_factory.mktemp("apa_pidv_sp_margins")


class _ProbeRaises:
    def __init__(self, exc):
        self.exc = exc

    def zeros(self, *a, **kw):
        raise self.exc


def _raising_importer(exc):
    def imp(name):
        raise exc
    return imp


def test_availability_guard_skips_only_on_no_device_conditions():
    with pytest.raises(pytest.skip.Exception):
        _require_device(_ProbeRaises(RuntimeError("cudaMalloc failed: no CUDA-capable device is detected")))
    for exc in (RuntimeError("cudaMalloc failed: out of memory"),
                RuntimeError("an illegal memory access was encountered"),
                ValueError("no CUDA-capable device is detected"),        # not the engine's error type
                AttributeError("module 'tensor_cuda' has no attribute 'zeros'")):
        with pytest.raises(type(exc)):
            _require_device(_ProbeRaises(exc))
    for name in sorted(EXTENSION_MODULES):
        with pytest.raises(pytest.skip.Exception):
            _import_engine(_raising_importer(ModuleNotFoundError(f"No module named '{name}'", name=name)))
    with pytest.raises(pytest.skip.Exception):
        _import_engine(_raising_importer(ImportError("libcudart.so.12: cannot open shared object file",
                                                     name="tensor_cuda._tensor_cuda")))
    for exc in (ImportError("cannot import name 'quantize_sp2_keys' from 'tensor_cuda.apa_sp2'",
                            name="tensor_cuda.apa_sp2"),
                ModuleNotFoundError("No module named 'numpy'", name="numpy"),
                ImportError("bare import error"),
                RuntimeError("no CUDA-capable device is detected")):        # not an import failure
        with pytest.raises(type(exc)):
            _import_engine(_raising_importer(exc))


# ---------------------------------------------------------------- detector logic (CPU)
def _attn_np(q, k, v, rep, lo, hi, scale, dt, head_map=None):
    """numpy attention, TRUE visible interval [lo, hi] per row; head h reads KV head head_map[h] (default h // rep)."""
    H = q.shape[1]
    hm = np.arange(H) // rep if head_map is None else np.asarray(head_map)
    kk, vv = k[:, hm], v[:, hm]
    s = np.einsum("bhld,bhsd->bhls", q, kk).astype(np.float32) * np.float32(scale)
    j = np.arange(k.shape[2])
    vis = (j >= lo[..., None]) & (j <= hi[..., None])
    s = np.where(vis, s, -np.inf).astype(np.float32)
    w = np.exp(s - s.max(-1, keepdims=True)).astype(np.float32)
    w = w / w.sum(-1, keepdims=True)
    return P.round_to(np.einsum("bhls,bhsd->bhld", w, vv).astype(np.float32), dt)


def test_pidv_layouts_cover_every_cut_and_every_slab():
    for L, S, nslab, VD, nruns in ((64, 1025, 16, 128, 1), (1, 8192, 16, 128, 1), (3, 33, 80, 64, 2),
                                   (3, 1025, 2, 512, 1), (8, 70, 4, 16, 1)):
        lo, hi = P.causal_intervals(2, 2, L, S)
        cuts = P2.frontier_cuts(lo, hi, S)
        lays = M.pidv_layouts(cuts, nslab, VD)
        assert len(lays) == nruns
        got_cuts, got_slabs = [], set()
        for lay in lays:
            assert np.all(lay["kind"] >= 0)
            cc = np.where(lay["kind"] == M.CUT)[0]
            got_cuts += [(int(lay["cut_t"][c]), "ge" if lay["cut_ge"][c] else "lt") for c in cc]
            got_slabs |= {int(s) for s in lay["slab"][lay["kind"] == M.COLOR]}
        assert sorted(got_cuts) == sorted(cuts) and len(got_cuts) == len(cuts)
        assert got_slabs == set(range(nslab))
    picked, rest = M.spread(range(10), 4)
    assert picked == sorted(picked) and len(set(picked)) == 4 and sorted(picked + rest) == list(range(10))


def test_pidv_operand_is_exact_zero_one():
    B, KVH, S = 2, 3, 40
    cuts = [(10, "ge"), (25, "ge"), (5, "lt"), (30, "lt")]
    lay = M.pidv_layouts(cuts, B * KVH, 16)[0]
    X = M.pidv_operand(B, KVH, S, lay)
    assert set(np.unique(X)) <= {0.0, 1.0}
    j = np.arange(S)
    for c in range(16):
        if lay["kind"][c] == M.CUT:
            t = lay["cut_t"][c]
            want = (j >= t) if lay["cut_ge"][c] else (j < t)
            assert np.array_equal(X[..., c], np.broadcast_to(want, (B, KVH, S)).astype(np.float32))
        else:
            s = lay["slab"][c]
            want = np.zeros((B, KVH, S), np.float32)
            want[s // KVH, s % KVH] = 1.0
            assert np.array_equal(X[..., c], want)


def _np_cell(dt, window=0, B=2, H=4, KVH=2, L=8, S=70, D=16, seed=7):
    rng = np.random.default_rng(seed)
    scale = 1 / math.sqrt(D)
    q, k = P.make_qk(rng, (B, H, L, D), (B, KVH, S, D), scale, dt)
    lo, hi = P.causal_intervals(B, H, L, S, window=window)
    return q, k, lo, hi, scale, dict(B=B, KVH=KVH, S=S, rep=H // KVH, cuts=P2.frontier_cuts(lo, hi, S), VD=16)


def _check(run, lo, hi, g):
    return M.pidv_check(run, g["B"], g["KVH"], g["S"], lo, hi, g["rep"], g["cuts"], g["VD"])


@pytest.mark.parametrize("dt", ["float32", "bfloat16", "float16"])
def test_pidv_numpy_kernels_correct_pass_and_leaks_alarm(dt):
    q, k, lo, hi, scale, g = _np_cell(dt)
    rep, KVH, S = g["rep"], g["KVH"], g["S"]
    good = lambda X: _attn_np(q, k, X, rep, lo, hi, scale, dt)  # noqa: E731
    r = _check(good, lo, hi, g)
    assert r["verdict"] == "PASS" and r["runs"] == 1, r
    bad = {}
    bad["offby1"] = lambda X: _attn_np(q, k, X, rep, lo, np.minimum(hi + 1, S - 1), scale, dt)  # noqa: E731
    for row in (0, 5):                                            # one leaky row (a single tile edge)
        hi2 = hi.copy()
        hi2[..., row] += 1
        bad[f"tile_edge_row{row}"] = lambda X, h2=hi2: _attn_np(q, k, X, rep, lo, h2, scale, dt)  # noqa: E731
    H = q.shape[1]
    bad["gqa_tile"] = lambda X: _attn_np(q, k, X, rep, lo, hi, scale, dt, head_map=np.arange(H) % KVH)  # noqa: E731
    bad["v_batch_stride"] = lambda X: _attn_np(q, k, P.batch_stride_off1(X), rep, lo, hi, scale, dt)  # noqa: E731

    def interior_one_slab(X, b0=1, g0=0, t0=32, n=8):              # I3-like: one slab, one interior tile
        Xm = X.copy()
        Xm[b0, g0, t0:t0 + n] = X[b0, (g0 + 1) % KVH, t0:t0 + n]
        return _attn_np(q, k, Xm, rep, lo, hi, scale, dt)
    bad["interior_one_slab"] = interior_one_slab
    for name, run in bad.items():
        r = _check(run, lo, hi, g)
        assert r["verdict"] == "ALARM" and r["violations"] > 0, (name, r)


def test_pidv_numpy_window_edges_use_lt_cuts():
    dt, W = "float32", 20
    q, k, lo, hi, scale, g = _np_cell(dt, window=W)
    assert any(side == "lt" for _, side in g["cuts"])
    good = lambda X: _attn_np(q, k, X, g["rep"], lo, hi, scale, dt)  # noqa: E731
    r = _check(good, lo, hi, g)
    assert r["verdict"] == "PASS" and r["runs_full"] == 2, r     # 8 ge + 8 lt cuts + 4 slabs > 16 columns
    lo2 = np.maximum(lo - 1, 0)                                    # window edge off by one
    bad = lambda X: _attn_np(q, k, X, g["rep"], lo2, hi, scale, dt)  # noqa: E731
    assert _check(bad, lo, hi, g)["verdict"] == "ALARM"


def test_pidv_blind_to_in_slab_reordering_of_visible_keys():
    """Documented blind spot (by design): keys 0..S-L are visible to every row; a kernel that reverses them inside
    each slab reads the same key SET per row, so every zero/nonzero expectation still holds and pIdV PASSes."""
    dt = "float32"
    q, k, lo, hi, scale, g = _np_cell(dt)
    n = g["S"] - (hi.shape[-1]) + 1                                # keys [0, S-L] are visible to every row

    def reordered(X):
        Xm = X.copy()
        Xm[:, :, :n] = X[:, :, :n][:, :, ::-1]
        return _attn_np(q, k, Xm, g["rep"], lo, hi, scale, dt)
    good = lambda X: _attn_np(q, k, X, g["rep"], lo, hi, scale, dt)  # noqa: E731
    Xr = np.random.default_rng(0).random((g["B"], g["KVH"], g["S"], g["VD"]), dtype=np.float32)
    assert not np.array_equal(reordered(Xr), good(Xr))           # the defect is real on a generic V ...
    assert _check(reordered, lo, hi, g)["verdict"] == "PASS"      # ... and invisible to pIdV


# ---------------------------------------------------------------- real engine
@pytest.mark.parametrize("cid,fam,dt,geo,L,S", EDGE_CELLS, ids=[c[0] + "-" + "-".join(map(str, c[1:])) for c in EDGE_CELLS])
def test_engine_pidv_edge_cell_passes_and_offby1_alarms(cid, fam, dt, geo, L, S, margins_dir):
    M_ = _engine()
    e = M_.make_cell(fam, dt, geo, L, S, margins_dir=margins_dir)
    r = M_.pidv_cell(e)
    assert r["verdict"] == "PASS", (cid, r)
    del e
    m = M_.make_cell(fam, dt, geo, L, S, mutant="offby1", margins_dir=margins_dir)
    r = M_.pidv_cell(m)
    assert r["verdict"] == "ALARM" and r["violations"] > 0, (cid, "offby1", r)


@pytest.mark.parametrize("cid,fam,dt,L", PROD_CELLS, ids=[c[0] + "-" + "-".join(map(str, c[1:])) for c in PROD_CELLS])
def test_engine_pidv_production_layout_passes_and_bstride_alarms(cid, fam, dt, L, margins_dir):
    M_ = _engine()
    try:
        e = M_.make_cell(fam, dt, "QW", L, PROD_S, margins_dir=margins_dir)
        r = M_.pidv_cell(e)
        assert r["verdict"] == "PASS" and r["runs"] == 1, (cid, r)
        del e
        m = M_.make_cell(fam, dt, "QW", L, PROD_S, mutant="bstride", margins_dir=margins_dir)
        r = M_.pidv_cell(m)
        assert r["verdict"] == "ALARM" and r["violations"] > 0, (cid, "bstride", r)
    finally:
        M_.tc.empty_cache()


@pytest.mark.skipif(not SLOW, reason="set TC_SLOW_TESTS=1 to run the S = 98,304 pIdV cell (~26 s, ~10.7 GB host RAM)")
def test_engine_pidv_long_context_98304_passes_and_bstride_alarms(margins_dir):
    M_ = _engine()
    cid, fam, dt, L, S = SLOW_CELL
    try:
        e = M_.make_cell(fam, dt, "QW", L, S, margins_dir=margins_dir)
        r = M_.pidv_cell(e)
        assert r["verdict"] == "PASS" and r["runs"] == 1, (cid, r)
        del e
        m = M_.make_cell(fam, dt, "QW", L, S, mutant="bstride", margins_dir=margins_dir)
        r = M_.pidv_cell(m)
        assert r["verdict"] == "ALARM" and r["violations"] > 0, (cid, "bstride", r)
    finally:
        M_.tc.empty_cache()
