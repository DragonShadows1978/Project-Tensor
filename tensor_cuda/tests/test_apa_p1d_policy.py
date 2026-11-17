"""APA-P1-D S4: CPU tests of the packed-default policy (R-A to R-H).

The engine cannot build a tensor without a visible GPU (open item
HOST-TENSOR-NEEDS-DEVICE), so these tests load apa_sp2.py and apa_packed.py from
source into a stub package whose _C, _spo2, quant, ApaSpq1Workspace and tensors
are recording fakes. They test the Python policy only; the device lines (L1-L6)
test the real module. APA_P1D_SRC points the loader at a mutant copy (S5).

Prior art: test doubles / recording fakes (Meszaros, xUnit Test Patterns, 2007);
exact rational re-derivation of IEEE 754 rounding with fractions.Fraction.
"""
from fractions import Fraction
import gc
import importlib
import itertools
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("APA_P1D_SRC", ROOT / "tensor_cuda/tensor_cuda"))
_COUNTER = itertools.count()


class FakeTensor:
    def __init__(self, shape, dtype="float32", device="cuda:0", tag="tensor"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.revision = 0
        self.tag = tag

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"FakeTensor({self.shape},{self.dtype},{self.device},{self.tag})"


class Recorder:
    def __init__(self):
        self.calls = []


def make_package(monkeypatch, packed_backend=True):
    rec = Recorder()
    name = f"tcp1d_{next(_COUNTER)}"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(SRC)]

    quant = types.ModuleType(name + ".quant")

    def _tables(D, bits, H, rot, device):
        rec.calls.append(("tables", D, bits, H, rot, device))
        return ("R", "CB", "BND")

    def _quantize_keys(k, *tables):
        rec.calls.append(("quantize_dense", k.shape, tables))
        return FakeTensor(k.shape, k.dtype, k.device, tag="dense_kq")
    quant._tables, quant._quantize_keys = _tables, _quantize_keys

    class FakeWorkspace:
        def __init__(self, k, capacity, *, fmt):
            rec.calls.append(("workspace", k.shape, capacity, fmt))
            B, H, S, D = k.shape
            self.fmt, self.capacity, self.valid_rows = fmt, capacity, 0
            self.geometry, self.dtype = (B, H, D), k.dtype
            self.codes = FakeTensor((B, H, capacity, D // 2), "uint8", k.device, "codes")
            self.scales = FakeTensor((B, H, capacity), "float32", k.device, "norms")
            self.rotation = FakeTensor((H, D, D), "float32", k.device, "rotation")
            self.codebook = FakeTensor((16,), "float32", k.device, "codebook")
            self.boundaries = FakeTensor((15,), "float32", k.device, "boundaries")
            self.append(k)

        @property
        def packed(self):
            return (self.codes, self.scales, self.rotation, self.codebook)

        def append(self, k):
            rec.calls.append(("append", self.valid_rows, k.shape[2]))
            self.valid_rows = k.shape[2]

    class FakeSpo2:
        def packed_append(self, *args):  # presence only; FakeWorkspace packs
            raise AssertionError("unused")

        def packed_run(self, *args, **kwargs):
            rec.calls.append(("packed_run", args, kwargs))
            return ("packed_out", "packed_mask") if args[8] else "packed_out"

    class FakeC:
        def apa_selective_attention_sp_epsilon(self, *args):
            rec.calls.append(("dense_epsilon", args))
            return ("dense_out", "dense_mask") if args[9] else "dense_out"

    spo2 = FakeSpo2()
    if not packed_backend:
        spo2 = None
    cmod = FakeC()
    pkg._C, pkg._spo2, pkg.quant, pkg.ApaSpq1Workspace = cmod, spo2, quant, FakeWorkspace
    monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, name + ".quant", quant)
    monkeypatch.setitem(sys.modules, name + "._C", cmod)
    sp2 = importlib.import_module(name + ".apa_sp2")
    packed = importlib.import_module(name + ".apa_packed")
    return types.SimpleNamespace(pkg=pkg, sp2=sp2, packed=packed, rec=rec)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("TC_APA_SP_STORAGE", raising=False)
    monkeypatch.setenv("TC_APA_SP", "1")
    return make_package(monkeypatch)


def keys(B=2, KV=2, S=257, D=128):
    return FakeTensor((B, KV, S, D))


def calls(ns, kind):
    return [c for c in ns.rec.calls if c[0] == kind]


def packed_table(ns, tmp_path):
    # REV3 step T / F2 R1: exercise the public loader on its real pinned file.
    # Prior art: SHA-256 pinning (NIST FIPS 180-4, 2015), inherited; recording
    # fakes (Meszaros, 2007) remain unchanged. Ours: use the shipping fixture.
    return ns.packed.PackedRegisteredMargins.load(ROOT / "artifacts/apa_p1d/packed_e_q_table.json")


def dense_table(ns, tmp_path):
    from_sp2 = ns.sp2
    table = dict(registration_sha256=from_sp2.REGISTRATION_SHA256, status="FROZEN",
                 quantizer=from_sp2.QUANTIZER, measurement_classes=64,
                 statistic="maximum_over_all_registered_G2_valid_scores",
                 entries={f"{b}:{D}": dict(e_q=.5, count=1, mean=.5, max=.5, p99_9=.5)
                          for b in (2, 4) for D in (64, 128)})
    path = tmp_path / f"synthetic_dense_UNIT_TEST_ONLY_{next(_COUNTER)}.json"
    path.write_text(json.dumps(table))
    return from_sp2.RegisteredMargins.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


# ---------------------------------------------------------------- R-A policy
def test_shipped_default_is_packed(env):
    assert env.packed.SHIPPED_DEFAULT == "packed"
    out = env.sp2.quantize_sp2_keys(keys(), 4)
    assert isinstance(out, env.packed.PackedSpKeys)
    route = env.packed.last_route("quantize")
    assert route["storage"] == "packed" and route["source"] == "default"
    assert not calls(env, "quantize_dense")


def test_environment_dense_selects_dense(env, monkeypatch):
    monkeypatch.setenv("TC_APA_SP_STORAGE", "dense")
    out = env.sp2.quantize_sp2_keys(keys(), 4)
    assert isinstance(out, FakeTensor) and out.tag == "dense_kq", "env dense must select dense"
    route = env.packed.last_route("quantize")
    assert route["storage"] == "dense" and route["source"] == "environment"
    assert not calls(env, "workspace")


def test_argument_beats_environment_and_environment_beats_default(env, monkeypatch):
    monkeypatch.setenv("TC_APA_SP_STORAGE", "dense")
    out = env.sp2.quantize_sp2_keys(keys(), 4, storage="packed")
    assert isinstance(out, env.packed.PackedSpKeys)
    assert env.packed.last_route("quantize")["source"] == "argument"
    monkeypatch.setenv("TC_APA_SP_STORAGE", "packed")
    out = env.sp2.quantize_sp2_keys(keys(), 4, storage="dense")
    assert isinstance(out, FakeTensor)
    assert env.packed.last_route("quantize")["source"] == "argument"
    assert isinstance(env.sp2.quantize_sp2_keys(keys(), 4), env.packed.PackedSpKeys)
    assert env.packed.last_route("quantize")["source"] == "environment"
    assert env.packed.resolve_storage(None) == ("packed", "environment")
    monkeypatch.delenv("TC_APA_SP_STORAGE")
    assert env.packed.resolve_storage(None) == ("packed", "default")
    assert env.packed.resolve_storage("dense") == ("dense", "argument")


@pytest.mark.parametrize("bad", ["Packed", "DENSE", "int4", "", " packed", 1, True, b"packed", ("packed",)])
def test_invalid_storage_argument_raises_before_device_work(env, bad):
    with pytest.raises(ValueError, match="storage must be"):
        env.sp2.quantize_sp2_keys(keys(), 4, storage=bad)
    assert env.rec.calls == []


@pytest.mark.parametrize("bad", ["Packed", "", "none", "dense ", "0"])
def test_invalid_storage_environment_raises_before_device_work(env, monkeypatch, bad):
    monkeypatch.setenv("TC_APA_SP_STORAGE", bad)
    with pytest.raises(ValueError, match="TC_APA_SP_STORAGE"):
        env.sp2.quantize_sp2_keys(keys(), 4)
    assert env.rec.calls == []


def test_existing_bits_and_geometry_refusals_unchanged(env):
    for bits in (3, 8, True, 4.0):
        with pytest.raises(ValueError, match="bulk_bits must be 2 or 4"):
            env.sp2.quantize_sp2_keys(keys(), bits)
    for k in (FakeTensor((1, 2, 3)), FakeTensor((1, 2, 3, 96)), FakeTensor((1, 2, 3, 64), "bfloat16")):
        with pytest.raises(ValueError, match="registered fp32 key geometry"):
            env.sp2.quantize_sp2_keys(k, 4)
    assert env.rec.calls == []


# ---------------------------------------------------------------- R-B fallback
def test_default_two_bit_takes_dense_with_reason_and_keeps_two_bits(env):
    out = env.sp2.quantize_sp2_keys(keys(D=64), 2)
    assert isinstance(out, FakeTensor) and out.tag == "dense_kq"
    tables = calls(env, "tables")
    assert tables and tables[-1][2] == 2, "two bits must stay two bits"
    route = env.packed.last_route("quantize")
    assert route["storage"] == "dense" and route["bulk_bits"] == 2
    assert "bulk_bits=2" in route["fallback_reason"]
    assert not calls(env, "workspace")


def test_environment_packed_is_the_default_policy_for_two_bits(env, monkeypatch):
    monkeypatch.setenv("TC_APA_SP_STORAGE", "packed")
    out = env.sp2.quantize_sp2_keys(keys(), 2)
    assert isinstance(out, FakeTensor)
    assert "bulk_bits=2" in env.packed.last_route("quantize")["fallback_reason"]


def test_explicit_packed_that_cannot_be_honoured_raises(env):
    with pytest.raises(ValueError, match="cannot be honoured"):
        env.sp2.quantize_sp2_keys(keys(), 2, storage="packed")
    assert not calls(env, "quantize_dense") and not calls(env, "workspace")


def test_explicit_dense_on_four_bits_never_packs(env):
    out = env.sp2.quantize_sp2_keys(keys(), 4, storage="dense")
    assert isinstance(out, FakeTensor) and calls(env, "tables")[-1][2] == 4
    route = env.packed.last_route("quantize")
    assert route["storage"] == "dense" and route["fallback_reason"] is None


def test_missing_packed_backend_default_falls_back_explicit_raises(monkeypatch):
    monkeypatch.delenv("TC_APA_SP_STORAGE", raising=False)
    ns = make_package(monkeypatch, packed_backend=False)
    out = ns.sp2.quantize_sp2_keys(keys(), 4)
    assert isinstance(out, FakeTensor)
    assert "no packed backend" in ns.packed.last_route("quantize")["fallback_reason"]
    with pytest.raises(ValueError, match="cannot be honoured"):
        ns.sp2.quantize_sp2_keys(keys(), 4, storage="packed")


def test_cpu_tensor_default_falls_back_explicit_raises(env):
    k = FakeTensor((1, 2, 5, 64), device="cpu")
    assert isinstance(env.sp2.quantize_sp2_keys(k, 4), FakeTensor)
    assert "CUDA" in env.packed.last_route("quantize")["fallback_reason"]
    with pytest.raises(ValueError, match="cannot be honoured"):
        env.sp2.quantize_sp2_keys(k, 4, storage="packed")


# ---------------------------------------------------------------- R-C object
def test_packed_object_fields(env):
    k = keys(B=2, KV=4, S=257, D=64)
    p = env.sp2.quantize_sp2_keys(k, 4)
    P = env.packed
    assert not isinstance(p, (tuple, list, FakeTensor))
    assert (p.batch, p.kv_heads, p.dim, p.capacity, p.visible_rows) == (2, 4, 64, 257, 257)
    assert (p.dtype, p.device, p.device_ordinal, p.bulk_bits) == ("float32", "cuda:0", 0, 4)
    assert p.codes.shape == (2, 4, 257, 32) and p.codes.dtype == "uint8"
    assert p.norms.shape == (2, 4, 257) and p.norms.dtype == "float32"
    assert p.rotation.shape == (4, 64, 64) and p.codebook.shape == (16,)
    assert p.quantizer == env.sp2.QUANTIZER and p.packed_format == "tq4"
    assert p.table_quantizer == P.PACKED_QUANTIZER
    assert p.table_registration_sha256 == P.PACKED_REGISTRATION_SHA256
    assert p.generation == 0
    prov = p.provenance()
    assert prov["backend"] == P.PACKED_BACKEND and prov["bulk_bits"] == 4
    assert calls(env, "workspace") == [("workspace", k.shape, 257, "tq4")]


def _reachable_tensors(obj):
    seen, out = set(), []
    def walk(x):
        if id(x) in seen:
            return
        seen.add(id(x))
        if isinstance(x, FakeTensor):
            out.append(x)
        elif isinstance(x, (list, tuple, set)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)
        elif hasattr(x, "__dict__") or hasattr(type(x), "__slots__"):
            for name in getattr(type(x), "__slots__", ()):
                if name != "__weakref__" and hasattr(x, name):
                    walk(getattr(x, name))
            for y in getattr(x, "__dict__", {}).values():
                walk(y)
    walk(obj)
    return out


def test_packed_object_owns_no_k_shaped_tensor_and_holds_k_weakly(env):
    k = keys(B=1, KV=2, S=300, D=128)
    p = env.sp2.quantize_sp2_keys(k, 4)
    owned = _reachable_tensors(p)
    k_shaped = [t for t in owned if len(t.shape) == 4 and t.shape[-1] == k.shape[-1]]
    assert not k_shaped, f"packed object retains a K-shaped tensor: {k_shaped}"
    assert all(t is not k for t in owned)
    assert set(p.owned_buffers()) == {"codes", "norms"}
    per_key = sum(int(np.prod(t.shape[3:] or (1,))) * (1 if t.dtype == "uint8" else 4)
                  for t in p.owned_buffers().values())
    assert per_key == 68
    del k
    gc.collect()
    assert p._source is None or p._source() is None


def test_append_is_explicit_validated_and_rebinds(env):
    k1 = keys(S=100)
    p = env.packed.PackedSpKeys(k1, capacity=128)
    k2 = keys(S=101)
    p.check_source(k1)
    with pytest.raises(ValueError, match="exact-K source"):
        p.check_source(k2)
    p.append(k2)
    assert p.visible_rows == 101 and p.generation == 1
    p.check_source(k2)
    with pytest.raises(ValueError, match="exact-K source"):
        p.check_source(k1)
    for bad in (keys(S=129), keys(S=99), FakeTensor((2, 2, 110, 64)), FakeTensor((2, 3, 110, 128)),
                FakeTensor((2, 2, 110, 128), device="cuda:1"), FakeTensor((2, 2, 110, 128), "bfloat16")):
        with pytest.raises(ValueError):
            p.append(bad)
    with pytest.raises(ValueError, match="adds no rows"):
        p.append(keys(S=101))
    assert p.append(k2) is p and p.generation == 1
    assert [c[1:] for c in calls(env, "append")] == [(0, 100), (100, 101)]


def test_stale_generation_is_refused(env, tmp_path):
    k = keys(S=64)
    p = env.sp2.quantize_sp2_keys(k, 4)
    p.check_source(k)
    p.invalidate()
    with pytest.raises(ValueError, match="stale packed keys"):
        p.check_source(k)
    q = FakeTensor((2, 4, 1, 128))
    v = FakeTensor((2, 2, 64, 128))
    with pytest.raises(ValueError, match="stale packed keys"):
        env.sp2.apa_selective_attention_sp(q, k, p, v, 1 / math.sqrt(128), .01, bulk_bits=4,
                                           margins=packed_table(env, tmp_path))
    with pytest.raises(ValueError, match="stale packed keys"):
        p.append(keys(S=65))
    assert not calls(env, "packed_run")


def test_in_place_edit_of_exact_k_is_refused(env):
    k = keys(S=64)
    p = env.sp2.quantize_sp2_keys(k, 4)
    k.revision += 1
    with pytest.raises(ValueError, match="edited in place"):
        p.check_source(k)


# ---------------------------------------------------------------- R-D wrapper
def _decode(env, D=128, S=257, G=2, KV=2, B=1):
    k = keys(B=B, KV=KV, S=S, D=D)
    p = env.sp2.quantize_sp2_keys(k, 4)
    q = FakeTensor((B, KV * G, 1, D))
    v = FakeTensor((B, KV, S, D))
    return q, k, p, v


def test_dense_table_on_packed_keys_is_refused(env, tmp_path):
    q, k, p, v = _decode(env)
    with pytest.raises(ValueError, match="dense margin table with packed keys"):
        env.sp2.apa_selective_attention_sp(q, k, p, v, 1 / math.sqrt(128), .01, bulk_bits=4,
                                           margins=dense_table(env, tmp_path))
    assert not calls(env, "packed_run") and not calls(env, "dense_epsilon")


def test_packed_table_on_dense_kq_is_refused(env, tmp_path):
    k = keys()
    kq = env.sp2.quantize_sp2_keys(k, 4, storage="dense")
    q = FakeTensor((2, 4, 1, 128))
    with pytest.raises(ValueError, match="packed margin table with dense Kq"):
        env.sp2.apa_selective_attention_sp(q, k, kq, k, 1 / math.sqrt(128), .01, bulk_bits=4,
                                           margins=packed_table(env, tmp_path))
    assert not calls(env, "dense_epsilon")


def test_packed_route_refusals_before_launch(env, tmp_path, monkeypatch):
    q, k, p, v = _decode(env)
    m = packed_table(env, tmp_path)
    s = 1 / math.sqrt(128)
    call = env.sp2.apa_selective_attention_sp
    with pytest.raises(ValueError, match="bulk_bits does not match"):
        call(q, k, p, v, s, .01, bulk_bits=2, margins=m)
    with pytest.raises(ValueError, match="exact-K source"):
        call(q, keys(), p, v, s, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="registered packed margin"):
        call(q, k, p, v, s, .01, bulk_bits=4, margins=None)
    with pytest.raises(ValueError, match="rank four"):
        call(FakeTensor((4, 1, 128)), k, p, v, s, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="scale=1/sqrt"):
        call(q, k, p, v, .2, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="fp32"):
        call(FakeTensor((1, 4, 1, 128), "bfloat16"), k, p, v, s, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="device"):
        call(FakeTensor((1, 4, 1, 128), device="cuda:1"), k, p, v, s, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="device"):
        call(q, k, p, v, s, .01, bulk_bits=4, margins=m, sinks=FakeTensor((4,), device="cuda:1"))
    for eps in (0., -1., 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="epsilon must be"):
            call(q, k, p, v, s, eps, bulk_bits=4, margins=m)
    monkeypatch.delenv("TC_APA_SP")
    with pytest.raises(RuntimeError, match="TC_APA_SP=1"):
        call(q, k, p, v, s, .01, bulk_bits=4, margins=m)
    assert not calls(env, "packed_run")


@pytest.mark.parametrize("D", [64, 128])
def test_packed_route_forwards_delta_causal_sinks_and_r2(env, tmp_path, D):
    q, k, p, v = _decode(env, D=D)
    m = packed_table(env, tmp_path)
    sinks = FakeTensor((q.shape[1],))
    eps = .001
    out = env.sp2.apa_selective_attention_sp(q, k, p, v, 1 / math.sqrt(D), eps, bulk_bits=4,
                                             margins=m, is_causal=True, sinks=sinks, diagnostics=True)
    assert out == ("packed_out", "packed_mask")
    (_, args, kwargs), = calls(env, "packed_run")
    # Expected margin comes from the pinned data, independently of lookup.
    e_q = json.loads((ROOT / "artifacts/apa_p1d/packed_e_q_table.json").read_text())["entries"][f"4:{D}"]["e_q"]
    want = env.packed.derive_delta(eps, e_q)
    assert args[0] is q and args[1] is k and args[2] is v
    assert args[3] == p._packed_tuple()
    assert args[4] == float(1 / math.sqrt(D))
    assert args[5] == want and args[5] != eps, "delta must be derived from epsilon and e_q"
    assert args[6] is True, "is_causal must be forwarded"
    assert args[7] is sinks, "sinks must be forwarded"
    assert args[8] is True
    assert kwargs == dict(fmt="tq4", carry="partition", gqa_path="r2")
    assert not calls(env, "dense_epsilon")


def test_dense_route_unchanged_and_reported(env, tmp_path):
    k = keys()
    kq = env.sp2.quantize_sp2_keys(k, 4, storage="dense")
    q = FakeTensor((2, 4, 1, 128))
    m = dense_table(env, tmp_path)
    out = env.sp2.apa_selective_attention_sp(q, k, kq, k, 1 / math.sqrt(128), .001, bulk_bits=4,
                                             margins=m, is_causal=True)
    assert out == "dense_out"
    (_, args), = calls(env, "dense_epsilon")
    assert args[2] is kq and args[5:8] == (.001, .5, True)
    route = env.packed.last_route("attention")
    assert route["storage"] == "dense" and route["backend"] == env.packed.DENSE_BACKEND
    assert not calls(env, "packed_run")


# ---------------------------------------------------------------- delta helper
def _independent_delta(epsilon, e_q):
    """Re-derivation: smallest fp32 >= the double margin, by exact rationals."""
    margin = -math.log(epsilon) + 2.0 * e_q          # the native's double expression
    target = Fraction(margin)
    near = np.float32(margin)
    candidates = [np.nextafter(near, np.float32(-np.inf)), near, np.nextafter(near, np.float32(np.inf))]
    ok = [c for c in candidates if np.isfinite(c) and Fraction(float(c)) >= target]
    return float(min(ok, key=lambda c: Fraction(float(c))))


def delta_fixtures():
    eps_grid = [1., .1, .03, .01, .003, .001, .0003, .0001, .05, 1e-300, 5e-324, np.nextafter(1., 0.)]
    eq_grid = [0., .05, .5, 2., 8., 2.0 ** -30, 0.0627, 0.1234567, 1e30]
    fixtures = [(e, q) for e in eps_grid for q in eq_grid]
    # Margins exactly representable in fp32 (no bump) and one ulp either side.
    for f in (1.0, 0.5, 3.0, 1.0 + 2.0 ** -23):
        fixtures.append((1., f / 2))
        fixtures.append((1., float(np.nextafter(np.float32(f), np.float32(0))) / 2 + 2.0 ** -60))
    return fixtures


def test_delta_helper_equals_independent_rule_on_boundary_fixtures(env):
    rounded_down = rounded_up = exact = 0
    for epsilon, e_q in delta_fixtures():
        got = env.packed.derive_delta(epsilon, e_q)
        assert got == _independent_delta(epsilon, e_q), (epsilon, e_q)
        margin = -math.log(epsilon) + 2.0 * e_q
        assert Fraction(got) >= Fraction(margin)
        assert np.float32(got) == got
        near = float(np.float32(margin))
        exact += near == margin
        rounded_down += near < margin
        rounded_up += near > margin
    assert exact and rounded_down and rounded_up  # every rounding case is exercised


def test_delta_helper_domain_refusals(env):
    for eps, e in [(0, 0), (-.1, 0), (1.1, 0), (np.nan, 0), (np.inf, 0), (.1, -1), (.1, np.nan),
                   (.1, np.inf), (.1, 1e100), (1., 2e38)]:
        with pytest.raises(ValueError):
            env.packed.derive_delta(eps, e)


# ---------------------------------------------------------------- R-F loader
def test_packed_loader_checks_its_own_registration_quantizer_and_entries(env, tmp_path):
    # REV3 step T: R1 refuses synthetic tables at the pin before inspecting
    # their registration/entries. Keep the malformed cases and assert that
    # earlier refusal, plus the pinned positive control and lookup domains.
    # Prior art: SHA-256 pinning (NIST FIPS 180-4, 2015), inherited; test doubles
    # (Meszaros, 2007). Ours: assertions for the revised public loader contract.
    pinned_path = ROOT / "artifacts/apa_p1d/packed_e_q_table.json"
    entries = json.loads(pinned_path.read_text())["entries"]
    m = packed_table(env, tmp_path)
    assert m.lookup(4, 64, "float32", 1 / math.sqrt(64)) == entries["4:64"]["e_q"]
    assert m.lookup(4, 128, "float32", 1 / math.sqrt(128)) == entries["4:128"]["e_q"]
    for args in [(2, 64, "float32", .125), (True, 64, "float32", .125), (4, 64, "float16", .125),
                 (4, 32, "float32", 1 / math.sqrt(32)), (4, 64, "float32", .2)]:
        with pytest.raises(ValueError):
            m.lookup(*args)

    def refuse_synthetic(**override):
        table = json.loads(pinned_path.read_text())
        table.update(override)
        path = tmp_path / f"synthetic_packed_UNIT_TEST_ONLY_{next(_COUNTER)}.json"
        path.write_text(json.dumps(table))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="cannot override the pin"):
            env.packed.PackedRegisteredMargins.load(path, expected_sha256=digest)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            env.packed.PackedRegisteredMargins.load(path)
        assert not env.rec.calls, "synthetic table refused before allocation or device work"

    for override in [dict(quantizer=env.sp2.QUANTIZER), dict(registration_sha256=env.sp2.REGISTRATION_SHA256),
                     dict(status="DRAFT"), dict(measurement_classes=64), dict(statistic="mean")]:
        refuse_synthetic(**override)
    refuse_synthetic(entries={"4:64": dict(e_q=.1, count=1, mean=.1, max=.1, p99_9=.1)})
    bad_entries = json.loads(pinned_path.read_text())["entries"]
    bad_entries["4:64"]["e_q"] = float("nan")
    refuse_synthetic(entries=bad_entries)
    path = tmp_path / "x.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="cannot override the pin"):
        env.packed.PackedRegisteredMargins.load(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        env.packed.PackedRegisteredMargins.load(path)
    # The dense loader refuses a packed table and vice versa.
    dense = dense_table(env, tmp_path)
    assert not isinstance(dense, env.packed.PackedRegisteredMargins)
    assert not isinstance(m, env.sp2.RegisteredMargins)
    assert not env.rec.calls, "all loader checks precede allocation or device work"


# ---------------------------------------------------------------- R-H routes
def test_routes_are_reported_for_quantize_and_attention(env, tmp_path):
    q, k, p, v = _decode(env)
    env.sp2.apa_selective_attention_sp(q, k, p, v, 1 / math.sqrt(128), .01, bulk_bits=4,
                                       margins=packed_table(env, tmp_path))
    route = env.packed.last_route("attention")
    assert route["storage"] == "packed" and route["backend"] == env.packed.PACKED_BACKEND
    assert route["fallback_reason"] is None and route["l"] == 1
    env.sp2.quantize_sp2_keys(keys(), 2)
    route = env.packed.last_route("quantize")
    assert route["storage"] == "dense" and route["fallback_reason"]
    with pytest.raises(ValueError):
        env.packed.last_route("other")


# ---------------------------------------------------------------- R-E prefill
def test_prefill_on_packed_keys_is_refused_naming_dense(env, tmp_path):
    """L3 found the transient reconstruction not bitwise equal: R-E refuses."""
    k = keys(B=1, KV=2, S=64, D=64)
    p = env.sp2.quantize_sp2_keys(k, 4)
    q = FakeTensor((1, 4, 8, 64))
    with pytest.raises(ValueError, match="storage='dense'"):
        env.sp2.apa_selective_attention_sp(q, k, p, FakeTensor((1, 2, 64, 64)), 1 / math.sqrt(64), .01,
                                           bulk_bits=4, margins=packed_table(env, tmp_path), is_causal=True)
    assert not calls(env, "packed_run") and not calls(env, "quantize_dense")
    route = env.packed.last_route("attention")
    assert route["storage"] == "packed" and route["l"] == 8 and "prefill" in route["fallback_reason"]
    # The object never gains a dense copy.
    assert not [t for t in _reachable_tensors(p) if t.shape[-1] == 64 and len(t.shape) == 4]
