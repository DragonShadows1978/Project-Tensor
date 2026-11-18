"""APA-P1-F2 S4: CPU tests of R1 (margin pin), R3 (capacity range), R4 (source order), R6 (ops.h
declaration, compile probe), R7 (SASS script) and R8 (factory capacity).

The engine cannot build a tensor without a visible GPU (HOST-TENSOR-NEEDS-DEVICE), so apa_sp2.py
and apa_packed.py are loaded from source into a stub package whose _C, _spo2, quant and
ApaSpq1Workspace are recording fakes (P1-D's method), and the packed region of __init__.py is
executed with recording fakes for the real ApaSpq1Workspace (F1's method). A recorded call named
"allocate"/"zeros" is an allocation; "packed_run"/"dense_epsilon" is a device call. The pinned
margin table is the real file artifacts/apa_p1d/packed_e_q_table.json.
APA_P1F2_SRC points the loader at a mutant copy of the package (S5); APA_P1F2_BASE_TREE is the
committed tree (git archive d01118b4) for the R6 probe's "before" side.

Prior art: test doubles / recording fakes (Meszaros, xUnit Test Patterns, 2007); SHA-256 pinning
(NIST FIPS 180-4, 2015); mutation testing (DeMillo, Lipton and Sayward, 1978) for the S5 mutants
these tests must kill. No new technique.
"""
import dataclasses
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
from types import MappingProxyType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("APA_P1F2_SRC", ROOT / "tensor_cuda/tensor_cuda"))
PINNED = ROOT / "artifacts/apa_p1d/packed_e_q_table.json"
BASE_TREE = Path(os.environ.get("APA_P1F2_BASE_TREE", "/tmp/apap1f2_treeF0"))
NVCC = Path("/usr/local/cuda-12.6/bin/nvcc")
INT_MAX = 2**31 - 1
_COUNTER = itertools.count()


class FakeTensor:
    def __init__(self, shape, dtype="float32", device="cuda:0", tag="tensor"):
        self.shape, self.dtype, self.device, self.tag = tuple(shape), dtype, device, tag
        self.revision = 0

    @property
    def ndim(self):
        return len(self.shape)


def make_package(monkeypatch):
    calls = []
    name = f"tcp1f2_{next(_COUNTER)}"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(SRC)]
    quant = types.ModuleType(name + ".quant")

    def _tables(D, bits, H, rot, device):
        calls.append(("tables", D, bits, H))
        return ("R", "CB", "BND")

    def _quantize_keys(k, *tables):
        calls.append(("quantize_dense", k.shape))
        return FakeTensor(k.shape, k.dtype, k.device, tag="dense_kq")
    quant._tables, quant._quantize_keys = _tables, _quantize_keys

    class FakeWorkspace:
        def __init__(self, k, capacity, *, fmt):
            calls.append(("allocate", tuple(k.shape), capacity, fmt))
            B, H, S, D = k.shape
            self.fmt, self.capacity, self.valid_rows = fmt, capacity, 0
            self.codes = FakeTensor((B, H, capacity, D // 2), "uint8", k.device, "codes")
            self.scales = FakeTensor((B, H, capacity), "float32", k.device, "norms")
            self.rotation = FakeTensor((H, D, D))
            self.codebook = FakeTensor((16,))
            self.boundaries = FakeTensor((15,))
            self.append(k)

        @property
        def packed(self):
            return (self.codes, self.scales, self.rotation, self.codebook)

        def append(self, k):
            calls.append(("append", self.valid_rows, k.shape[2]))
            self.valid_rows = k.shape[2]

    class FakeSpo2:
        def packed_append(self, *args):
            raise AssertionError("unused")

        def packed_run(self, *args, **kwargs):
            calls.append(("packed_run", args, kwargs))
            return ("packed_out", "packed_mask") if args[8] else "packed_out"

        def packed_prefill(self, *args, **kwargs):
            # Recording test double (Meszaros, xUnit Test Patterns, 2007).
            calls.append(("packed_prefill", args, kwargs))
            return ("packed_prefill_out", "packed_mask") if args[8] else "packed_prefill_out"

    class FakeC:
        def apa_selective_attention_sp_epsilon(self, *args):
            calls.append(("dense_epsilon", args))
            return "dense_out"

    cmod = FakeC()
    pkg._C, pkg._spo2, pkg.quant, pkg.ApaSpq1Workspace = cmod, FakeSpo2(), quant, FakeWorkspace
    monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, name + ".quant", quant)
    monkeypatch.setitem(sys.modules, name + "._C", cmod)
    sp2 = importlib.import_module(name + ".apa_sp2")
    packed = importlib.import_module(name + ".apa_packed")
    return types.SimpleNamespace(sp2=sp2, packed=packed, calls=calls)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("TC_APA_SP_STORAGE", raising=False)
    monkeypatch.setenv("TC_APA_SP", "1")
    return make_package(monkeypatch)


def kinds(env, kind):
    return [c for c in env.calls if c[0] == kind]


def device_calls(env):
    return [c for c in env.calls if c[0] in ("packed_run", "dense_epsilon")]


def pinned(env):
    return env.packed.PackedRegisteredMargins.load(PINNED)


def decode(env, D=128, S=257, G=2, KV=2, B=1):
    k = FakeTensor((B, KV, S, D))
    p = env.sp2.quantize_sp2_keys(k, 4)
    return FakeTensor((B, KV * G, 1, D)), k, p, FakeTensor((B, KV, S, D))


def attend(env, margins, D=128, **kw):
    q, k, p, v = decode(env, D=D)
    return env.sp2.apa_selective_attention_sp(q, k, p, v, 1 / math.sqrt(D), .01, bulk_bits=4, margins=margins, **kw)


# ------------------------------------------------------------------ R1 (P1R2-F01)
def test_r1_pinned_table_loads_and_matches_the_pin(env):
    P = env.packed
    assert hashlib.sha256(PINNED.read_bytes()).hexdigest() == P.PACKED_TABLE_SHA256
    table = json.loads(PINNED.read_text())
    assert {k: float(v["e_q"]) for k, v in table["entries"].items()} == dict(P.PACKED_PINNED_ENTRIES)
    m = pinned(env)
    assert type(m) is P.PackedRegisteredMargins and m.sha256 == P.PACKED_TABLE_SHA256
    assert m.lookup(4, 64, "float32", 1 / math.sqrt(64)) == P.PACKED_PINNED_ENTRIES["4:64"]
    assert m.lookup(4, 128, "float32", 1 / math.sqrt(128)) == P.PACKED_PINNED_ENTRIES["4:128"]
    assert attend(env, m) == "packed_out" and len(kinds(env, "packed_run")) == 1, "pinned table: positive control"


def hand_built(env):
    P = env.packed
    pins = dict(P.PACKED_PINNED_ENTRIES)
    other = dict(pins, **{"4:64": 0.0})
    return {
        "wrong_sha_pinned_entries": P.PackedRegisteredMargins("0" * 64, MappingProxyType(pins)),
        "pinned_sha_zero_entries": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            {"4:64": 0.0, "4:128": 0.0})),
        "pinned_sha_one_entry_changed": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(other)),
        "pinned_sha_one_ulp_off": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            dict(pins, **{"4:128": float(np.nextafter(pins["4:128"], 0.0))}))),
        "pinned_sha_extra_key": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            dict(pins, **{"2:64": 0.0}))),
        "pinned_sha_missing_key": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            {"4:64": pins["4:64"]})),
        "pinned_sha_float32_values": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            {k: np.float32(v) for k, v in pins.items()})),
        "pinned_sha_string_values": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            {k: repr(v) for k, v in pins.items()})),
        "pinned_sha_not_a_mapping": P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, None),
    }


@pytest.mark.parametrize("case", ["wrong_sha_pinned_entries", "pinned_sha_zero_entries", "pinned_sha_one_entry_changed",
                                  "pinned_sha_one_ulp_off", "pinned_sha_extra_key", "pinned_sha_missing_key",
                                  "pinned_sha_float32_values", "pinned_sha_string_values", "pinned_sha_not_a_mapping"])
def test_r1_hand_built_table_refused_at_lookup_and_at_attention_before_any_device_call(env, case):
    m = hand_built(env)[case]
    with pytest.raises(ValueError, match="pinned"):
        m.lookup(4, 128, "float32", 1 / math.sqrt(128))
    with pytest.raises(ValueError, match="pinned"):
        attend(env, m)
    assert not device_calls(env), "refused before any device call"


def test_r1_subclass_and_entries_replaced_after_load_are_refused(env):
    P = env.packed

    class Experimental(P.PackedRegisteredMargins):
        def lookup(self, *args):
            return 0.0
    sub = Experimental(P.PACKED_TABLE_SHA256, MappingProxyType(dict(P.PACKED_PINNED_ENTRIES)))
    with pytest.raises(ValueError, match="pinned"):
        attend(env, sub)
    m = pinned(env)
    object.__setattr__(m, "entries", MappingProxyType({"4:64": 0.0, "4:128": 0.0}))
    with pytest.raises(ValueError, match="pinned"):
        m.lookup(4, 64, "float32", .125)
    with pytest.raises(ValueError, match="pinned"):
        attend(env, m)
    m2 = pinned(env)
    object.__setattr__(m2, "sha256", "f" * 64)
    with pytest.raises(ValueError, match="pinned"):
        attend(env, m2)
    assert not device_calls(env)


def test_r1_load_cannot_override_the_pin(env, tmp_path):
    P = env.packed
    with pytest.raises(ValueError, match="cannot override the pin"):
        P.PackedRegisteredMargins.load(PINNED, expected_sha256="0" * 64)
    table = json.loads(PINNED.read_text())
    table["entries"]["4:64"]["e_q"] = 2.0
    table["entries"]["4:64"]["max"] = 1.5
    path = tmp_path / "other_UNIT_TEST_ONLY.json"
    path.write_text(json.dumps(table))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cannot override the pin"):
        P.PackedRegisteredMargins.load(path, expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        P.PackedRegisteredMargins.load(path)
    assert P.PackedRegisteredMargins.load(PINNED, expected_sha256=P.PACKED_TABLE_SHA256).sha256 == P.PACKED_TABLE_SHA256
    assert not env.calls


# Coverage of the P1-D policy tests whose synthetic tables R1 now refuses (EXPECTED.json cpu.intended_changes),
# restated on the pinned table.
@pytest.mark.parametrize("D", [64, 128])
def test_r1_pinned_route_forwards_delta_causal_sinks_and_r2(env, D):
    q, k, p, v = decode(env, D=D)
    sinks = FakeTensor((q.shape[1],))
    out = env.sp2.apa_selective_attention_sp(q, k, p, v, 1 / math.sqrt(D), .001, bulk_bits=4, margins=pinned(env),
                                             is_causal=True, sinks=sinks, diagnostics=True)
    assert out == ("packed_out", "packed_mask")
    (_, args, kwargs), = kinds(env, "packed_run")
    want = env.packed.derive_delta(.001, env.packed.PACKED_PINNED_ENTRIES[f"4:{D}"])
    assert args[3] == p._packed_tuple() and args[5] == want and args[6] is True and args[7] is sinks
    assert kwargs == dict(fmt="tq4", carry="partition", gqa_path="r2")
    route = env.packed.last_route("attention")
    assert route["storage"] == "packed" and route["e_q"] == env.packed.PACKED_PINNED_ENTRIES[f"4:{D}"]


def test_r1_pinned_route_refusals_still_precede_any_device_call(env, monkeypatch):
    q, k, p, v = decode(env)
    m, s, call = pinned(env), 1 / math.sqrt(128), env.sp2.apa_selective_attention_sp
    for kwargs, match in [(dict(bulk_bits=2), "bulk_bits does not match"), (dict(margins=None), "registered packed margin")]:
        args = {**dict(bulk_bits=4, margins=m), **kwargs}
        with pytest.raises(ValueError, match=match):
            call(q, k, p, v, s, .01, **args)
    with pytest.raises(ValueError, match="exact-K source"):
        call(q, FakeTensor(k.shape), p, v, s, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="scale=1/sqrt"):
        call(q, k, p, v, .2, .01, bulk_bits=4, margins=m)
    with pytest.raises(ValueError, match="epsilon must be"):
        call(q, k, p, v, s, 0., bulk_bits=4, margins=m)
    assert not device_calls(env) and not kinds(env, "packed_prefill")
    prefill_q = FakeTensor((1, 4, 8, 128))
    assert call(prefill_q, k, p, v, s, .01, bulk_bits=4, margins=m, is_causal=True) == "packed_prefill_out"
    prefill_calls = kinds(env, "packed_prefill")
    (_, args, _), = prefill_calls
    assert args[0] is prefill_q
    p.invalidate()
    with pytest.raises(ValueError, match="stale packed keys"):
        call(q, k, p, v, s, .01, bulk_bits=4, margins=m)
    kd = env.sp2.quantize_sp2_keys(k, 4, storage="dense")
    with pytest.raises(ValueError, match="packed margin table with dense Kq"):
        call(q, k, kd, v, s, .01, bulk_bits=4, margins=m)
    monkeypatch.delenv("TC_APA_SP")
    with pytest.raises(RuntimeError, match="TC_APA_SP=1"):
        call(q, k, p, v, s, .01, bulk_bits=4, margins=m)
    assert not device_calls(env) and kinds(env, "packed_prefill") == prefill_calls


# ------------------------------------------------------------------ R3 (P1R2-F04)
def test_r3_packed_capacity_at_the_bound_is_accepted_one_past_is_refused_before_allocation(env):
    k = FakeTensor((1, 1, 4, 64))
    p = env.packed.PackedSpKeys(k, capacity=INT_MAX)
    assert p.capacity == INT_MAX and kinds(env, "allocate") == [("allocate", (1, 1, 4, 64), INT_MAX, "tq4")]
    before = len(env.calls)
    with pytest.raises(ValueError, match="capacity"):
        env.packed.PackedSpKeys(k, capacity=INT_MAX + 1)
    with pytest.raises(ValueError, match="capacity"):
        env.packed.PackedSpKeys(k, capacity=2**40)
    assert len(env.calls) == before, "nothing allocated for a capacity past the native int bound"


def test_r3_prepare_grid_product_at_and_past_the_bound(env):
    # B*KV*S is the int prepare grid of the initial pack: 4 * (2**29 - 1) fits, 4 * 2**29 does not.
    ok = FakeTensor((2, 2, 2**29 - 1, 64))
    env.packed.PackedSpKeys(ok)
    before = len(env.calls)
    with pytest.raises(ValueError, match="prepare grid"):
        env.packed.PackedSpKeys(FakeTensor((2, 2, 2**29, 64)))
    assert len(env.calls) == before


def test_r3_append_product_refused_before_the_generation_moves(env):
    k = FakeTensor((2, 2, 1, 64))
    p = env.packed.PackedSpKeys(k, capacity=INT_MAX)
    gen, before = p.generation, len(env.calls)
    with pytest.raises(ValueError, match="prepare grid"):
        p.append(FakeTensor((2, 2, 2**29 + 1, 64)))
    assert p.generation == gen and p.visible_rows == 1 and len(env.calls) == before
    p.append(FakeTensor((2, 2, 2**29, 64)))   # 4 * (2**29 - 1) new rows: fits
    assert p.visible_rows == 2**29


def test_r3_factory_capacity_past_the_bound_refused_before_allocation(env):
    with pytest.raises(ValueError, match="capacity"):
        env.sp2.quantize_sp2_keys(FakeTensor((1, 2, 8, 128)), 4, capacity=INT_MAX + 1)
    assert not kinds(env, "allocate") and not kinds(env, "tables")


def region_source():
    text = (SRC / "__init__.py").read_text()
    return text[text.index("# APA_SPO2_D1_BEGIN binding"):text.index("# APA_SPO2_D1_END binding")]


def exec_region(monkeypatch):
    """The real ApaSpq1Workspace from __init__.py's packed region, with recording fakes (F1's method)."""
    calls = []
    name = f"tcf2_{next(_COUNTER)}"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(SRC)]
    quant = types.ModuleType(name + ".quant")
    quant._tables = lambda D, bits, H, rot, device: (FakeTensor((H, D, D)), FakeTensor((16,)), FakeTensor((15,)))
    monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, name + ".quant", quant)

    def zeros(shape, dtype="float32", device=None):
        calls.append(("zeros", tuple(shape), dtype))
        return FakeTensor(shape, dtype, device)

    class Spo2:
        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **kw: calls.append((attr, a, kw)) or ("rot", "norms")
    spec = importlib.util.spec_from_file_location(f"apa_packed_f2_{next(_COUNTER)}", SRC / "apa_packed.py")
    ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap)
    ns = {"__name__": name, "__package__": name, "_C": types.SimpleNamespace(__file__="unused"), "zeros": zeros,
          "__all__": [], "apa_packed": ap, "__fake_spo2__": Spo2()}
    src = region_source().replace("_spo2 = _spo2_load(_C.__file__)", "_spo2 = __fake_spo2__")
    exec(compile(src, str(SRC / "__init__.py"), "exec"), ns)
    return ns, calls


@pytest.mark.parametrize("fmt", ["tq4", "int4"])
def test_r3_workspace_capacity_at_the_bound_accepted_one_past_refused_before_allocation(monkeypatch, fmt):
    ns, calls = exec_region(monkeypatch)
    k = FakeTensor((1, 1, 4, 64))
    w = ns["ApaSpq1Workspace"](k, INT_MAX, fmt=fmt)
    assert w.capacity == INT_MAX and ("zeros", (1, 1, INT_MAX, 32), "uint8") in calls
    before = len(calls)
    for bad in (INT_MAX + 1, 2**40):
        with pytest.raises(ValueError, match="capacity"):
            ns["ApaSpq1Workspace"](k, bad, fmt=fmt)
    with pytest.raises(ValueError, match="integer"):
        ns["ApaSpq1Workspace"](k, 64.0, fmt=fmt)
    assert len(calls) == before, "no allocation and no native call for a refused capacity"


# ------------------------------------------------------------------ R8 (P1R2-F09)
def test_r8_factory_signature():
    spec = importlib.util.spec_from_file_location(f"apa_sp2_sig_{next(_COUNTER)}", SRC / "apa_sp2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = inspect.signature(mod.quantize_sp2_keys).parameters["capacity"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


def test_r8_capacity_is_forwarded_to_packed_keys(env):
    k = FakeTensor((1, 2, 100, 128))
    p = env.sp2.quantize_sp2_keys(k, 4, capacity=164)
    assert isinstance(p, env.packed.PackedSpKeys) and p.capacity == 164
    assert kinds(env, "allocate") == [("allocate", (1, 2, 100, 128), 164, "tq4")], "capacity must reach the workspace"
    p.append(FakeTensor((1, 2, 164, 128)))
    assert p.visible_rows == 164
    with pytest.raises(ValueError, match="within capacity"):
        p.append(FakeTensor((1, 2, 165, 128)))
    p2 = env.sp2.quantize_sp2_keys(k, 4, storage="packed", capacity=128)
    assert p2.capacity == 128


def test_r8_capacity_none_is_unchanged(env):
    k = FakeTensor((1, 2, 100, 128))
    p = env.sp2.quantize_sp2_keys(k, 4)
    assert p.capacity == 100 and kinds(env, "allocate") == [("allocate", (1, 2, 100, 128), 100, "tq4")]
    assert env.sp2.quantize_sp2_keys(k, 4, capacity=None).capacity == 100
    d = env.sp2.quantize_sp2_keys(k, 4, storage="dense", capacity=None)
    assert d.tag == "dense_kq"


def test_r8_dense_storage_refuses_a_capacity_before_device_work(env, monkeypatch):
    k = FakeTensor((1, 2, 100, 128))
    with pytest.raises(ValueError, match="dense storage"):
        env.sp2.quantize_sp2_keys(k, 4, storage="dense", capacity=128)
    monkeypatch.setenv("TC_APA_SP_STORAGE", "dense")
    with pytest.raises(ValueError, match="dense storage"):
        env.sp2.quantize_sp2_keys(k, 4, capacity=128)
    monkeypatch.delenv("TC_APA_SP_STORAGE")
    with pytest.raises(ValueError, match="fell back to dense"):
        env.sp2.quantize_sp2_keys(k, 2, capacity=128)   # 2-bit: the default falls back to dense
    assert not kinds(env, "tables") and not kinds(env, "quantize_dense") and not kinds(env, "allocate")


def test_r8_capacity_below_rows_or_not_an_int_is_refused(env):
    k = FakeTensor((1, 2, 100, 128))
    for bad in (99, 0, 128.0, True, "128"):
        with pytest.raises(ValueError, match="capacity"):
            env.sp2.quantize_sp2_keys(k, 4, capacity=bad)
    assert not kinds(env, "allocate")


# ------------------------------------------------------------------ R4 (source order)
def test_r4_refusals_precede_the_first_launch_in_source():
    spo = ROOT / "tensor_cuda/src/apa_spo"
    eb3 = (spo / "eb3.cuh").read_text()
    run = eb3[eb3.index("std::vector<NDArray> apa_eb3_run("):]
    assert run.index("if(timeline && G!=2 && G!=4)") < run.index("NDArray arena(") < run.index("<<<")
    pl = (spo / "packed_launcher.cuh").read_text()
    for head, guard in (("NDArray apa_spq1_rotate(", 'apa_spo_require_positive_extents("apa_spq1_rotate'),
                        ("std::pair<NDArray,NDArray> apa_spq1_prepare(", 'apa_spo_require_positive_extents("apa_spq1_prepare'),
                        ("void apa_spq1_pack(", 'apa_spo_require_positive_extents("apa_spq1_pack')):
        body = pl[pl.index(head):]
        assert body.index(guard) < body.index("<<<"), head
    eb2b = (spo / "eb2b.cuh").read_text()
    for head, guard in (("NDArray apa_eb2b_values(", 'apa_spo_require_positive_extents("apa_eb2b_values"'),
                        ("std::pair<NDArray,NDArray> apa_eb2b_quantize(", 'apa_spo_require_positive_extents("apa_eb2b_quantize"')):
        body = eb2b[eb2b.index(head):]
        assert body.index(guard) < body.index("<<<"), head
    guards = (spo / "launcher.cuh").read_text()
    assert "throw std::invalid_argument" in guards[guards.index("inline void apa_spo_require_positive_extents"):]


# ------------------------------------------------------------------ R6 (P1R2-F07)
R6_PROBE = """// R6 probe: ops.h + the eb2b.cuh definition (via kernels.cu, its include context),
// then the address of tc::apa_eb2b_run with no target type (ambiguous iff two overloads).
#include "tc/ops.h"
#include "kernels.cu"
static auto apa_p1f2_r6_probe = &tc::apa_eb2b_run;
"""


def r6_compile(tree, tmp_path):
    probe = tmp_path / "r6_probe.cu"
    probe.write_text(R6_PROBE)
    cmd = ["timeout", "600s", str(NVCC), "-std=c++17", "-arch=sm_89", "-fdevice-syntax-only", "-Xcompiler",
           "-fsyntax-only", "-DTC_APA_PACKED=1", "-DTC_APA_EB2B=1", "-DTC_APA_EB3=1", "-DTC_APA_EB4=1",
           "-DAPA_EB2B_MUTANT=0", "-DAPA_FXTC1_VARIANT_DEFAULT=2", f"-I{tree}/tensor_cuda/include",
           f"-I{tree}/tensor_cuda/src", "-c", str(probe), "-o", str(tmp_path / "probe.o")]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)


def test_r6_ops_h_declares_the_one_definition():
    ops = (ROOT / "tensor_cuda/include/tc/ops.h").read_text()
    decl = [l for l in ops.splitlines() if "apa_eb2b_run(" in l]
    assert decl == ["std::vector<NDArray> apa_eb2b_run(const NDArray&,const NDArray&,const NDArray&,const NDArray&,"
                    "const NDArray&,const NDArray&,const NDArray&,const NDArray&,const NDArray&,float,float,float,"
                    "const std::string&,bool,bool,bool);"]
    bindings = (ROOT / "tensor_cuda/src/bindings.cpp").read_text()
    assert "tc::apa_eb2b_run(q.data()" in bindings and "apa_p1f1_eb2b_run" not in bindings


def test_r6_compile_probe_fails_on_the_base_tree_and_compiles_after(tmp_path):
    assert NVCC.exists(), "nvcc 12.6 is the probe's compiler"
    assert (BASE_TREE / "tensor_cuda/include/tc/ops.h").exists(), f"stage the base tree at {BASE_TREE} (git archive)"
    (tmp_path / "base").mkdir()
    (tmp_path / "after").mkdir()
    base = r6_compile(BASE_TREE, tmp_path / "base")
    assert base.returncode != 0 and 'cannot deduce "auto" type' in base.stdout + base.stderr, \
        (base.returncode, (base.stdout + base.stderr)[-2000:])
    after = r6_compile(ROOT, tmp_path / "after")
    assert after.returncode == 0, (after.stdout + after.stderr)[-2000:]


# ------------------------------------------------------------------ R7 (P1R2-F08)
def sass_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("apa_p1f1_sass")


def block(name, body):
    return f"\t\tFunction : {name}\n{body}\t\t..........\n"


def test_r7_renames_and_additions_are_counted_apart_and_the_active_expectation_is_recorded(tmp_path):
    S = sass_module()
    art = tmp_path / "art"
    (art / "sass").mkdir(parents=True)
    keep = "_ZN2tc51_GLOBAL__N__fc2e73a4_11_gemm_apa_cu_{}17blend_dense_exactEv"
    base = (block(keep.format("e8bf8eed_581904"), "  /*0000*/ MOV R1, c[0x0][0x28] ;\n") +
            block("_ZN2tc4spq17prepareIfEEvv", "  /*0000*/ FADD R0, R1, R2 ;\n") +
            block("_ZN2tc4spq14bulkEv", "  /*0000*/ EXIT ;\n") +
            block("__nv_static_old_gone_cu_0a0b0c0d_5", "  /*0000*/ EXIT ;\n"))
    new = (block(keep.format("c79198e4_921183"), "  /*0000*/ MOV R1, c[0x0][0x28] ;\n") +
           block("_ZN2tc4spq17prepareIfEEvv", "  /*0000*/ FMUL R0, R1, R2 ;\n") +
           block("_ZN2tc4spq14bulkEv", "  /*0000*/ EXIT ;\n") +
           block("_ZN2tc4spq114pack_int4_rowsIfEEvv", "  /*0000*/ EXIT ;\n") +
           block("__nv_static_added_two_cu_1a2b3c4d_3", "  /*0000*/ EXIT ;\n"))
    (art / "sass/BASE.sass").write_text(base)
    (art / "sass/NEW.sass").write_text(new)
    mods = {}
    for label in ("BASE", "NEW"):
        mods[label] = art / f"{label}.so"
        mods[label].write_bytes(label.encode())
    expected = art / "EXPECTED.json"
    expected.write_text(json.dumps(dict(sass=dict(
        expected_changed={"r": ["void tc::spq1::prepare<float>"]},
        expected_removed={"r": ["__nv_static_old_gone_cu_00000000_0"]},
        expected_added={"r": ["void tc::spq1::pack_int4_rows<float>", "__nv_static_added_two_cu_00000000_0"]}))))
    override = art / "EXPECTED_SASS_NEW.json"
    override.write_text(json.dumps(dict(expected_removed=["__nv_static_old_gone_cu_00000000_0"])))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    r = S.compare_builds("BASE", mods["BASE"], sha(mods["BASE"]), "NEW", mods["NEW"], sha(mods["NEW"]), art, expected,
                         override, root=tmp_path)
    assert r["status"] == "PASS", r["unexpected_differences"]
    assert r["symbol_suffix_renames"] == 1, "one paired kernel changed only its _cu_ suffix"
    assert r["symbol_only_in_new"] == 2 and r["symbol_only_in_base"] == 1, "additions/removals counted apart"
    assert r["counts"] == dict(equal=2, changed=1, removed=1, added=2)
    assert r["expected_json_sha256"] == sha(expected), "the original registration is still recorded"
    assert r["expected_active"] == dict(path="art/EXPECTED_SASS_NEW.json", sha256=sha(override)), \
        "the expectation actually compared against is recorded"
    r2 = S.compare_builds("BASE", mods["BASE"], sha(mods["BASE"]), "NEW", mods["NEW"], sha(mods["NEW"]), art, expected,
                          None, out_name="SECOND.json", root=tmp_path)
    assert r2["expected_active"] == dict(path="art/EXPECTED.json", sha256=sha(expected))


def test_r7_optional_numeric_suffix_maps_bijectively(tmp_path):
    S = sass_module()
    a = tmp_path / "a.sass"
    b = tmp_path / "b.sass"
    a.write_text(block("_ZN2tc_cu_0123abcdfooEv", "X\n") + block("_ZN2tc_cu_0123abcd_7barEv", "Y\n"))
    b.write_text(block("_ZN2tc_cu_89abcdeffooEv", "X\n") + block("_ZN2tc_cu_89abcdef_9barEv", "Y\n"))
    ba, na = S.read_blocks(a)
    bb, nb = S.read_blocks(b)
    assert set(ba) == set(bb) == {"_ZN2tc_cu_BUILDfooEv", "_ZN2tc_cu_BUILDbarEv"} and ba == bb
