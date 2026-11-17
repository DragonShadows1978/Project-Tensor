"""APA-P1-F1 S4: CPU tests of R3, R4, R8, R9, R10 and R12.

The engine cannot build a tensor without a visible GPU (HOST-TENSOR-NEEDS-DEVICE),
so the packed Python region of tensor_cuda/__init__.py (APA_SPO2_D1_BEGIN binding
to APA_SPO2_D1_END binding) is executed from source in a namespace whose _C,
zeros, quant tables and native _spo2 entries are recording fakes; apa_packed.py
is loaded from source. R9's loader runs against real binaries in child
processes: a module built without TC_APA_PACKED (production's) and a packed
build (Build F, path from APA_P1F1_PACKED_MODULE). R12 preprocesses eb3.cuh.
APA_P1F1_SRC points the loader at a mutant copy of the package (S5).

Prior art: test doubles / recording fakes (Meszaros, xUnit Test Patterns, 2007);
POSIX dlopen/dlsym; the C preprocessor's #error (ISO C). No new technique.
"""
import importlib.util
import inspect
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("APA_P1F1_SRC", ROOT / "tensor_cuda/tensor_cuda"))
NON_PACKED_MODULE = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/"
                         "_tensor_cuda.cpython-312-x86_64-linux-gnu.so")
PACKED_MODULE = os.environ.get("APA_P1F1_PACKED_MODULE")
_COUNTER = itertools.count()


class FakeTensor:
    def __init__(self, shape, dtype="bfloat16", device="cuda:0", tag="tensor"):
        self.shape, self.dtype, self.device, self.tag = tuple(shape), dtype, device, tag
        self.revision = 0

    @property
    def ndim(self):
        return len(self.shape)


class FakeSpo2:
    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "packed_prepare":
                return ("rotated", "norms")
            return "out"
        return call


def region_source():
    text = (SRC / "__init__.py").read_text()
    start = text.index("# APA_SPO2_D1_BEGIN binding")
    end = text.index("# APA_SPO2_D1_END binding")
    return text[start:end]


def load_apa_packed():
    spec = importlib.util.spec_from_file_location(f"apa_packed_f1_{next(_COUNTER)}", SRC / "apa_packed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exec_region(monkeypatch, spo2_present=True):
    """Execute the packed region with fakes; returns (namespace, calls)."""
    calls = []
    name = f"tcf1_{next(_COUNTER)}"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(SRC)]
    quant = types.ModuleType(name + ".quant")

    def _tables(D, bits, H, rot, device):
        calls.append(("tables", D, bits, H))
        return (FakeTensor((H, D, D), "float32"), FakeTensor((16,), "float32"), FakeTensor((15,), "float32"))
    quant._tables = _tables
    monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, name + ".quant", quant)

    def zeros(shape, dtype="float32", device=None):
        calls.append(("zeros", tuple(shape), dtype))
        return FakeTensor(shape, dtype, device)
    fake_c = types.SimpleNamespace(__file__="unused")
    spo2 = FakeSpo2(calls) if spo2_present else None
    ns = {"__name__": name, "__package__": name, "_C": fake_c, "zeros": zeros, "__all__": [],
          "apa_packed": load_apa_packed()}
    src = region_source()
    # The loader line is the only statement that touches the binary; the fake
    # backend replaces its result (R9's loader is tested on real binaries below).
    assert src.count("_spo2 = _spo2_load(_C.__file__)") == 1
    src = src.replace("_spo2 = _spo2_load(_C.__file__)", "_spo2 = __fake_spo2__")
    ns["__fake_spo2__"] = spo2
    exec(compile(src, str(SRC / "__init__.py"), "exec"), ns)
    return ns, calls


@pytest.fixture
def region(monkeypatch):
    return exec_region(monkeypatch)


def kv(S=64, D=128):
    return FakeTensor((1, 2, S, D))


# ------------------------------------------------------------------ R3
def test_r3_eb3_eb4_default_carry_is_partition(region):
    ns, calls = region
    for name in ("apa_eb3", "apa_eb4"):
        p = inspect.signature(ns[name]).parameters["carry"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default == "partition", f"{name} default carry must be partition"
    q, k, v = FakeTensor((1, 4, 1, 128)), kv(), kv()
    ns["apa_eb3"](q, k, v, ("c", "s", "r", "b"), 0.1, 2.0)
    ns["apa_eb4"](q, k, v, "vi", "vs", ("c", "s", "r", "b"), 0.1, 2.0, 1 / 32)
    eb3 = [c for c in calls if c[0] == "eb3_run"][-1]
    eb4 = [c for c in calls if c[0] == "eb4_run"][-1]
    assert eb3[1][7] == "partition" and eb4[1][10] == "partition", "default carry must reach the binding as partition"
    ns["apa_eb3"](q, k, v, ("c", "s", "r", "b"), 0.1, 2.0, carry="row")
    assert [c for c in calls if c[0] == "eb3_run"][-1][1][7] == "row", "explicit row carry stays available"


def test_r3_binding_defaults_are_partition():
    text = (ROOT / "tensor_cuda/src/bindings.cpp").read_text()
    module = text[text.index("PYBIND11_MODULE(_tensor_cuda_spo2"):]
    for entry in ("eb3_run", "eb4_run"):
        block = module[module.index(f'm.def("{entry}"'):]
        block = block[:block.index(";")]
        tail = module[module.index(f'm.def("{entry}"'):]
        tail = tail[tail.index("},py::arg"):tail.index(");", tail.index("},py::arg"))]
        assert 'py::arg("carry")="partition"' in tail, f"{entry} binding default carry must be partition"
        assert 'py::arg("carry")="row"' not in tail


# ------------------------------------------------------------------ R4
def test_r4_eb2b_delta_and_carry_are_required_keywords(region):
    ns, calls = region
    sig = inspect.signature(ns["apa_eb2b"])
    for name in ("delta", "carry"):
        p = sig.parameters[name]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty, name
    q, k, v = FakeTensor((1, 4, 1, 128)), kv(), kv()
    args = (q, k, v, "vi", "vs", ("c", "s", "r", "b"), 0.1, 1 / 32)
    with pytest.raises(TypeError):
        ns["apa_eb2b"](*args, carry="row")
    with pytest.raises(TypeError):
        ns["apa_eb2b"](*args, delta=8.1)
    assert not [c for c in calls if c[0] == "eb2b_run"]
    ns["apa_eb2b"](*args, delta=2.5, carry="partition")
    call = [c for c in calls if c[0] == "eb2b_run"][-1]
    assert call[2]["delta"] == 2.5 and call[2]["carry"] == "partition", "delta and carry must be forwarded"


def test_r4_no_launch_site_literal_and_binding_requires_both():
    eb2b = (ROOT / "tensor_cuda/src/apa_spo/eb2b.cuh").read_text()
    run = eb2b[eb2b.index("std::vector<NDArray> apa_eb2b_run("):]
    run = run[:run.index("\n}\n")]
    assert "8.1f" not in run and "NC*256,scale" not in run, "EB2b launch site must not hard-code delta or carry"
    assert 'carry_mode!="partition" && carry_mode!="row"' in run
    assert "delta<0.f" in run and "std::isfinite(delta)" in run
    assert run.index('carry_mode!="partition"') < run.index("<<<") and run.index("std::isfinite(delta)") < run.index("<<<")
    text = (ROOT / "tensor_cuda/src/bindings.cpp").read_text()
    module = text[text.index("PYBIND11_MODULE(_tensor_cuda_spo2"):]
    tail = module[module.index('m.def("eb2b_run"'):]
    tail = tail[tail.index("},py::arg"):tail.index(");", tail.index("},py::arg"))]
    assert 'py::kw_only(),py::arg("delta"),py::arg("carry"),' in tail, "binding: delta and carry required keywords"


# ------------------------------------------------------------------ R8
def test_r8_shared_identity_helper(region):
    ns, _ = region
    P = ns["apa_packed"]
    src = (SRC / "apa_packed.py").read_text()
    assert "check_exact_k(self, k)" in src and "bind_exact_k(self, k)" in src, "PackedSpKeys uses the shared helper"
    region_text = region_source()
    assert "apa_packed.check_exact_k(self, k)" in region_text and "apa_packed.bind_exact_k(self, k)" in region_text, \
        "ApaSpq1Workspace uses the same shared helper"
    assert callable(P.check_exact_k) and callable(P.bind_exact_k)


def test_r8_workspace_binds_and_accepts_its_exact_k(region):
    ns, calls = region
    k = kv()
    w = ns["ApaSpq1Workspace"](k, 128, fmt="tq4")
    w.check_source(k)
    w.append(k)   # no new rows, same K: accepted
    assert [c[0] for c in calls].count("packed_append") == 1


def test_r8_workspace_refuses_different_k_same_rows(region):
    ns, calls = region
    k = kv()
    w = ns["ApaSpq1Workspace"](k, 128, fmt="tq4")
    other = kv()
    with pytest.raises(ValueError):
        w.check_source(other)
    with pytest.raises(ValueError):
        w.append(other)


def test_r8_workspace_refuses_in_place_edit_and_stale_generation(region):
    ns, _ = region
    k = kv()
    w = ns["ApaSpq1Workspace"](k, 128, fmt="tq4")
    k.revision += 1
    with pytest.raises(ValueError, match="edited in place"):
        w.check_source(k)
    k2 = kv()
    w2 = ns["ApaSpq1Workspace"](k2, 128, fmt="tq4")
    w2.invalidate()
    with pytest.raises(ValueError, match="stale"):
        w2.check_source(k2)
    with pytest.raises(ValueError, match="stale"):
        w2.append(kv(S=65))


def test_r8_append_extends_and_rebinds(region):
    ns, calls = region
    k1 = kv(S=64)
    w = ns["ApaSpq1Workspace"](k1, 128, fmt="int4")
    k2 = kv(S=80)
    w.append(k2)
    assert w.valid_rows == 80 and w.generation == 1
    w.check_source(k2)
    with pytest.raises(ValueError):
        w.check_source(k1)
    appends = [c for c in calls if c[0] == "packed_append"]
    assert [a[1][6] for a in appends] == [0, 64]


def test_r8_entries_refuse_a_different_k_before_the_native_call(region):
    ns, calls = region
    k = kv()
    w = ns["ApaSpq1Workspace"](k, 128, fmt="tq4")
    other = kv()
    q, v = FakeTensor((1, 4, 1, 128)), kv()
    entries = [
        lambda kk: ns["_apa_spq4_entry"](q, kk, v, w, 0.1, 2.0, fmt="tq4"),
        lambda kk: ns["apa_eb2b"](q, kk, v, "vi", "vs", w, 0.1, 1 / 32, delta=8.1, carry="row"),
        lambda kk: ns["apa_eb3"](q, kk, v, w, 0.1, 2.0),
        lambda kk: ns["apa_eb4"](q, kk, v, "vi", "vs", w, 0.1, 2.0, 1 / 32),
    ]
    before = len(calls)
    for fn in entries:
        with pytest.raises(ValueError):
            fn(other)
    assert len(calls) == before, "no native call after an identity refusal"
    for fn in entries:
        fn(k)
    natives = [c[0] for c in calls[before:]]
    assert natives == ["packed_run", "eb2b_run", "eb3_run", "eb4_run"]


# ------------------------------------------------------------------ R10
def test_r10_one_definition_of_the_d1_packed_entry(region):
    ns, _ = region
    text = region_source()
    assert "def _apa_spq1_entry" not in text, "the dead first wrapper is removed"
    assert len(re.findall(r"_C\.apa_selective_attention_sp_d1_packed\s*=", text)) == 1
    assert ns["_C"].apa_selective_attention_sp_d1_packed is ns["_apa_spq4_entry"]


# ------------------------------------------------------------------ R9
LOADER_CHILD = r'''
import importlib.util, json, sys, types
src, so, mode = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src + "/__init__.py").read()
region = text[text.index("# APA_SPO2_D1_BEGIN binding"):text.index("# APA_SPO2_D1_END binding")]
calls = []
def zeros(*a, **k):
    calls.append("zeros")
    raise AssertionError("allocated")
if mode == "simulate_failure":
    real = importlib.util.module_from_spec
    def failing(spec):
        raise ImportError("simulated failure of the second initializer")
    importlib.util.module_from_spec = failing
ns = {"__name__": "tcf1_loader", "__package__": "tcf1_loader", "_C": types.SimpleNamespace(__file__=so),
      "zeros": zeros, "__all__": [], "apa_packed": None}
result = {}
try:
    exec(compile(region, src + "/__init__.py", "exec"), ns)
    result["import"] = "ok"
    result["spo2_is_none"] = ns["_spo2"] is None
    result["entries"] = sorted(x for x in dir(ns["_spo2"]) if not x.startswith("__")) if ns["_spo2"] is not None else []
    errors = {}
    for name in ("ApaSpq1Workspace", "apa_eb2b", "apa_eb3", "apa_eb4", "apa_eb2b_quantize"):
        try:
            if name == "ApaSpq1Workspace":
                ns[name](None, 1, fmt="tq4")
            else:
                ns[name]()
            errors[name] = "no error"
        except Exception as e:
            errors[name] = type(e).__name__ + ": " + str(e)
    result["errors"] = errors
    result["all"] = ns["__all__"]
except Exception as e:
    result["import"] = type(e).__name__ + ": " + str(e)
    result["cause"] = repr(e.__cause__)
result["zeros_calls"] = len(calls)
print("LOADER_JSON " + json.dumps(result))
'''


def run_loader(so, mode="normal"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-c", LOADER_CHILD, str(SRC), str(so), mode], capture_output=True,
                       text=True, timeout=120, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("LOADER_JSON ")]
    assert line, p.stdout[-2000:] + p.stderr[-2000:]
    return json.loads(line[-1][12:])


def test_r9_non_packed_module_raises_clear_error_before_allocating():
    assert NON_PACKED_MODULE.exists()
    r = run_loader(NON_PACKED_MODULE)
    assert r["import"] == "ok" and r["spo2_is_none"], r
    for name, err in r["errors"].items():
        assert err.startswith("RuntimeError") and "built without TC_APA_PACKED" in err, (name, err)
    assert r["zeros_calls"] == 0, "nothing allocated"
    assert "apa_eb3" not in r["all"]


@pytest.mark.skipif(not PACKED_MODULE, reason="APA_P1F1_PACKED_MODULE (Build F) not given")
def test_r9_packed_build_loads_and_a_load_failure_raises_at_import():
    ok = run_loader(PACKED_MODULE)
    assert ok["import"] == "ok" and not ok["spo2_is_none"] and "packed_run" in ok["entries"], ok
    bad = run_loader(PACKED_MODULE, "simulate_failure")
    assert bad["import"].startswith("ImportError") and "simulated failure of the second initializer" in bad["import"], bad
    assert "PyInit__tensor_cuda_spo2" in bad["import"], "the message names the packed build's initializer"


# ------------------------------------------------------------------ R12
@pytest.mark.parametrize("define", [False, True])
def test_r12_eb3_refuses_per_thread_default_stream(define, tmp_path):
    cmd = ["timeout", "60s", "c++", "-E", "-x", "c++", "-I", str(ROOT / "tensor_cuda/src/apa_spo")]
    if define:
        cmd.append("-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
    cmd += [str(ROOT / "tensor_cuda/src/apa_spo/eb3.cuh"), "-o", str(tmp_path / "eb3.i")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if define:
        assert p.returncode != 0 and "single-stream ticket contract" in p.stderr, p.stderr[-1000:]
    else:
        assert p.returncode == 0, p.stderr[-1000:]
