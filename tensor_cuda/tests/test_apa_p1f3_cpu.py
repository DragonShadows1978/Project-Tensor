"""APA-P1-F3 S4: CPU tests of R1' (canonical pinned margins at use), R2' (NaN norm on any non-finite computed
norm, source form) and R3' (int64 range checks and the operand-naming ValueError for oversized attention extents).

The engine cannot build a tensor without a visible GPU (HOST-TENSOR-NEEDS-DEVICE), so:
- R1' reuses F2's recording-fake package (tensor_cuda/tests/test_apa_p1f2_cpu.py make_package, imported, not
  copied): apa_sp2.py and apa_packed.py are loaded from source (APA_P1F3_SRC, default this worktree's package; a
  mutant arm's package for S5) into a stub package whose _spo2.packed_run records its arguments.
- R3' is tested with a host-level fake: the real guard blocks of tensor_cuda/src/apa_spo/launcher.cuh and the real
  prologue of apa_eb2b_run (eb2b.cuh, from its first statement up to its GQA check) are cut from the source files
  (APA_P1F3_ENGINE_ROOT, default this worktree; a mutant tree for S5), compiled with g++ -std=c++17 -O0 -fwrapv
  against a stub NDArray that has a shape only, and run; -fwrapv makes a wrapped signed int product the modular
  value REV3 illustrated (an undefined product in ISO C++17 otherwise). Nothing here touches CUDA. The native
  refusals on the real entries run on GPU in LF3 (DLPack re-imports with large extents).

Prior art: test doubles / recording fakes (Meszaros, xUnit Test Patterns, 2007); time-of-check to time-of-use
(MITRE CWE-367) for the stateful mapping; ISO C++17 signed overflow and narrowing; mutation testing (DeMillo,
Lipton and Sayward, 1978) for the S5 mutants these tests must kill. No new technique.
"""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_apa_p1f2_cpu as f2t  # noqa: E402  (F2's recording fakes, imported)

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("APA_P1F3_SRC", ROOT / "tensor_cuda/tensor_cuda"))
ENGINE = Path(os.environ.get("APA_P1F3_ENGINE_ROOT", ROOT))
PINNED = ROOT / "artifacts/apa_p1d/packed_e_q_table.json"
INT_MAX = 2**31 - 1


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(f2t, "SRC", SRC)
    monkeypatch.delenv("TC_APA_SP_STORAGE", raising=False)
    monkeypatch.setenv("TC_APA_SP", "1")
    return f2t.make_package(monkeypatch)


def bits(x):
    return np.float64(x).view(np.uint64).item()


class StatefulEntries(dict):
    """A dict subclass a caller can hand-build: its first `pinned_reads` item reads return the pinned values,
    every later read returns 0.0 (REV3's P1R3-F01 attack). `reads` counts item reads, so a re-read after the
    check is visible in every assertion message."""

    def __init__(self, pins, pinned_reads):
        super().__init__(pins)
        self.pinned_reads, self.reads = pinned_reads, 0

    def __getitem__(self, key):
        self.reads += 1
        return dict.__getitem__(self, key) if self.reads <= self.pinned_reads else 0.0


def stateful_table(env, pinned_reads):
    P = env.packed
    return P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, StatefulEntries(dict(P.PACKED_PINNED_ENTRIES), pinned_reads))


# ------------------------------------------------------------------ R1' (P1R3-F01)
@pytest.mark.parametrize("D", [64, 128])
def test_r1p_stateful_mapping_at_lookup_returns_the_pinned_value(env, D):
    m = stateful_table(env, 2)          # pinned to the check's two reads, 0.0 to any later read
    got = m.lookup(4, D, "float32", 1 / math.sqrt(D))
    pin = env.packed.PACKED_PINNED_ENTRIES[f"4:{D}"]
    assert type(got) is float and bits(got) == bits(pin), \
        f"lookup returned {got!r}, not the pinned {pin!r}; mapping item reads = {m.entries.reads}"
    assert m.entries.reads == 2, f"lookup re-read the caller's mapping after the check (reads = {m.entries.reads})"


@pytest.mark.parametrize("pinned_reads", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("D", [64, 128])
def test_r1p_stateful_mapping_at_attention_gets_the_pinned_e_q_or_is_refused(env, D, pinned_reads):
    m = stateful_table(env, pinned_reads)
    pin = env.packed.PACKED_PINNED_ENTRIES[f"4:{D}"]
    try:
        out = f2t.attend(env, m, D=D)
    except ValueError as e:
        assert "pinned" in str(e), str(e)
        assert not f2t.device_calls(env), "refused only before any device call"
        return
    assert out == "packed_out"
    (_, args, kwargs), = f2t.kinds(env, "packed_run")
    want = env.packed.derive_delta(.01, pin)
    assert bits(args[5]) == bits(want), \
        f"the backend received delta {args[5]!r}, not the pinned route's {want!r}; mapping item reads = {m.entries.reads}"
    route = env.packed.last_route("attention")
    assert bits(route["e_q"]) == bits(pin), f"e_q at use {route['e_q']!r} != pinned {pin!r}"


@pytest.mark.parametrize("D", [64, 128])
def test_r1p_loaded_pinned_table_is_unchanged(env, D):
    m = f2t.pinned(env)
    e_q = json.loads(PINNED.read_text())["entries"][f"4:{D}"]["e_q"]
    got = m.lookup(4, D, "float32", 1 / math.sqrt(D))
    assert type(got) is float and bits(got) == bits(e_q) == bits(m.entries[f"4:{D}"])
    f2t.attend(env, m, D=D)
    (_, args, _), = f2t.kinds(env, "packed_run")
    assert bits(args[5]) == bits(env.packed.derive_delta(.01, e_q))
    assert bits(env.packed.last_route("attention")["e_q"]) == bits(e_q)


def test_r1p_margin_mapping_reads_census():
    """Every read of a margin mapping in apa_packed.py: only check_pinned_margins reads `entries` (validation);
    lookup returns the canonical PACKED_PINNED_ENTRIES; packed_attention reaches values only through lookup."""
    text = (SRC / "apa_packed.py").read_text()
    lookup = text[text.index("    def lookup(self, bulk_bits, dim, dtype, scale):"):text.index("def check_pinned_margins(")]
    assert "self.entries" not in lookup and "return PACKED_PINNED_ENTRIES[" in lookup
    attention = text[text.index("def packed_attention("):]
    assert ".entries" not in attention and "PackedRegisteredMargins.lookup(margins," in attention
    check = text[text.index("def check_pinned_margins("):text.index("def packed_attention(")]
    assert check.count("entries[key]") == 1


# ------------------------------------------------------------------ R2' (P1R3-F02), source form
def test_r2p_prepare_stores_nan_for_any_nonfinite_computed_norm_and_keeps_the_shared_norm():
    text = (ENGINE / "tensor_cuda/src/apa_spo/packed.cuh").read_text()
    body = text[text.index("template<class T> __global__ void prepare("):text.index("// APA_SPQ1_END deterministic_prepare")]
    assert "norms[row]=(nonfinite || !isfinite(norm))?__int_as_float(0x7fc00000):norm;" in body
    assert "unit[d]=__fdiv_rn(ld<T>(k,src+d),__fadd_rn(norm,1e-12f));" in body, "the shared norm feeds the unit vector"
    assert body.count("norm=__fsqrt_rn(ss);") == 1


# ------------------------------------------------------------------ R3' (P1R3-F03), host-level fake
ENTRIES = [
    ("packed_launcher.cuh", "std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq1_run(", "apa_spq1 packed r2 (packed_run)"),
    ("spq4.cuh", "std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_run(", "apa_spq4_run (packed_run)"),
    ("spq4.cuh", "std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_tc(", "apa_spq4 tc int4"),
    ("spq6.cuh", "std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq6_tq4_tc(", "apa_spq6 tc tq4"),
    ("eb2b.cuh", "std::vector<NDArray> apa_eb2b_run(", "apa_eb2b"),
    ("eb3.cuh", "std::vector<NDArray> apa_eb3_run(", "apa_eb3"),
    ("eb4.cuh", "std::vector<NDArray> apa_eb4_run(", "apa_eb4"),
]


OPEN = re.compile(r"\)\s*\{\n")


def function_body(path, head):
    """Text of the function whose signature starts with `head`, from after its opening brace to its closing one."""
    text = path.read_text()
    body = text[OPEN.search(text, text.index(head)).end():]
    return body[:body.index("\n}\n")]


@pytest.mark.parametrize("file,head,entry", ENTRIES, ids=[e[2] for e in ENTRIES])
def test_r3p_every_packed_attention_entry_checks_named_extents_first(file, head, entry):
    body = function_body(ENGINE / "tensor_cuda/src/apa_spo" / file, head)
    code = [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith("//")]
    assert code[0] == f'apa_spo_require_attention_extents("{entry}",{{{{"q",&q}},{{"k",&k}},{{"v",&v}}}});', code[:2]


def guard_blocks():
    text = (ENGINE / "tensor_cuda/src/apa_spo/launcher.cuh").read_text()
    out = []
    for tag in ("APA_P1F1", "APA_P1F2", "APA_P1F3"):
        a, b = f"// {tag}_BEGIN guards\n", f"// {tag}_END guards"
        out.append(text[text.index(a) + len(a):text.index(b)])
    return "\n".join(out)


def eb2b_prologue():
    """apa_eb2b_run from its first statement up to (not including) its GQA check line."""
    body = function_body(ENGINE / "tensor_cuda/src/apa_spo/eb2b.cuh", "std::vector<NDArray> apa_eb2b_run(")
    return body[:body.index("  if(B<1 || H<1")]


HOST_MAIN = r'''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define TC_APA_SPLITK_PART_KEYS 2048
// Stub: a shape only. The guards read shape (and data_ptr for the alignment helper, unused here).
struct NDArray { std::vector<int64_t> shape; int ndim() const { return (int)shape.size(); }
                 const void* data_ptr() const { return nullptr; } };
GUARDS
void eb2b_prologue(const NDArray& q,const NDArray& k,const NDArray& v,const std::string& carry_mode,float delta,
                   bool diag,bool profile){
PROLOGUE
  (void)B;(void)H;(void)KVH;(void)S;(void)diag;
}
static void run(const char* id,std::vector<int64_t> qs,std::vector<int64_t> ks){
  NDArray q{qs},k{ks},v{ks};
  std::string kind="none",what;
  try{ eb2b_prologue(q,k,v,"partition",2.0f,false,false); }
  catch(const std::invalid_argument& e){ kind="ValueError"; what=e.what(); }
  catch(const std::runtime_error& e){ kind="RuntimeError"; what=e.what(); }
  std::printf("CASE\t%s\t%s\t%s\n",id,kind.c_str(),what.c_str());
}
int main(){
  run("eb2b_BH_at_bound",{1,536870911,1,128},{1,1,1,128});
  run("eb2b_BH_past_bound",{1,536870912,1,128},{1,134217728,1,128});
  run("eb2b_BH_2p32",{65536,65536,1,128},{65536,8192,1,128});
  run("eb2b_k_S_2p31",{1,8,1,128},{1,2,2147483648LL,128});
  run("eb2b_q_B_2p31",{2147483648LL,8,1,128},{2147483648LL,2,16,128});
  run("eb2b_q_B0",{0,8,1,128},{1,2,16,128});
  run("eb2b_valid",{1,8,1,128},{1,2,16,128});
  return 0;
}
'''


@pytest.fixture(scope="module")
def host_cases(tmp_path_factory):
    d = tmp_path_factory.mktemp("r3p_host")
    src = HOST_MAIN.replace("GUARDS", guard_blocks()).replace("PROLOGUE", eb2b_prologue())
    (d / "r3p_host.cpp").write_text(src)
    cc = subprocess.run(["timeout", "300s", "g++", "-std=c++17", "-O0", "-fwrapv", "-o", str(d / "r3p_host"),
                         str(d / "r3p_host.cpp")], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr[-3000:]
    run = subprocess.run(["timeout", "60s", str(d / "r3p_host")], capture_output=True, text=True,
                         env=dict(os.environ, TC_APA_SP="1"))
    assert run.returncode == 0, run.stderr[-2000:]
    rows = {}
    for line in run.stdout.splitlines():
        if line.startswith("CASE\t"):
            _, cid, kind, what = line.split("\t", 3)
            rows[cid] = (kind, what)
    assert len(rows) == 7, run.stdout
    return rows


def test_r3p_eb2b_product_at_its_bound_is_not_refused_by_the_product_check(host_cases):
    kind, what = host_cases["eb2b_BH_at_bound"]
    assert kind == "none", (kind, what)      # the real entry then refuses it by the GQA check (LF3)


def test_r3p_eb2b_product_one_past_its_bound_raises_value_error_naming_it(host_cases):
    kind, what = host_cases["eb2b_BH_past_bound"]
    assert kind == "ValueError" and "apa_eb2b" in what and "B*H" in what and "536870912" in what, (kind, what)


def test_r3p_eb2b_product_that_wraps_a_signed_int_raises_value_error_naming_it(host_cases):
    kind, what = host_cases["eb2b_BH_2p32"]
    assert kind == "ValueError" and "apa_eb2b" in what and "B*H" in what and "4294967296" in what, (kind, what)


@pytest.mark.parametrize("cid,operand", [("eb2b_k_S_2p31", "operand 'k' axis 2"), ("eb2b_q_B_2p31", "operand 'q' axis 0")])
def test_r3p_attention_extent_above_int_max_raises_value_error_naming_entry_and_operand(host_cases, cid, operand):
    kind, what = host_cases[cid]
    assert kind == "ValueError" and what.startswith("apa_eb2b: ") and operand in what and "2147483648" in what, (kind, what)


def test_r3p_zero_extent_keeps_the_runtime_error(host_cases):
    kind, what = host_cases["eb2b_q_B0"]
    assert kind == "RuntimeError" and "dimensions must be positive" in what, (kind, what)
    assert host_cases["eb2b_valid"][0] == "none"


def test_r3p_guard_refuses_only_above_int_max_and_products_form_in_int64():
    text = (ENGINE / "tensor_cuda/src/apa_spo/launcher.cuh").read_text()
    block = text[text.index("// APA_P1F3_BEGIN guards"):text.index("// APA_P1F3_END guards")]
    guard = block[block.index("inline void apa_spo_require_attention_extents"):block.index("inline void apa_spo_require_at_most")]
    assert "if(n>kApaSpoIntMax)" in guard and "throw std::invalid_argument" in guard and "n<=0" not in guard
    eb2b = function_body(ENGINE / "tensor_cuda/src/apa_spo/eb2b.cuh", "std::vector<NDArray> apa_eb2b_run(")
    assert eb2b.index("apa_spo_product({q.shape[0],q.shape[1]})") < eb2b.index("int B=q.shape[0]") < eb2b.index("<<<")
    assert not re.search(r"B\*H\s*>", eb2b), "no signed int B*H comparison remains in EB2b"
    pl = (ENGINE / "tensor_cuda/src/apa_spo/packed_launcher.cuh").read_text()
    for head in ("std::pair<NDArray,NDArray> apa_spq1_prepare(", "void apa_spq1_pack("):
        body = pl[pl.index(head):]
        body = body[:body.index("\n}\n")]
        assert "const int64_t N64=k.shape[2]-(int64_t)start;" in body and "N=S-start" not in body, head
