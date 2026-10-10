"""APA-REGT2: the packed identity-V leak detector (pIdV), ported from APA-LD3 for the production regression tests.

pIdV (APA-LD3 `pidv_layouts`, `pidv_operand`, `pidv_check`; `pidv_cell` from the LD3 campaign runner) -- the
strongest identity-style baseline APA-LD3 could build without power-of-two labels:
  * V in {0, 1} (exact in every dtype); oracle = exact zero / nonzero of every output entry (0*w = 0 exactly,
    and V, w >= 0 cannot cancel), so no reference run, no repeat run and no determinism dependence.
  * cut-indicator columns: V = 1 on positions >= t (or < t) in every slab -- identity-V generalised from one key
    per column to a key SET per column; output must be nonzero iff the row's true visible set meets the set.
    One column per row boundary of the TRUE visibility (APA-LD2 `frontier_cuts`), replicated in every slab.
  * slab-colour columns: V = 1 on ALL positions of one slab s, 0 in every other slab; output must be nonzero
    iff the row reads slab s. Any cross-slab read of any key lands in a column that must be exactly zero.
    Colours are interleaved over the columns (column k -> slab k mod nslab), spread evenly.
  * one kernel run per cell whenever #cuts + B*KVH <= VD (every production-layout cell: 64 cuts + 16 slabs <= 128);
    otherwise more runs, until every slab has a colour column and every cut has a column.

What it catches (APA-LD3 receipts, docs/APA_LD3_REPORT.md in the apa-ld3 branch): wrong-source reads (wrong KV
head, wrong batch, batch stride, a wrong-head interior key tile, also on one slab only) and causal-edge leaks
(one extra future key; a single tile-edge row), 1 run each, with 0 false alarms on 35 correct-kernel hosts.

What it CANNOT catch, by design (say so before relying on a PASS):
  * IN-SLAB REORDERING OF VISIBLE KEYS. A kernel that reads the right slab but permutes keys inside each row's true
    visible set (or reads a wrong-but-visible key of the same slab) produces the same zero/nonzero pattern: every
    column is an OR over a key set, and the set a row reads is unchanged. Every V-tag method shares this blind spot.
    The CPU test `test_pidv_blind_to_in_slab_reordering_of_visible_keys` pins it.
  * Numerics, selection rules, statistics paths (see the ZWS tests), backward passes, kernel speed.
  * fp16 tiny-weight underflow: an fp16 kernel that flushes a legitimately tiny weight to 0 would read as a "blind"
    entry; the registered fp16 cells (S <= 1025, LD1 operand regime |score| <= 3) stay far above the fp16 floor.
    Larger fp16 S is not covered.
  * The GRM gemma4 sliding path (production tests must not import GRM).

Prior art (Prior Art Directive; kept from APA-LD3's annotation at scripts/apa_ld3_pdwl2.py; external items
unverified -- lead to check; search terms: "non-adaptive group testing multiplexing", "combinatorial group testing
Du Hwang", "tagged value attention mask test", "Dorfman 1943 group testing", "DeMillo Lipton Sayward 1978",
"FlashAttention-2 bottom-right causal", "identity matrix probe attention mask test"):
  - Dorfman (1943) group testing and non-adaptive / multiplexed group testing (e.g. Du & Hwang, combinatorial group
    testing): several independent tests share one measurement when their responses are separable. TAKEN as the
    general idea; pIdV's set-indicator columns are literally group tests (OR over a key set), and the column
    packing is a non-adaptive multiplexed design whose separability comes from the kernel's column independence.
  - identity-V (APA-LD1 / APA-LD2 baselines idV-F, idV-A: V = one-hot keys, output must be exactly zero off the
    visible set): TAKEN -- the exact zero/nonzero oracle; pIdV generalises one key per column to a key SET per column.
  - DeMillo, Lipton & Sayward (1978), mutation testing: the registered known-bad builds and the in-test Python
    mutants (bstride, offby1) measure the detector.
  - APA-LD2 `frontier_cuts` (imported, not copied): every row boundary of the true visibility.
  - Dao (2023) FlashAttention-2: the bottom-right causal convention the cut columns check.
  - MINE (APA-LD3): the pIdV design (set-indicator columns + interleaved slab colours).
  - MINE (APA-REGT2): nothing algorithmic; the port, the three extra entry-family adapters (lines taken from
    APA-LD1's adapter, below), the scoped TC_APA_SP switch and the caller-owned margin-table directory.

APA-REGT2 port (from the apa-ld3 branch, 24f8b11; files read from the fork's disk, sha256 receipts in the REGT2
report). Every detector line below is verbatim from scripts/apa_ld3_pdwl2.py or scripts/apa_ld3_run.py; lines
were only removed (v2, labels, twins, column_check and the campaigns are not ported). Changes, each marked
"APA-REGT2" at its site: the `V2` alias (pidv_cell called the detector as V2.<name>); `PidvAttn`, which adds to the
production adapter `apa_ld1_pdwl.Attn` (imported, not edited) the three entry families pIdV ran on in APA-LD3 that
the APA-REGT1 port did not keep (sdpa fused causal softmax, apa_int4, apa_sp); `make_cell`, the LD2 `attn2`
constructor restricted to these families; `bind`, which hands the importable engine to both modules. Nothing in
this file writes to the file system except `synthetic_margins`, into a directory its caller owns.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import apa_ld1_pdwl as P
import apa_ld2_pdwl as P2
from apa_ld1_pdwl import down, expand_heads, up

V2 = sys.modules[__name__]   # APA-REGT2: pidv_cell (apa_ld3_run.py) called this module as V2

BASE, TWIN, CUT, LABEL, COLOR = 0, 1, 2, 3, 4   # APA-LD3 kind constants; pIdV uses CUT and COLOR

tc = None
F = None
Q = None


def bind(tc_, F_, Q_):
    """APA-REGT2: hand the importable engine (tensor_cuda, its functional and quant modules) to the production adapter
    module (apa_ld1_pdwl, whose Attn reads its globals tc / F / Q) and to this module."""
    global tc, F, Q
    P.tc, P.F, P.Q = tc_, F_, Q_
    tc, F, Q = tc_, F_, Q_


# ------------------------------------------------------------------ helpers (verbatim, apa_ld3_pdwl2.py)
def spread(free, n):
    """n positions from the list `free`, evenly spaced (strictly increasing); returns (picked, rest)."""
    free = list(free)
    m = len(free)
    if n <= 0:
        return [], free
    if n > m:
        raise ValueError(f"spread: {n} > {m}")
    off = (m // n) // 2
    idx = [(k * m) // n + off for k in range(n)]
    picked = [free[i] for i in idx]
    ps = set(idx)
    return picked, [free[i] for i in range(m) if i not in ps]


def _empty_layout(VD):
    return dict(kind=np.full(VD, -1, np.int64), cut_t=np.zeros(VD, np.int64), cut_ge=np.zeros(VD, bool),
                pat=np.full(VD, -1, np.int64), slab=np.full(VD, -1, np.int64), base=-1)


def _cut_nt(lo, hi, t, ge):
    """(B,H,L,n) count of tagged positions in each row's TRUE visible interval, per cut column."""
    lo_, hi_ = lo[..., None], hi[..., None]
    a = np.where(ge, np.maximum(lo_, t), lo_)
    b_ = np.where(ge, hi_, np.minimum(hi_, t - 1))
    return np.maximum(0, b_ - a + 1)


# ------------------------------------------------------------------ pIdV: packed identity-V baseline (verbatim)
def pidv_layouts(cuts, nslab, VD):
    """Run 1: every cut (up to VD - 1) as a set-indicator column, the rest slab colours (interleaved, spread).
    Further runs only while some slab has no colour column yet (nslab > free columns) or cuts remain."""
    out = []
    cuts = list(cuts)
    covered = 0
    while True:
        lay = _empty_layout(VD)
        ch, cuts = cuts[:VD - 1], cuts[VD - 1:]
        ncol = VD - len(ch)
        col_pos, free = spread(list(range(VD)), ncol)
        for k, c in enumerate(col_pos):
            lay["kind"][c], lay["slab"][c] = COLOR, (covered + k) % nslab
        covered += ncol
        for (t, side), c in zip(ch, free):
            lay["kind"][c], lay["cut_t"][c], lay["cut_ge"][c] = CUT, int(t), side == "ge"
        out.append(lay)
        if not cuts and covered >= nslab:
            return out


def pidv_operand(B, KVH, S, lay):
    kind = lay["kind"]
    VD = len(kind)
    X = np.zeros((B, KVH, S, VD), np.float32)
    for c in np.where(kind == COLOR)[0]:
        s = int(lay["slab"][c])
        X[s // KVH, s % KVH, :, c] = 1.0
    cc = np.where(kind == CUT)[0]
    ge, lt = cc[lay["cut_ge"][cc]], cc[~lay["cut_ge"][cc]]
    if ge.size:
        t = lay["cut_t"][ge]
        j0 = int(t.min())
        m = (np.arange(j0, S)[:, None] >= t[None, :]).astype(np.float32)
        X[:, :, j0:, ge] = m[None, None]
    if lt.size:
        t = lay["cut_t"][lt]
        j0, j1 = int(t.min()), int(t.max())
        X[:, :, :j0, lt] = 1.0
        m = (np.arange(j0, j1)[:, None] < t[None, :]).astype(np.float32)
        X[:, :, j0:j1, lt] = m[None, None]
    return X


def pidv_check(run, B, KVH, S, lo, hi, rep, cuts, VD, *, layouts=None):
    """packed identity-V. Expected: cut column nonzero iff the row's TRUE visible set meets the tagged set;
    colour column of slab s nonzero iff the row reads slab s (and sees >= 1 key); every other entry EXACTLY 0.
    Stops at the first violating run."""
    t0 = time.perf_counter()
    nslab = B * KVH
    lays = layouts or pidv_layouts(cuts, nslab, VD)
    Bq, H, L = lo.shape
    s_of = np.arange(B)[:, None] * KVH + (np.arange(H) // rep)[None, :]
    nonempty = hi >= lo
    runs, t_kill, first, nviol = 0, None, None, 0
    for lay in lays:
        X = pidv_operand(B, KVH, S, lay)
        Y = run(X)
        runs += 1
        kind = lay["kind"]
        exp = np.zeros(Y.shape, bool)
        cc = np.where(kind == CUT)[0]
        if cc.size:
            exp[..., cc] = _cut_nt(lo, hi, lay["cut_t"][cc], lay["cut_ge"][cc]) > 0
        col = np.where(kind == COLOR)[0]
        if col.size:
            exp[..., col] = (s_of[:, :, None, None] == lay["slab"][col][None, None, None, :]) & nonempty[..., None]
        viol = (Y != 0) != exp
        n = int(np.count_nonzero(viol))
        if n:
            nviol = n
            t_kill = time.perf_counter() - t0
            f = np.argwhere(viol)[0]
            c = int(f[3])
            first = dict(b=int(f[0]), h=int(f[1]), i=int(f[2]), col=c, kind=int(kind[c]),
                         slab=int(lay["slab"][c]) if kind[c] == COLOR else None,
                         cut=[int(lay["cut_t"][c]), "ge" if lay["cut_ge"][c] else "lt"] if kind[c] == CUT else None,
                         got_nonzero=bool(Y[tuple(f)] != 0))
            break
    return dict(mode="pidv", verdict="ALARM" if t_kill is not None else "PASS", violations=nviol, runs=runs,
                runs_full=len(lays), first_violation=first, t_kill=None if t_kill is None else round(t_kill, 4),
                seconds=round(time.perf_counter() - t0, 4))


# ------------------------------------------------------------------ cell runner (verbatim, apa_ld3_run.py:133)
def pidv_cell(e):
    t0 = time.perf_counter()
    lo, hi, _ = e.intervals()
    cuts = P2.frontier_cuts(lo, hi, e.S)
    r = V2.pidv_check(e.run, e.B, e.KVH, e.S, lo, hi, e.rep, cuts, e.VD)
    r.update(method="pidv", wall=round(time.perf_counter() - t0, 4))
    if r["t_kill"] is not None:
        r["t_kill_wall"] = round(r["t_kill"] + (r["wall"] - r["seconds"]), 4)
    return r


# =================================================================================================
# Engine adapter additions (APA-REGT2). The production adapter apa_ld1_pdwl.Attn (APA-REGT1's port of APA-LD1's
# scripts/apa_ld1_entries.py) runs sdpa, apa_quant and apa_sel. APA-LD3's pIdV also ran on the sdpa fused causal
# softmax path (sdpa_fused_cs), apa_int4 and apa_sp, so their lines come back here, from apa_ld1_entries.py @ 24f8b11
# (lines 132-141 in _prep, 185-191 / 210-211 / 214-216 in run, 412-423 _synthetic_margins), verbatim except:
#   * apa_sp's TC_APA_SP=1 switch is set and restored around the call (LD1 set it process-wide in _prep, which
#     would leak into every later test in the same pytest process); the restore follows LD1's own sdpa_fused_nc
#     pattern (apa_ld1_entries.py:192-201);
#   * the synthetic SP2 margin table (PDWL / pIdV do not depend on e_q: the operand is V, the table only feeds the
#     SP epsilon rule) is written into a directory the caller owns (the tests pass a pytest tmp dir), not under
#     artifacts/; content and the RegisteredMargins.load validation path are unchanged.
# Geometry table GEO: the three entries pIdV's edge and production cells use, verbatim from apa_ld1_run.py:30-34.
# Prior art: none known to me for the adapters themselves (APA-LD1's own code).
# =================================================================================================
GEO = {
    "QW": dict(H=16, KVH=8, D=128),                         # Qwen3 16q/8kv D128
    "G4G": dict(H=16, KVH=1, D=512, scale=1.0),              # Gemma-4 global (MQA)
    "MC3": dict(H=40, KVH=40, D=96, VD=64),                  # MiniCPM3 MLA composite 96 / value 64
}
EXTRA_FAMILIES = ("sdpa_fused_cs", "apa_int4", "apa_sp")
SP_MARGINS_NAME = "synthetic_sp2_margins_PDWL_ONLY.json"


def synthetic_margins(dirpath):
    """SP2 wrapper requires a RegisteredMargins table; PDWL is V-linear and independent of e_q, so a
    SYNTHETIC table (same shape as test_apa_sp2.example_table) is used and labelled as such.
    APA-REGT2: written into `dirpath` (caller-owned), not artifacts/apa_ld1/."""
    from tensor_cuda.apa_sp2 import REGISTRATION_SHA256
    d = Path(dirpath) / SP_MARGINS_NAME
    if not d.exists():
        d.write_text(json.dumps(dict(registration_sha256=REGISTRATION_SHA256, status="FROZEN",
                                     quantizer="turboquant_lloyd_max_rotated_fp32", measurement_classes=64,
                                     statistic="maximum_over_all_registered_G2_valid_scores",
                                     entries={f"{b}:{D}": dict(e_q=.5, count=1, mean=.5, max=.5, p99_9=.5)
                                              for b in (2, 4) for D in (64, 128)})))
    return tc.RegisteredMargins.load(d, expected_sha256=hashlib.sha256(d.read_bytes()).hexdigest())


class PidvAttn(P.Attn):
    """production Attn (sdpa / apa_quant / apa_sel and the Python mutants bstride / offby1 / tile / topleft) plus the
    entry families sdpa_fused_cs, apa_int4 and apa_sp. margins_dir: required for apa_sp only."""

    def __init__(self, *a, margins_dir=None, **kw):
        self.margins_dir = margins_dir
        super().__init__(*a, **kw)

    def _prep(self):
        f = self.family
        if f == "apa_sp":
            if self.margins_dir is None:
                raise ValueError("PidvAttn: apa_sp needs margins_dir (a caller-owned directory)")
            kq = tc.quantize_sp2_keys(up(self.k, "float32"), 2)
            self.kq = down(kq)
            self.sigma_rows = [self.k, self.kq]
            self.kq_np = self._kv_layout(self.kq)
        if f in ("apa_int4", "apa_int4_ws", "apa_gemm", "apa_gemm_cached", "apa_int4_nc"):
            self.sigma_rows = [self.k * P.QUANT_SLACK]   # internal quantizer: slack on |k|
        super()._prep()
        if f == "apa_sp":
            self.margins = synthetic_margins(self.margins_dir)

    def run(self, X):
        dt, f = self.dtype, self.family
        if f not in EXTRA_FAMILIES:
            return super().run(X)
        rep, how = self._heads_for_kernel()
        v_t = up(expand_heads(self._kv_layout(X), rep, how), dt)
        q, k = self.q_t, self.k_t
        caus = self.causal
        if self.mutant == "topleft":
            caus = True       # square truncation: bottom-right on (L, L) == top-left on (L, S)
        with tc.no_grad():
            if f == "sdpa_fused_cs":
                old = F.USE_FUSED_SOFTMAX
                F.USE_FUSED_SOFTMAX = True
                try:
                    y = F.scaled_dot_product_attention(q, k, v_t, is_causal=caus, scale=self.scale)
                finally:
                    F.USE_FUSED_SOFTMAX = old
            elif f == "apa_int4":
                y = tc.apa_selective_attention_int4(q, k, v_t, self.scale, self.zthr, caus)
            elif f == "apa_sp":
                old = os.environ.get("TC_APA_SP")
                os.environ["TC_APA_SP"] = "1"
                try:
                    y = tc.apa_selective_attention_sp(q, k, self.kq_t, v_t, self.scale, 0.05, bulk_bits=2,
                                                      margins=self.margins, is_causal=caus)
                finally:
                    if old is None:
                        os.environ.pop("TC_APA_SP", None)
                    else:
                        os.environ["TC_APA_SP"] = old
            else:
                raise ValueError(f)
        return down(y)


def make_cell(fam, dt, geo, L, S, *, B=2, mutant=None, seed=0, margins_dir=None, **kw):
    """APA-LD2 attn2 (apa_ld2_run.py:94-101) for the causal entry families above: GEO lookup, causal, seed 0 and the
    production refine threshold (zthr default) -- the operand instance APA-LD3 ran pIdV on."""
    g = dict(GEO[geo])
    causal = True
    return PidvAttn(fam, dt, B, g.pop("H"), g.pop("KVH"), L, S, g.pop("D"), g.pop("VD", None), causal=causal,
                    mutant=mutant, seed=seed, scale=g.pop("scale", None), margins_dir=margins_dir, **kw)


def operand_footprint_bytes(geo, L, S, *, B=2):
    """host bytes of the fp32 q, k and v operands one cell builds (the S = 98,304 budget note)."""
    g = GEO[geo]
    D, VD = g["D"], g.get("VD", g["D"])
    return 4 * B * (g["H"] * L * D + g["KVH"] * S * (D + VD))

