"""APA-MF2 G3/item 5: GRM Qwen3-1.7B on one wikitext-2 window, read-only GRM import.

Prefill 8192 tokens in 16 chunks of 512 (KV cache), then 32 teacher-forced steps
(MF1 item-6 "chunked" protocol, scripts/apa_mf1_grm.py). Arms, one process, one model:
  standard_after   tree functional.py (SDPA with the -inf predicate mask)
  standard_before  MF1/fd32b6d functional.py (-1e4 additive bias), loaded from file
  apa_mf1AB        attention_mode="apa", MF1 A+B quant.py + MF1 functional.py (files)
  apa_mf2          attention_mode="apa", tree quant.py + functional.py
  standard_after_repeat   determinism control
Arm switching uses the module attributes GRM reads at call time (F._causal_mask,
F.scaled_dot_product_attention, tc.apa_quant_attention); nothing in GRM is edited.
Item 2 claim: standard_before and standard_after logits are byte-identical.
P3: |KL(apa_mf2) - KL(apa_mf1AB)| / KL(apa_mf1AB) < 0.05 (KL vs standard_after).
Prior art: teacher-forced KL against a reference (standard LM evaluation; unverified --
lead to check "KL divergence teacher forcing quantized model evaluation"). Harness:
MF1's, adapted (mine).
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import functional as F  # noqa: E402
from tensor_cuda import quant as Q  # noqa: E402

assert tc.__file__.startswith(ROOT) and tc._C.__file__.startswith(ROOT), (tc.__file__, tc._C.__file__)
GRM = "/mnt/ForgeRealm/GraftRepository"
sys.path.insert(0, GRM)
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import core.mistral7b_tc as M  # noqa: E402

assert M.tc is tc and M.F is F
for name, mod in list(sys.modules.items()):
    if name == "tensor_cuda" or name.startswith("tensor_cuda.") or name == "_tensor_cuda":
        f = getattr(mod, "__file__", None)
        assert f is None or f.startswith(ROOT), (name, f)

P, CHUNK, NSTEP, PREFILL = 0.15, 512, 32, 8192
BEF = os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BFN = load("tensor_cuda._mf2_grm_bef_functional", os.path.join(BEF, "tensor_cuda__tensor_cuda__functional.py"))
BQ = load("tensor_cuda._mf2_grm_bef_quant", os.path.join(BEF, "tensor_cuda__tensor_cuda__quant.py"))


@contextlib.contextmanager
def arm_ctx(arm):
    saved = (F._causal_mask, F.scaled_dot_product_attention, tc.apa_quant_attention)
    try:
        if arm in ("standard_before", "apa_mf1AB"):
            F._causal_mask, F.scaled_dot_product_attention = BFN._causal_mask, BFN.scaled_dot_product_attention
        if arm == "apa_mf1AB":
            tc.apa_quant_attention = BQ.apa_quant_attention
        yield
    finally:
        F._causal_mask, F.scaled_dot_product_attention, tc.apa_quant_attention = saved


def tokens(n):
    from transformers import AutoTokenizer
    import pandas as pd
    snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*")))[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    pq = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/test-00000-of-00001.parquet"))[0]
    text = "".join(pd.read_parquet(pq)["text"].tolist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return np.asarray(ids[:n], dtype=np.int64), pq


def run(model, ids):
    caches = None
    with tc.no_grad():
        for c0 in range(0, PREFILL, CHUNK):
            logits, caches = model(ids[None, c0:c0 + CHUNK], kv_caches=caches, position_offset=c0, last_token_only=True)
        pre = logits.float().numpy().reshape(-1).astype(np.float32)
        steps = []
        for j in range(NSTEP):
            logits, caches = model(ids[None, PREFILL + j:PREFILL + j + 1], kv_caches=caches,
                                   position_offset=PREFILL + j, last_token_only=True)
            steps.append(logits.float().numpy().reshape(-1).astype(np.float32))
    del caches
    return pre, np.stack(steps)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def compare(ref, arm, targets):
    lr, la = logsm(ref), logsm(arm)
    kl = (np.exp(lr) * (lr - la)).sum(-1)
    top1 = ref.argmax(-1) == arm.argmax(-1)
    nll = -la[np.arange(len(targets)), targets]
    return dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()), kl_per_step=kl.tolist(),
                top1_agree=int(top1.sum()), top1_n=int(top1.size), nll_true_mean=float(nll.mean()))


def main():
    out = sys.argv[1]
    ids, pq = tokens(PREFILL + NSTEP + 1)
    t0 = time.time()
    model, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard")
    res = {"prefill": PREFILL, "chunk": CHUNK, "steps": NSTEP, "engine": tc._C.__file__, "tensor_cuda": tc.__file__,
           "grm": GRM, "wikitext_parquet": pq, "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
           "t_load_s": time.time() - t0, "arms": {}}
    logits = {}
    order = ["standard_after", "standard_before", "apa_mf1AB", "apa_mf2", "standard_after_repeat"]
    for arm in order:
        mode = "standard" if arm.startswith("standard") else "apa"
        model.set_attention_mode(mode, refine_percentile=P)
        for L_ in model.layers:
            assert L_.self_attn.attention_mode == mode and L_.self_attn.bulk_bits == 8
        t1 = time.time()
        with arm_ctx(arm):
            pre, steps = run(model, ids)
        logits[arm] = (pre, steps)
        res["arms"][arm] = {"t_s": time.time() - t1,
                            "sha256_logits": hashlib.sha256(pre.tobytes() + steps.tobytes()).hexdigest(),
                            "finite": bool(np.isfinite(pre).all() and np.isfinite(steps).all())}
        print(arm, res["arms"][arm], flush=True)
    targets = ids[PREFILL + 1:PREFILL + 1 + NSTEP]
    ref_pre, ref_steps = logits["standard_after"]
    for arm in order:
        pre, steps = logits[arm]
        c = compare(ref_steps, steps, targets)
        c["prefill_final"] = compare(ref_pre[None], pre[None], ids[PREFILL:PREFILL + 1])
        res["arms"][arm].update(c)
        print(arm, {k: v for k, v in c.items() if k != "kl_per_step"}, flush=True)
    res["standard_before_vs_after_bytes_equal"] = (
        logits["standard_before"][0].tobytes() == logits["standard_after"][0].tobytes()
        and logits["standard_before"][1].tobytes() == logits["standard_after"][1].tobytes())
    res["standard_repeat_bytes_equal"] = (logits["standard_after"][1].tobytes() == logits["standard_after_repeat"][1].tobytes()
                                          and logits["standard_after"][0].tobytes() == logits["standard_after_repeat"][0].tobytes())
    k1, k2 = res["arms"]["apa_mf1AB"]["kl_mean"], res["arms"]["apa_mf2"]["kl_mean"]
    res["P3_rel_change_kl_mean"] = (k2 - k1) / k1
    res["P3_holds"] = abs(res["P3_rel_change_kl_mean"]) < 0.05
    res["t_total_s"] = time.time() - t0
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("standard_before_vs_after_bytes_equal", res["standard_before_vs_after_bytes_equal"],
          "standard_repeat_bytes_equal", res["standard_repeat_bytes_equal"],
          "P3_rel", res["P3_rel_change_kl_mean"], "WROTE", out, flush=True)


if __name__ == "__main__":
    main()
