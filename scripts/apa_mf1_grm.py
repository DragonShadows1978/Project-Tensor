"""APA-MF1 item 6: GRM live path, Qwen3-1.7B attention_mode="apa", before / A / A+B.

GraftRepository is imported READ-ONLY. This worktree's tensor_cuda is imported
first (PYTHONPATH) and asserted; GRM's own sys.path insert of the production
engine is then a no-op for the already-imported package. The arm switch is the
module attribute GRM calls at run time (`tc.apa_quant_attention`), pointed at
the before copy, the A snapshot, or the tree's A+B function; nothing in GRM is
edited. Standard mode on the same chunking is the reference.

phase chunked: prefill 8192 tokens in 16 chunks of 512 (KV cache), then 32
teacher-forced single-token steps. phase control: unchunked prefill (S == L) of
the first N tokens, then 32 teacher-forced steps.
Metric: per step KL(P_standard || P_arm) and top-1 agreement over the 32 step
distributions (prefill-final reported separately); true-token NLL per arm.

Prior art: teacher-forced next-token KL against a reference model is standard
LM-evaluation practice (e.g. quantization/KV-cache compression papers; not
verified for a specific first source — lead to check: "KL divergence teacher
forcing quantized model evaluation"). Harness: mine.
"""
from __future__ import annotations

import argparse
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
assert tc.__file__.startswith(ROOT) and tc._C.__file__.startswith(ROOT), (tc.__file__, tc._C.__file__)
from tensor_cuda import quant as QAB  # noqa: E402

GRM = "/mnt/ForgeRealm/GraftRepository"
sys.path.insert(0, GRM)
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import core.mistral7b_tc as M  # noqa: E402

assert M.tc is tc and M.F is tc.functional
for name, mod in list(sys.modules.items()):
    if name == "tensor_cuda" or name.startswith("tensor_cuda.") or name == "_tensor_cuda":
        f = getattr(mod, "__file__", None)
        assert f is None or f.startswith(ROOT), (name, f)

P = 0.15
CHUNK = 512
NSTEP = 32


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tokens(n):
    from transformers import AutoTokenizer
    import pandas as pd
    snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*")))[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    pq = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/test-00000-of-00001.parquet"))[0]
    text = "".join(pd.read_parquet(pq)["text"].tolist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= n, len(ids)
    return np.asarray(ids[:n], dtype=np.int64), pq


def run(model, ids, prefill, chunk):
    caches = None
    with tc.no_grad():
        for c0 in range(0, prefill, chunk):
            logits, caches = model(ids[None, c0:min(prefill, c0 + chunk)], kv_caches=caches,
                                   position_offset=c0, last_token_only=True)
        pre = logits.float().numpy().reshape(-1).astype(np.float32)
        steps = []
        for j in range(NSTEP):
            logits, caches = model(ids[None, prefill + j:prefill + j + 1], kv_caches=caches,
                                   position_offset=prefill + j, last_token_only=True)
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
    top1 = (ref.argmax(-1) == arm.argmax(-1))
    nll = -la[np.arange(len(targets)), targets]
    return dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()), kl_per_step=kl.tolist(),
                top1_agree=int(top1.sum()), top1_n=int(top1.size), nll_true_mean=float(nll.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("chunked", "control"), required=True)
    ap.add_argument("--control-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prefill = 8192 if a.phase == "chunked" else a.control_len
    chunk = CHUNK if a.phase == "chunked" else prefill
    ids, pq = tokens(prefill + NSTEP + 1)
    arms_fn = {
        "before": load_mod("tensor_cuda._mf1_before", os.path.join(ROOT, "artifacts/apa_mf1/before/tensor_cuda__tensor_cuda__quant.py")).apa_quant_attention,
        "A": load_mod("tensor_cuda._mf1_A", os.path.join(ROOT, "artifacts/apa_mf1/variants/quant_A.py")).apa_quant_attention,
        "AB": QAB.apa_quant_attention,
    }
    orig = tc.apa_quant_attention
    assert orig is QAB.apa_quant_attention
    t0 = time.time()
    model, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard")
    t_load = time.time() - t0
    res = {"phase": a.phase, "prefill": prefill, "chunk": chunk, "steps": NSTEP, "engine": tc._C.__file__,
           "tensor_cuda": tc.__file__, "grm": GRM, "wikitext_parquet": pq,
           "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "t_load_s": t_load, "arms": {}}
    logits = {}
    order = ["standard", "before", "A", "AB", "standard_repeat"]
    for arm in order:
        mode = "standard" if arm.startswith("standard") else "apa"
        model.set_attention_mode(mode, refine_percentile=P)
        for L_ in model.layers:
            assert L_.self_attn.attention_mode == mode and L_.self_attn.bulk_bits == 8
            assert L_.self_attn.refine_percentile == P
        tc.apa_quant_attention = arms_fn.get(arm, orig)
        t1 = time.time()
        try:
            pre, steps = run(model, ids, prefill, chunk)
        finally:
            tc.apa_quant_attention = orig
        logits[arm] = (pre, steps)
        res["arms"][arm] = {"t_s": time.time() - t1,
                            "sha256_logits": hashlib.sha256(pre.tobytes() + steps.tobytes()).hexdigest()}
        print(arm, res["arms"][arm], flush=True)
    targets = ids[prefill + 1:prefill + 1 + NSTEP]
    ref_pre, ref_steps = logits["standard"]
    for arm in order:
        pre, steps = logits[arm]
        c = compare(ref_steps, steps, targets)
        c["prefill_final"] = compare(ref_pre[None], pre[None], ids[prefill:prefill + 1])
        res["arms"][arm].update(c)
        print(arm, {k: v for k, v in c.items() if k != "kl_per_step"}, flush=True)
    res["before_vs_A_bytes_equal"] = (logits["before"][0].tobytes() == logits["A"][0].tobytes()
                                      and logits["before"][1].tobytes() == logits["A"][1].tobytes())
    res["standard_repeat_bytes_equal"] = (logits["standard"][1].tobytes() == logits["standard_repeat"][1].tobytes())
    res["t_total_s"] = time.time() - t0
    with open(a.out, "x") as f:
        json.dump(res, f, indent=1)
    print("before_vs_A_bytes_equal", res["before_vs_A_bytes_equal"], "standard_repeat_bytes_equal",
          res["standard_repeat_bytes_equal"], "WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
