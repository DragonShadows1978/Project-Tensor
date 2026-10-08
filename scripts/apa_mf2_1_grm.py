"""APA-MF2.1 G3(b) GRM arm + required item 6: GRM Qwen3-1.7B on one wikitext-2 window.

ONE ARM PER PROCESS (scripts/apa_mf2_1_grm.sh): 'before' = MF2 engine + MF2 package,
'after' = MF2.1 engine + tree package. Same protocol as scripts/apa_mf2_grm.py (MF1
item-6 "chunked"): prefill 8192 tokens in 16 chunks of 512 (KV cache), then 32
teacher-forced steps; modes 'standard' and 'apa' (refine 0.15, bulk 8). GRM is
imported read-only. Logits (prefill-final + 32 steps) are saved per arm as .npy; the
compare step (--compare) checks standard logits byte-identical across arms and the
'apa' KL vs each arm's own standard reference, paired per step.
Prior art: teacher-forced KL against a reference (standard LM evaluation; unverified --
lead to check). Harness: MF2's (scripts/apa_mf2_grm.py), adapted (mine).
"""
from __future__ import annotations
import glob, hashlib, json, os, sys, time
import numpy as np

P, CHUNK, NSTEP, PREFILL = 0.15, 512, 32, 8192
OUT = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/artifacts/apa_mf2_1/grm"


def tokens(n):
    from transformers import AutoTokenizer
    import pandas as pd
    snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*")))[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    pq = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/test-00000-of-00001.parquet"))[0]
    text = "".join(pd.read_parquet(pq)["text"].tolist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return np.asarray(ids[:n], dtype=np.int64), pq


def run_arm(arm):
    import tensor_cuda as tc
    sys.path.insert(0, "/mnt/ForgeRealm/GraftRepository")
    from core.qwen3_1p7b_tc import Qwen3_1p7b_TC
    ids, pq = tokens(PREFILL + NSTEP + 1)
    t0 = time.time()
    model, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard")
    meta = {"arm": arm, "tensor_cuda": tc.__file__, "engine": tc._C.__file__,
            "engine_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "wikitext": pq, "modes": {}}
    for mode in ("standard", "apa"):
        model.set_attention_mode(mode, refine_percentile=P)
        t1 = time.time()
        caches = None
        with tc.no_grad():
            for c0 in range(0, PREFILL, CHUNK):
                logits, caches = model(ids[None, c0:c0 + CHUNK], kv_caches=caches, position_offset=c0, last_token_only=True)
            rows = [logits.float().numpy().reshape(-1)]
            for j in range(NSTEP):
                logits, caches = model(ids[None, PREFILL + j:PREFILL + j + 1], kv_caches=caches,
                                       position_offset=PREFILL + j, last_token_only=True)
                rows.append(logits.float().numpy().reshape(-1))
        del caches
        arr = np.stack(rows).astype(np.float32)
        np.save(f"{OUT}/{arm}_{mode}.npy", arr)
        meta["modes"][mode] = {"t_s": time.time() - t1, "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                               "finite": bool(np.isfinite(arr).all())}
        print(arm, mode, meta["modes"][mode], flush=True)
    meta["targets"] = ids[PREFILL + 1:PREFILL + 1 + NSTEP].tolist()
    meta["t_s"] = time.time() - t0
    json.dump(meta, open(f"{OUT}/{arm}_meta.json", "x"), indent=1)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def compare():
    L = {(a, m): np.load(f"{OUT}/{a}_{m}.npy") for a in ("before", "after") for m in ("standard", "apa")}
    meta = json.load(open(f"{OUT}/after_meta.json"))
    tg = np.array(meta["targets"])
    res = {"standard_bytes_equal": L["before", "standard"].tobytes() == L["after", "standard"].tobytes()}
    kl = {}
    for a in ("before", "after"):
        lr, la = logsm(L[a, "standard"][1:]), logsm(L[a, "apa"][1:])
        kl[a] = (np.exp(lr) * (lr - la)).sum(-1)
        res[f"{a}_apa"] = {"kl_mean": float(kl[a].mean()), "kl_max": float(kl[a].max()),
                           "top1_agree": int((L[a, "standard"][1:].argmax(-1) == L[a, "apa"][1:].argmax(-1)).sum()),
                           "nll_true_mean": float(-la[np.arange(NSTEP), tg].mean())}
    d = kl["after"] - kl["before"]
    res["paired_kl_diff_mean"] = float(d.mean())
    res["paired_kl_diff_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    res["paired_kl_diff_in_se"] = float(d.mean() / res["paired_kl_diff_se"]) if res["paired_kl_diff_se"] > 0 else 0.0
    res["rel_change_kl_mean"] = float((kl["after"].mean() - kl["before"].mean()) / kl["before"].mean())
    res["apa_logits_bytes_equal"] = L["before", "apa"].tobytes() == L["after", "apa"].tobytes()
    res["P4_within_noise_2se"] = abs(res["paired_kl_diff_in_se"]) < 2.0
    json.dump(res, open(f"{OUT}/compare.json", "x"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    compare() if sys.argv[1] == "--compare" else run_arm(sys.argv[1])
