"""APA-MF2.1: summarize artifacts/apa_mf2_1/defn/*.json into tables.md rows + totals (mine)."""
import json, collections, sys
D = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/artifacts/apa_mf2_1/defn"
F = ("n_visible", "mismatch", "in_band", "out_of_band", "invisible_selected", "thr_rows", "thr_rows_rel_err_gt_band",
     "thr_gt_band_rows_cnt_le2", "thr_gt_band_rows_cnt_gt2")
out = {}
lines = []
for name, files, keyf in (("real", ["real_Qwen3-1.7B.json", "real_Qwen3-4B.json"], lambda r: "bfloat16"),
                          ("synthetic", ["synthetic.json"], lambda r: r["dtype"]),
                          ("near_tie", ["near_tie.json"], lambda r: r.get("dtype", "bfloat16"))):
    agg = collections.defaultdict(collections.Counter); mx = collections.defaultdict(float); gap = collections.defaultdict(float)
    oob = collections.defaultdict(collections.Counter); cells = collections.Counter(); nonfinite = 0
    for f in files:
        for r in json.load(open(f"{D}/{f}"))["rows"]:
            nonfinite += not r["output_finite"]
            for who in ("composed", "kernel_h", "kernel_a"):
                if who not in r:
                    continue
                k = (who, keyf(r)); c = r[who]; cells[k] += 1
                for x in F:
                    agg[k][x] += c.get(x, 0)
                mx[k] = max(mx[k], c.get("thr_max_rel_err", 0.0)); gap[k] = max(gap[k], c.get("max_rel_gap_mismatch", 0.0))
                if c.get("out_of_band_row_visible_counts"):
                    oob[k].update(c["out_of_band_row_visible_counts"])
    out[name] = {f"{w}|{dt}": {**agg[(w, dt)], "cells": cells[(w, dt)], "thr_max_rel_err": mx[(w, dt)],
                              "max_rel_gap_mismatch": gap[(w, dt)], "oob_rows": dict(oob[(w, dt)])} for (w, dt) in agg}
    out[name]["_nonfinite_outputs"] = nonfinite
    lines.append(f"### {name} (outputs non-finite: {nonfinite})")
    lines.append("| impl | dtype | cells | visible | mismatch | in band | OUT of band | invisible sel | max gap (mism.) | thr rows | thr err > 2e-5 (cnt<=2 / >2) | max thr rel err |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for (w, dt) in sorted(agg):
        a = agg[(w, dt)]
        lines.append(f"| {w} | {dt} | {cells[(w, dt)]} | {a['n_visible']} | {a['mismatch']} | {a['in_band']} | {a['out_of_band']} | "
                     f"{a['invisible_selected']} | {gap[(w, dt)]:.2e} | {a['thr_rows']} | {a['thr_rows_rel_err_gt_band']} "
                     f"({a['thr_gt_band_rows_cnt_le2']} / {a['thr_gt_band_rows_cnt_gt2']}) | {mx[(w, dt)]:.2e} |")
    lines.append("")
json.dump(out, open(f"{D}/summary.json", "x"), indent=1)
open(f"{D}/tables.md", "x").write("\n".join(lines) + "\n")
print("\n".join(lines))
