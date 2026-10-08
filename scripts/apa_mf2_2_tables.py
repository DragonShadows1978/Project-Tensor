"""APA-MF2.2: render N1 / N2 / adjudication tables from the receipts (no GPU). Mine."""
import collections, glob, json, os
A = "artifacts/apa_mf2_2/"
out = []
cmp_ = json.load(open(sorted(glob.glob(A + "identity/compare_*.json"))[-1]))
# exact stamped names (a glob arm_mf2_* would also match arm_mf2_1_* / arm_mf2_2_*: v1 table bug)
arms = {a: json.load(open(A + f"identity/arm_{a}_{cmp_['stamp']}.json")) for a in ("fd32b6d", "mf2", "mf2_1", "mf2_2")}
for a, r in arms.items():
    assert r["provenance"]["arm"] == a and r["provenance"]["engine_sha256"] == cmp_["engines"][a]["engine_sha256"], a
out.append("## N1 (G2): -inf-first rows, one engine per process (" + os.path.basename(sorted(glob.glob(A + 'identity/compare_*.json'))[-1]) + ")\n")
out.append("| class | rows | fd32b6d finite | MF2 finite | MF2.1 finite | MF2.2 finite | MF2.2 max abs err vs fp64 | MF2.2 bytes == MF2 / fd32b6d |")
out.append("|---|---|---|---|---|---|---|---|")
cls = collections.OrderedDict()
for k in arms["mf2_2"]["n1"]:
    p = k.split("/"); c = "/".join(p[:3]) if p[0].startswith("sweep") else "witness/" + p[1]
    cls.setdefault(c, []).append(k)
for c, ks in cls.items():
    fin = {a: sum(arms[a]["n1"][k]["finite"] for k in ks) for a in arms}
    err = max(arms["mf2_2"]["n1"][k]["max_err"] for k in ks)
    eq = (sum(arms["mf2_2"]["n1"][k]["sha256"] == arms["mf2"]["n1"][k]["sha256"] for k in ks),
          sum(arms["mf2_2"]["n1"][k]["sha256"] == arms["fd32b6d"]["n1"][k]["sha256"] for k in ks))
    out.append(f"| {c} | {len(ks)} | {fin['fd32b6d']} | {fin['mf2']} | {fin['mf2_1']} | {fin['mf2_2']} | {err:.3g} | {eq[0]}/{len(ks)} / {eq[1]}/{len(ks)} |")
out.append("")
out.append("Identity census (finite non-overflow): " + "; ".join(f"MF2.2 vs {b}: {cmp_['identity_vs_'+b]['equal']}/{cmp_['identity_vs_'+b]['keys']} equal" for b in ("fd32b6d", "mf2", "mf2_1")))
out.append("Engines: " + "; ".join(f"{a} {r['provenance']['engine_sha256'][:12]}" for a, r in arms.items()))
obs = {a: arms[a]["observations"] for a in arms}
out.append("\nObservations (not gated): " + "; ".join(
    f"{k}: " + ", ".join(f"{a}={'finite '+str(obs[a][k]['values'][0]) if obs[a][k]['finite'] else 'NaN'}" for a in arms)
    for k in obs["mf2_2"] if isinstance(obs["mf2_2"][k], dict)))
out.append("bf16: " + "; ".join(f"{a}: {obs[a]['bf16_fused_sdpa']}" for a in arms))
# ---------------- N2
sw = {a: json.load(open(sorted(glob.glob(A + f"n2/sweep_{a}_*.json"))[-1])) for a in ("mf2_1", "mf2_2")}
out.append("\n## N2 (G3): isolated statistics, |bulk| row-max scale x 6 patterns x S {1,2,3,256,2048} x causal/non-causal\n")
out.append("| row max scale | cells | visible entries | MF2.1 out-of-band | MF2.1 constant rows all selected | MF2.2 mismatches (in band) | MF2.2 out-of-band | MF2.2 constant rows all selected |")
out.append("|---|---|---|---|---|---|---|---|")
for k in sw["mf2_2"]["isolated"]:
    b, a = sw["mf2_1"]["isolated"][k], sw["mf2_2"]["isolated"][k]
    cb = sw["mf2_1"]["constant_rows_all_selected"].get(k, "-"); ca = sw["mf2_2"]["constant_rows_all_selected"].get(k, "-")
    out.append(f"| {k} | {a['cells']} | {a['entries_visible']} | {b['out_of_band']} | {cb} | {a['mismatch']} ({a['in_band']}) | {a['out_of_band']} | {ca} |")
out.append("\nFull API apa_quant_attention (Q = mag*N(0,1), K,V ~ N(0,1), bits 4, p .15, scale 1, S {3,256,2048}, causal/non-causal):\n")
out.append("| dtype/mag | cells | MF2.1 out-of-band | MF2.2 mismatches (in band) | MF2.2 out-of-band | MF2.2 output finite |")
out.append("|---|---|---|---|---|---|")
for k in sw["mf2_2"]["full_api"]:
    b, a = sw["mf2_1"]["full_api"][k], sw["mf2_2"]["full_api"][k]
    out.append(f"| {k} | {a['cells']} | {b['out_of_band']} | {a['mismatch']} ({a['in_band']}) | {a['out_of_band']} | {a['output_finite']} |")
out.append("\nReview N2 witnesses (isolated, bulk = Q.K): " + "; ".join(
    f"{k}: MF2.1 {sw['mf2_1']['witness'][k]['selection']} oob {sw['mf2_1']['witness'][k]['out_of_band']} / MF2.2 {sw['mf2_2']['witness'][k]['selection']} oob {sw['mf2_2']['witness'][k]['out_of_band']}"
    for k in sw["mf2_2"]["witness"]))
out.append("Zero rows all-visible selected: MF2.1 " + str(all(sw['mf2_1']['zero_rows'].values())) + ", MF2.2 " + str(all(sw['mf2_2']['zero_rows'].values())))
for arm in ("mf2_1", "mf2_2"):
    t = collections.Counter(); g = 0.0
    for f in sorted(glob.glob(A + f"n2/real_*_{arm}_*.json")):
        for r in json.load(open(f))["rows"]:
            t["cells"] += 1
            for o in ("oracle_fp32bulk", "oracle_mf2_1_gemm"):
                for kk in ("mismatch", "in_band", "out_of_band", "invisible_selected"):
                    t[o + "." + kk] += r[o][kk]
            t["visible"] += r["oracle_fp32bulk"]["entries_visible"]; t["finite"] += r["output_finite"]
    out.append(f"Real captures {arm}: " + json.dumps(dict(t)))
# ---------------- adjudication
def last(rev, cell, var):
    fs = sorted(glob.glob(f"logs/apa_mf2_2_probe_{rev}_{cell}_{var}_*.log"))
    return fs[-1] if fs else None
def rows(f):
    r = {}
    if not f: return r
    for l in open(f, errors="replace"):
        try: d = json.loads(l)
        except Exception: continue
        if "name" in d: r[d["name"]] = d
    return r
lead = {}
for f in glob.glob("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf21rev/logs/apa_mf21rev_gpu_*_20261007T2*.log"):
    cell = os.path.basename(f).split("apa_mf21rev_gpu_")[1].rsplit("_20261007", 1)[0]
    lead[cell] = rows(f)
out.append("\n## G4 adjudication: every RED row of the lead's re-review logs, same probe (copy) per engine, one process each\n")
out.append("| cell | row(s) | lead MF2.1 | MF2.1 rerun | MF2 (048949d) | fd32b6d | MF2.2 | MF2.2 with real bf16 leaf |")
out.append("|---|---|---|---|---|---|---|---|")
def st(d, name):
    if name not in d: return "n/a"
    x = d[name]
    return "PASS" if x["passed"] else "RED: " + (x.get("error") or "")[:40].replace("\n", " ").replace("|", "/")
for cell in ("sentinel", "reductions", "training", "masks", "fused_overflow", "definition"):
    reds = [n for n, d in lead.get(cell, {}).items() if not d["passed"]]
    per = {rev: rows(last(rev, cell, "orig")) for rev in ("c5c468d", "048949d", "fd32b6d", "mf2_2")}
    leaf = rows(last("mf2_2", cell, "leaf"))
    for n in reds:
        out.append(f"| {cell} | {n} | {st(lead[cell], n)} | {st(per['c5c468d'], n)} | {st(per['048949d'], n)} | {st(per['fd32b6d'], n)} | {st(per['mf2_2'], n)} | {st(leaf, n) if leaf else '-'} |")
open(A + "tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
