"""APA-MF1 summary: tables from the receipts (logs + junit + real/grm json). Read-only on receipts."""
from __future__ import annotations

import glob
import json
import os
import re
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts/apa_mf1")
LOG = os.path.join(ROOT, "logs")


def _progress(log):
    """Only the -s progress stream (before pytest's FAILURES section), so each case is read once."""
    return re.split(r"=+ FAILURES =+", open(log).read())[0]


def junit_order(xml, prefix):
    out = []
    for tc in ET.parse(xml).iter("testcase"):
        if tc.get("name").startswith(prefix):
            s = "passed"
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    s = ch.tag
            out.append((tc.get("name"), s))
    return out


QRE = re.compile(r"L=(\d+) S=(\d+) (\w+) rule=(\w): nontie_disagree=(\d+) ties=(\d+) agree=([\d.]+) "
                 r"eligible_frac=([\d.]+) rel=([\d.e+-]+) maxabs=([\d.e+-]+)")
TRE = re.compile(r"(sdpa|flash|apa_full|apa) L=(\d+) S=(\d+)(.*?) out:rel=([\d.e+-]+),maxabs=([\d.e+-]+) "
                 r"dq:rel=([\d.e+-]+),maxabs=[\d.e+-]+ dk:rel=([\d.e+-]+),maxabs=[\d.e+-]+ dv:rel=([\d.e+-]+),maxabs=[\d.e+-]+")


def quant_lines(log, xml):
    lines = [m for m in QRE.finditer(_progress(log))]
    names = junit_order(xml, "test_mf1_quant_causal_reference")
    assert len(lines) == len(names), (log, len(lines), len(names))
    rows = {}
    for m, (name, st) in zip(lines, names):
        L, S, dt = int(m.group(1)), int(m.group(2)), m.group(3)
        bits = int(name.rstrip("]").split("-")[-1])
        assert name == f"test_mf1_quant_causal_reference[{L}-{S}-{dt}-{bits}]", (name, L, S, dt)
        rows[(L, S, dt, bits)] = dict(status=st, disagree=int(m.group(5)), ties=int(m.group(6)), agree=float(m.group(7)),
                                      eligible=float(m.group(8)), rel=float(m.group(9)), maxabs=float(m.group(10)))
    return rows


def tgv2_lines(log):
    rows = {}
    for m in TRE.finditer(_progress(log)):
        key = (m.group(1), int(m.group(2)), int(m.group(3)), "tiled=True" in m.group(4))
        rows[key] = dict(out=float(m.group(5)), maxabs=float(m.group(6)), dq=float(m.group(7)),
                         dk=float(m.group(8)), dv=float(m.group(9)))
    return rows


def main():
    rec = os.path.join(ART, "receipts")
    q = {
        "before_ruleA": quant_lines(f"{LOG}/apa_mf1_before.log", f"{rec}/before_ruleA.xml"),
        "A_ruleA": quant_lines(f"{LOG}/apa_mf1_after_A.log", f"{rec}/after_A_ruleA.xml"),
        "before_ruleB": quant_lines(f"{LOG}/apa_mf1_before_ruleB.log", f"{rec}/before_ruleB.xml"),
        "A_ruleB": quant_lines(f"{LOG}/apa_mf1_A_ruleB_before_fixB.log", f"{rec}/A_ruleB.xml"),
        "AB_ruleB": quant_lines(f"{LOG}/apa_mf1_after_AB.log", f"{rec}/after_AB_ruleB.xml"),
    }
    t = {"before": tgv2_lines(f"{LOG}/apa_mf1_before.log"), "A": tgv2_lines(f"{LOG}/apa_mf1_after_A.log"),
         "AB": tgv2_lines(f"{LOG}/apa_mf1_after_AB.log")}
    print("## quant.py synthetic (fix A metric: rel err vs bottom-right rule-A reference; G3: agreement with fused rule)")
    print("| L | S | dtype | bits | before rel (A-ref) | A rel (A-ref) | before agree(fused) | A agree(fused) | A+B agree(fused) | A+B rel (B-ref) | A+B ties |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    g3_min = 1.0
    for key in sorted(q["AB_ruleB"]):
        L, S, dt, b = key
        r = [q[n][key] for n in ("before_ruleA", "A_ruleA", "before_ruleB", "A_ruleB", "AB_ruleB")]
        g3_min = min(g3_min, r[4]["agree"])
        print(f"| {L} | {S} | {dt} | {b} | {r[0]['rel']:.2e} | {r[1]['rel']:.2e} | {r[2]['agree']:.6f} | {r[3]['agree']:.6f} | "
              f"{r[4]['agree']:.6f} | {r[4]['rel']:.2e} | {r[4]['ties']} |")
    print(f"\nG3 synthetic min A+B agreement: {g3_min}")
    print("\n## tensor_gpu_v2 synthetic (max rel err over out/dq/dk/dv vs fp64 bottom-right reference)")
    print("| path | L | S | tiled | before max rel | A max rel | A+B max rel |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(t["AB"], key=lambda k: (k[0], k[3], k[1], k[2])):
        vals = []
        for n in ("before", "A", "AB"):
            d = t[n].get(key)
            vals.append(max(d["out"], d["dq"], d["dk"], d["dv"]) if d else float("nan"))
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {vals[0]:.2e} | {vals[1]:.2e} | {vals[2]:.2e} |")
    print("\n## real captures")
    print("| model | layer | L | bits | arm | eligible | refined | agree(fused) | out rel_F | out maxabs | excl/row = pred |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    real = []
    g3_real_min = 1.0
    for f in sorted(glob.glob(os.path.join(ART, "real/*.json"))):
        d = json.load(open(f))
        for c in d["cells"]:
            real.append(dict(model=d["model"], layer=d["layer"], **c))
            if c["arm"] == "AB":
                g3_real_min = min(g3_real_min, c["agree_fused"])
            print(f"| {d['model']} | {d['layer']} | {c['L']} | {c['bulk_bits']} | {c['arm']} | {c['eligible_frac']:.5f} | "
                  f"{c['refined_frac']:.5f} | {c['agree_fused']:.6f} | {c['out_rel_F']:.5f} | {c['out_maxabs']:.4f} | "
                  f"{c['excluded_per_row_matches_prediction']} |")
    print(f"\nG3 real min A+B agreement: {g3_real_min}")
    print("\n## kernel diag (one q head per KV head): agreement of arm mask vs kernel mask a / h; kernel vs fp64 rule")
    print("| model | layer | L | bits | kernel a vs rule | kernel h vs rule | before vs a | A vs a | A+B vs a | A+B vs h |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    kd_rows = []
    for f in sorted(glob.glob(os.path.join(ART, "real/*.json"))):
        d = json.load(open(f))
        for k in d.get("kernel_diag", []):
            kd_rows.append(dict(model=d["model"], layer=d["layer"], **k))
            av = k["arm_vs_kernel"]
            print(f"| {d['model']} | {d['layer']} | {k['L']} | {k['bulk_bits']} | {k['kernel_vs_rule_fp64']['a']:.7f} | "
                  f"{k['kernel_vs_rule_fp64']['h']:.7f} | {av['before']['a']:.6f} | {av['A']['a']:.6f} | {av['AB']['a']:.6f} | {av['AB']['h']:.6f} |")
    print("\n## GRM")
    grm = {}
    for f in sorted(glob.glob(os.path.join(ART, "grm/*.json"))):
        d = json.load(open(f))
        grm[os.path.basename(f)] = d
        print(f"### {os.path.basename(f)} prefill={d['prefill']} chunk={d['chunk']} before==A bytes: {d['before_vs_A_bytes_equal']} "
              f"standard repeat bytes equal: {d['standard_repeat_bytes_equal']}")
        print("| arm | KL mean (32 steps) | KL max | top-1 agree | true-token NLL | prefill-final KL | logits sha256 |")
        print("|---|---|---|---|---|---|---|")
        for arm, r in d["arms"].items():
            print(f"| {arm} | {r['kl_mean']:.6f} | {r['kl_max']:.6f} | {r['top1_agree']}/{r['top1_n']} | {r['nll_true_mean']:.4f} | "
                  f"{r['prefill_final']['kl_mean']:.3e} | {r['sha256_logits'][:16]} |")
    summary = dict(g3_synthetic_min_AB=g3_min, g3_real_min_AB=g3_real_min,
                   quant={n: {"|".join(map(str, k)): v for k, v in d.items()} for n, d in q.items()},
                   tgv2={n: {"|".join(map(str, k)): v for k, v in d.items()} for n, d in t.items()},
                   real=real, kernel_diag=kd_rows,
                   grm={k: {a: {kk: vv for kk, vv in r.items() if kk != "kl_per_step"} for a, r in v["arms"].items()}
                        for k, v in grm.items()})
    with open(os.path.join(ART, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
