"""APA-MF2.2b: condensed census table, finite counts per path x special row x engine. Mine."""
import collections, json
A1, A2 = "artifacts/apa_mf2_2a/census/", "artifacts/apa_mf2_2b/census/"
ARM = {"fd32b6d": A1 + "arm_fd32b6d_20261007T2327Z.json", "MF2": A1 + "arm_mf2_20261007T2327Z.json",
       "MF2.1": A1 + "arm_mf2_1_20261007T2327Z.json", "MF2.2": A1 + "arm_mf2_2_20261007T2327Z.json",
       "after1": A1 + "arm_after_20261007T2327Z.json", "after2": A2 + "arm_after2_20261008T0000Z.json"}
C = {a: json.load(open(p))["cells"] for a, p in ARM.items()}
st = lambda c: "err" if "error" in c else ("F" if all(c[n]["finite"] for n in c) else "N")  # noqa: E731
def row(k):
    r = k.split("/")[-1]
    return "S=1 blocked" if (r == "neg_inf_first" and "/S1/" in k) else r
g = collections.OrderedDict()
for k in C["fd32b6d"]:
    g.setdefault((k.split("/")[0], row(k)), []).append(k)
out = ["| path | special row | cells | " + " | ".join(ARM) + " | fd32b6d finite & after2 NaN |", "|---|---|---|" + "---|" * (len(ARM) + 1)]
for (p, r), ks in g.items():
    ks = [k for k in ks if "error" not in C["fd32b6d"][k]]
    if not ks: continue
    f = [str(sum(st(C[a][k]) == "F" for k in ks)) for a in ARM]
    bad = sum(st(C["fd32b6d"][k]) == "F" and st(C["after2"][k]) == "N" for k in ks)
    out.append(f"| {p} | {r} | {len(ks)} | " + " | ".join(f) + f" | {bad} |")
open("artifacts/apa_mf2_2b/census_condensed.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
