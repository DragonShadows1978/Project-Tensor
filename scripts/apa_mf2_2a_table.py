"""APA-MF2.2a: census table (path x special row x engine) + checks, from the census arms. Mine."""
import collections, json, sys
D = "artifacts/apa_mf2_2a/census/"; ST = "20261007T2327Z"
ARMS = ("fd32b6d", "mf2", "mf2_1", "mf2_2", "after")
A = {a: json.load(open(D + f"arm_{a}_{ST}.json")) for a in ARMS}
def status(c):
    return "err" if "error" in c else ("F" if all(c[n]["finite"] for n in c) else "NaN")
keys = list(A["fd32b6d"]["cells"])
# S=1 neg_inf_first / all_blocked are fully blocked rows: check after == fd32b6d bytes on every fused fully-blocked cell
fused = ("fused_sdpa", "int4_all", "int4_partial")
fb = [k for k in keys if k.split("/")[0] in fused and (k.endswith("/all_blocked") or (k.endswith("/neg_inf_first") and "/S1/" in k))
      and "error" not in A["after"]["cells"][k]]
eq = sum(A["after"]["cells"][k]["out"]["sha256"] == A["fd32b6d"]["cells"][k]["out"]["sha256"] for k in fb)
other = [k for k in keys if k not in fb and json.dumps(A["after"]["cells"][k], sort_keys=True) != json.dumps(A["mf2_2"]["cells"][k], sort_keys=True)]
out = [f"Fused fully-blocked cells (all_blocked + S=1 neg_inf_first): {len(fb)}; after bytes == fd32b6d: {eq}",
       f"after differs from MF2.2 on any other cell: {len(other)} {other[:5]}", "",
       "| path | special row | dtype | cells | fd32b6d | MF2 | MF2.1 | MF2.2 | after | value (fd32b6d -> after), first cell |",
       "|---|---|---|---|---|---|---|---|---|---|"]
grp = collections.OrderedDict()
for k in keys:
    p = k.split("/"); grp.setdefault((p[0], p[-1], p[1]), []).append(k)
def fmt(cell):
    if "error" in cell: return "error"
    v = cell["out"]["values"][0]; return "NaN" if v is None else f"{v:.4g}"
for (path, row, dt), ks in grp.items():
    cnt = {a: collections.Counter(status(A[a]["cells"][k]) for k in ks) for a in ARMS}
    s = lambda c: " ".join(f"{n}{c[n]}" for n in ("F", "NaN", "err") if c[n])  # noqa: E731
    out.append(f"| {path} | {row} | {dt} | {len(ks)} | {s(cnt['fd32b6d'])} | {s(cnt['mf2'])} | {s(cnt['mf2_1'])} | {s(cnt['mf2_2'])} | {s(cnt['after'])} | "
               f"{fmt(A['fd32b6d']['cells'][ks[0]])} -> {fmt(A['after']['cells'][ks[0]])} |")
open("artifacts/apa_mf2_2a/census_table.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
