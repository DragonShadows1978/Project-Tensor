#!/usr/bin/env python3
"""ENG1 G1: scope of the change against 1680764d. Read-only git. Writes source.diff, tracked_after.json, G1_scope.json.
Each diff hunk is mapped to the enclosing function of the candidate file (or to the E6 lines); every other tracked file
must be byte-identical (sha256 maps before/after). Prior art: unified diff hunks (GNU diffutils); SHA-256. Ours: the
hunk-to-function mapping.
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "artifacts/eng1"
BASE = "1680764d"
git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout
diff = git("diff", BASE, "--", ".", ":!artifacts/eng1", ":!orders", ":!logs")
(E / "source.diff").write_text(diff)
files = git("diff", "--name-only", BASE, "--", ".", ":!artifacts/eng1", ":!orders", ":!logs").split()
FUNCS = {
 "tensor_cuda/src/kernels.cu": {"NDArray::reshape": r"^NDArray NDArray::reshape\(", "apa_blend_softmax": r"^NDArray apa_blend_softmax\(",
                                "topk_nd": r"^std::tuple<NDArray, NDArray> topk_nd\(", "causal_softmax": r"^NDArray causal_softmax\(",
                                "rms_norm": r"^NDArray rms_norm\("},
 "tensor_cuda/src/ops.cpp": {"pow_scalar": r"^Tensor pow_scalar\("},
 "tensor_cuda/src/conv.cu": {"make_dims": r"^ConvDims make_dims\("},
}
def spans(path):
    lines = (ROOT / path).read_text().split("\n")
    out = {}
    for name, pat in FUNCS.get(path, {}).items():
        starts = [i for i, l in enumerate(lines) if re.match(pat, l)]
        assert len(starts) == 1, (path, name, starts)
        s = starts[0]; e = s
        while lines[e] != "}": e += 1
        out[name] = (s + 1, e + 1)
    return out
hunks = []
cur = None
for line in diff.splitlines():
    if line.startswith("+++ b/"):
        cur = line[6:]
    m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
    if m and cur:
        ns, nc = int(m.group(3)), int(m.group(4) or 1)
        hunks.append(dict(file=cur, new_start=ns, new_count=nc, old_start=int(m.group(1)), old_count=int(m.group(2) or 1)))
# -U0 hunks give exact changed ranges
d0 = git("diff", "-U0", BASE, "--", ".", ":!artifacts/eng1", ":!orders", ":!logs")
exact, cur = [], None
for line in d0.splitlines():
    if line.startswith("+++ b/"): cur = line[6:]
    m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
    if m and cur:
        ns, nc = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
        os_, oc = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        exact.append(dict(file=cur, new=(ns, ns + max(nc, 1) - 1), new_count=nc, old=(os_, os_ + max(oc, 1) - 1), old_count=oc))
violations = []
E6_DEFS = {"def test_terrain_render_%s(tmp_path):" % n for n in ("wo12g_cpu_repro_and_receipts", "wo9br2_ppm_ladder",
           "wo9e_ppm_receipts", "wo9ae_ppm_receipts", "wo8a_ppm_pair", "wo8c_ppm_triplet")}
terrain = (ROOT / "tensor_cuda/tests/test_terrain_render.py").read_text().split("\n")
for h in exact:
    f = h["file"]
    if f in FUNCS:
        sp = spans(f)
        owner = [n for n, (a, b) in sp.items() if a <= h["new"][0] and h["new"][1] <= b]
        h["function"] = owner[0] if owner else None
        if not owner: violations.append(h)
    elif f == "tensor_cuda/tests/test_terrain_render.py":
        changed = terrain[h["new"][0] - 1:h["new"][1]]
        ok = all(l in E6_DEFS or ("_write_ppm(tmp_path /" in l) or l.strip().startswith("tmp_path / f\"wo8c") for l in changed)
        h["function"] = "E6 def/write lines" if ok else None
        if not ok: violations.append(h)
    else:
        h["function"] = None; violations.append(h)
def shamap():
    m = {}
    for f in git("ls-files", "-z").split("\0"):
        if not f: continue
        p = ROOT / f
        m[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() and not p.is_symlink() else ("symlink" if p.is_symlink() else "MISSING")
    return m
after = shamap()
(E / "tracked_after.json").write_text(json.dumps(dict(head=git("rev-parse", "HEAD").strip(), count=len(after), sha256=after), indent=1, sort_keys=True) + "\n")
before = json.loads((E / "tracked_before.json").read_text())["sha256"]
changed_tracked = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
untracked = [l[3:] for l in git("status", "--porcelain", "-uall").splitlines() if l.startswith("?? ")]
# Authorized outputs (order: writable target): artifacts/eng1/, logs/, and the two build directories.
untracked_outside = [u for u in untracked if not u.startswith(("artifacts/eng1/", "logs/", "tensor_cuda/build-eng1/",
                                                              "tensor_cuda/build-eng1-base/"))]
allowed_changed = {"tensor_cuda/src/kernels.cu", "tensor_cuda/src/ops.cpp", "tensor_cuda/src/conv.cu", "tensor_cuda/tests/test_terrain_render.py"}
res = dict(diff_files=files, hunks=exact, violations=violations, changed_tracked=changed_tracked,
           tracked_count_before=len(before), tracked_count_after=len(after), untracked_outside_eng1=untracked_outside,
           passed=(not violations and set(changed_tracked) == allowed_changed and set(files) == allowed_changed
                   and untracked_outside == ["tensor_cuda/tests/test_eng1.py"]))
(E / "G1_scope.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(dict(passed=res["passed"], files=files, changed_tracked=changed_tracked, untracked_outside=untracked_outside,
                      hunks=[(h["file"].split("/")[-1], h["new"], h["function"]) for h in exact], violations=len(violations))))
sys.exit(0 if res["passed"] else 1)
