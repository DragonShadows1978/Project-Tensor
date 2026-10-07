"""APA-MF2.1: build artifacts/apa_mf2_1/item{1..4}.patch from the before copies, the per-item
source snapshots (artifacts/apa_mf2_1/snap/itemK) and the final test files, then verify
that applying the four patches in order to the before state reproduces the tree
byte-for-byte (patch -p1 in a /tmp scratch). Prior art: none (mine; diff -u / patch).
"""
import filecmp, os, shutil, subprocess, sys
W = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2"
A = f"{W}/artifacts/apa_mf2_1"
SCR = "/tmp/claude-1000/apa_mf2_1_patches2"
SRC = ["tensor_cuda/tensor_cuda/functional.py", "tensor_cuda/tensor_cuda/quant.py", "apa_cuda/csrc/apa.cpp",
       "tensor_cuda/src/kernels.cu"]
TESTS = {1: ["tensor_cuda/tests/test_apa_mf2_1_short.py", "tensor_cuda/tests/test_apa_mf1_quant.py",
             "apa_cuda/tests/test_apa_mf2_1_extension.py", "apa_cuda/tests/test_apa_mf2_1_source.py"],
         2: ["tensor_cuda/tests/test_apa_mf2_1_sentinel.py"],
         3: ["tensor_cuda/tests/test_apa_mf2_1_definition.py", "tensor_cuda/tests/test_apa_mf2_rule.py",
             "tensor_cuda/tests/test_apa_mf2_mask.py", "scripts/apa_mf2_1_defn.py",
             "artifacts/apa_mf2_1/variants/quant_defn_additive_mask.py"],
         4: ["tensor_cuda/tests/test_apa_phase6.py", "tensor_cuda/tests/test_apa_mf2_1_gaps.py"]}
BEFORE_TESTS = {"tensor_cuda/tests/test_apa_mf1_quant.py", "tensor_cuda/tests/test_apa_mf2_rule.py",
                "tensor_cuda/tests/test_apa_mf2_mask.py", "tensor_cuda/tests/test_apa_phase6.py"}
flat = lambda p: p.replace("/", "__")
os.makedirs(SCR, exist_ok=True)


def state(k):
    d = f"{SCR}/state{k}"
    if os.path.exists(d):
        raise SystemExit(f"scratch {d} exists; use a fresh SCR")
    for p in SRC:
        src = f"{A}/before/files/{flat(p)}"
        for j in range(k, 0, -1):          # latest snapshot at or before item k (unchanged -> earlier)
            if os.path.exists(f"{A}/snap/item{j}/{flat(p)}"):
                src = f"{A}/snap/item{j}/{flat(p)}"
                break
        os.makedirs(os.path.dirname(f"{d}/{p}"), exist_ok=True); shutil.copyfile(src, f"{d}/{p}")
    for j, files in TESTS.items():
        for p in files:
            if j <= k:
                src = f"{W}/{p}"
            elif p in BEFORE_TESTS:
                src = f"{A}/before/files/{flat(p)}"
            else:
                continue
            os.makedirs(os.path.dirname(f"{d}/{p}"), exist_ok=True); shutil.copyfile(src, f"{d}/{p}")
    return d


states = [state(k) for k in range(5)]
for k in range(1, 5):
    for side, st in (("a", states[k - 1]), ("b", states[k])):
        if os.path.lexists(f"{SCR}/{side}"):
            os.rename(f"{SCR}/{side}", f"{SCR}/{side}_old{k}")
        shutil.copytree(st, f"{SCR}/{side}")
    r = subprocess.run(["diff", "-ruN", "a", "b"], cwd=SCR, capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    open(f"{A}/item{k}.patch", "x").write(r.stdout)
    print(f"item{k}.patch", len(r.stdout.splitlines()), "lines")
# verify: apply in order to state0 copy
v = f"{SCR}/verify"
shutil.copytree(states[0], v)
for k in range(1, 5):
    r = subprocess.run(["patch", "-p1", "-s", "-i", f"{A}/item{k}.patch"], cwd=v, capture_output=True, text=True)
    assert r.returncode == 0, (k, r.stdout, r.stderr)
ok = True
for p in SRC + [p for fs in TESTS.values() for p in fs]:
    same = filecmp.cmp(f"{v}/{p}", f"{W}/{p}", shallow=False)
    ok &= same
    print(("OK  " if same else "DIFF"), p)
print("PATCH CHAIN REPRODUCES TREE:", ok)
