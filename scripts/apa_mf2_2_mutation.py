"""APA-MF2.2 mutation run (registered mutants K1-K6, P1-P7; registration.json "mutants").

usage: apa_mf2_2_mutation.py prep            -> writes scratch mutant trees under /tmp/claude-1000/apa_mf2_2_mut
       apa_mf2_2_mutation.py run <id> <engine_dir> <pkg_dir> <out.json>
The tree's sources are NEVER edited: kernel mutants are scratch copies of tensor_cuda/{src,include}
built by scripts/apa_mf2_2_build.sh (APA_MF2_2_SRC_ROOT / APA_MF2_2_BUILD_DIR); Python mutants are
scratch copies of the package. Kill = the registered test set has >= 1 failure; a mutant whose
build fails is an ERROR (excluded from the denominator, reported).
Prior art: mutation testing, DeMillo, Lipton & Sayward (1978, "Hints on test data selection") --
unverified, lead to check. Harness: mine.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "/tmp/claude-1000/apa_mf2_2_mut"
TESTS = ["tensor_cuda/tests/test_apa_mf2_2_n1.py", "tensor_cuda/tests/test_apa_mf2_2_n2.py",
         "tensor_cuda/tests/test_apa_mf2_1_definition.py", "tensor_cuda/tests/test_apa_mf2_1_sentinel.py"]
GUARD = "    const float m_ref = m_new == -INFINITY ? 0.f : m_new;\n"
START = "  float m = -INFINITY;\n  float l = 0.f;\n"


def kernel_sub(src, which, old, new):
    """Replace the first `old` after the definition of kernel `which` (0 fused, 1 int4)."""
    anchor = ["__global__ void fused_sdpa_noncausal_kernel(", "__global__ void apa_int4_sdpa_noncausal_kernel("][which]
    i = src.index(anchor)
    j = src.index(old, i)
    nxt = src.find("__global__", i + len(anchor))
    assert nxt == -1 or j < nxt, "substitution would leave the kernel"
    return src[:j] + new + src[j + len(old):]


KERNEL = {
    "K1": (0, GUARD, "    const float m_ref = m_new;\n"),
    "K2": (1, GUARD, "    const float m_ref = m_new;\n"),
    "K3": (0, START, "  float m = -3.0e38f;\n  float l = 0.f;\n"),
    "K4": (1, START, "  float m = -3.0e38f;\n  float l = 0.f;\n"),
    "K5": (0, GUARD, "    const float m_ref = m == -INFINITY ? 0.f : m_new;\n"),
    "K6": (1, GUARD, "    const float m_ref = m == -INFINITY ? 0.f : m_new;\n"),
}
PY = {
    "P1": ("    u = absb / amax                          # u in [0, 1]\n", "    u = absb\n"),
    "P2": ("    amax = amax + (amax <= 0.0)              # all-zero row: divide by 1, never 0/0\n", "    amax = amax\n"),
    "P3": ("    sel = u.ge(thr)                          # scaled-domain comparison\n", "    sel = absb.ge(thr)\n"),
    "P4": ("    amax = absb.max([-1], True)              # visible row max (invisible entries are 0)\n",
           "    amax = absb.max([-2, -1], True)\n"),
    "P5": ("    var = (dev_ * dev_).sum([-1], True) / cnt\n", "    var = (u * u).sum([-1], True) / cnt - mean * mean\n"),
    "P6": ("        dev_ = dev_ * vis                    # invisible deviations -> 0 (0/1 product)\n", "        dev_ = dev_\n"),
    "P7": ("    sel = u.ge(thr)                          # scaled-domain comparison\n", "    sel = u.gt(thr)\n"),
}


def prep():
    os.makedirs(M, exist_ok=True)
    src = open(os.path.join(W, "tensor_cuda/src/kernels.cu")).read()
    man = {}
    for k, (which, old, new) in KERNEL.items():
        root = os.path.join(M, "src_" + k, "tensor_cuda")
        shutil.copytree(os.path.join(W, "tensor_cuda/src"), os.path.join(root, "src"))
        shutil.copytree(os.path.join(W, "tensor_cuda/include"), os.path.join(root, "include"))
        mut = kernel_sub(src, which, old, new)
        open(os.path.join(root, "src/kernels.cu"), "w").write(mut)
        man[k] = {"kernels_sha256": hashlib.sha256(mut.encode()).hexdigest(), "replace": [old, new], "kernel": which}
    q = open(os.path.join(W, "tensor_cuda/tensor_cuda/quant.py")).read()
    for k, (old, new) in PY.items():
        assert q.count(old) == 1, k
        pkg = os.path.join(M, "pkg_" + k)
        shutil.copytree(os.path.join(W, "tensor_cuda/tensor_cuda"), os.path.join(pkg, "tensor_cuda"))
        mut = q.replace(old, new)
        open(os.path.join(pkg, "tensor_cuda/quant.py"), "w").write(mut)
        man[k] = {"quant_sha256": hashlib.sha256(mut.encode()).hexdigest(), "replace": [old, new]}
    json.dump(man, open(os.path.join(W, "artifacts/apa_mf2_2/mutation/mutants.json"), "x"), indent=1)
    print("prepared", list(man))


def run(mid, engine, pkg, out):
    env = dict(os.environ, PYTHONPATH=f"{engine}:{pkg}:{W}", PYTHONDONTWRITEBYTECODE="1")
    chk = subprocess.run([sys.executable, "-c", "import tensor_cuda as t, tensor_cuda.quant as Q; print(t._C.__file__); print(Q.__file__)"],
                         env=env, capture_output=True, text=True, cwd=W)
    paths = chk.stdout.split()
    assert paths and paths[0].startswith(engine) and paths[1].startswith(pkg), chk.stdout + chk.stderr
    p = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-x"],
                       env=env, capture_output=True, text=True, cwd=W, timeout=560)
    last = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
    first_fail = [l for l in p.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")][:3]
    r = {"mutant": mid, "engine": paths[0], "quant": paths[1], "pytest_rc": p.returncode,
         "summary": last[-1] if last else p.stdout[-400:], "first_failures": first_fail,
         "killed": p.returncode == 1, "error": p.returncode not in (0, 1)}
    json.dump(r, open(out, "x"), indent=1)
    print(json.dumps(r))


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        run(*sys.argv[2:6])
