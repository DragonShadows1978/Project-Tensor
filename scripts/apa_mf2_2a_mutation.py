"""APA-MF2.2a mutation run (registered A1-A6). Scratch copies only; kill = >= 1 failure in
tensor_cuda/tests/test_apa_mf2_2a_*.py + test_apa_mf2_2_n1.py. Prior art: DeMillo, Lipton & Sayward
(1978) mutation testing -- unverified, lead to check. Harness: mine (MF2.2 apa_mf2_2_mutation.py)."""
import glob, hashlib, json, os, shutil, subprocess, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "/tmp/claude-1000/apa_mf2_2a_mut"
TESTS = sorted(glob.glob(os.path.join(W, "tensor_cuda/tests/test_apa_mf2_2a_*.py"))) + [os.path.join(W, "tensor_cuda/tests/test_apa_mf2_2_n1.py")]
GREF = "  const float g_ref = gmax == -INFINITY ? 0.f : gmax;\n"
MREF = "    const float m_ref = m_new == -INFINITY ? 0.f : m_new;\n"
START = "  float m = -INFINITY;\n  float l = 0.f;\n"
MUT = {"A1": (0, GREF, "  const float g_ref = gmax;\n"), "A2": (1, GREF, "  const float g_ref = gmax;\n"),
       "A3": (0, GREF, "  const float g_ref = 0.f;\n"), "A4": (1, GREF, "  const float g_ref = 0.f;\n"),
       "A5": (0, MREF, "    const float m_ref = m_new;\n"), "A6": (0, START, "  float m = -3.0e38f;\n  float l = 0.f;\n")}


def sub(src, which, old, new):
    anchor = ["__global__ void fused_sdpa_noncausal_kernel(", "__global__ void apa_int4_sdpa_noncausal_kernel("][which]
    i = src.index(anchor); j = src.index(old, i); nxt = src.find("__global__", i + len(anchor))
    assert nxt == -1 or j < nxt
    return src[:j] + new + src[j + len(old):]


if sys.argv[1] == "prep":
    src = open(os.path.join(W, "tensor_cuda/src/kernels.cu")).read(); man = {}
    for k, (w, old, new) in MUT.items():
        root = os.path.join(M, "src_" + k, "tensor_cuda")
        shutil.copytree(os.path.join(W, "tensor_cuda/src"), os.path.join(root, "src"))
        shutil.copytree(os.path.join(W, "tensor_cuda/include"), os.path.join(root, "include"))
        mut = sub(src, w, old, new); open(os.path.join(root, "src/kernels.cu"), "w").write(mut)
        man[k] = {"kernel": w, "replace": [old, new], "kernels_sha256": hashlib.sha256(mut.encode()).hexdigest()}
    os.makedirs(os.path.join(W, "artifacts/apa_mf2_2a/mutation"), exist_ok=True)
    json.dump(man, open(os.path.join(W, "artifacts/apa_mf2_2a/mutation/mutants.json"), "x"), indent=1); print("prepared", list(man))
else:
    mid, engine, out = sys.argv[2:5]
    env = dict(os.environ, PYTHONPATH=f"{engine}:{W}/tensor_cuda:{W}", PYTHONDONTWRITEBYTECODE="1")
    chk = subprocess.run([sys.executable, "-c", "import tensor_cuda as t; print(t._C.__file__)"], env=env, capture_output=True, text=True, cwd=W)
    assert chk.stdout.strip().startswith(engine), chk.stdout + chk.stderr
    p = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-x"], env=env, capture_output=True, text=True, cwd=W, timeout=560)
    last = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
    ff = [l for l in p.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")][:2]
    r = {"mutant": mid, "engine": chk.stdout.strip(), "pytest_rc": p.returncode, "summary": last[-1] if last else p.stdout[-300:],
         "first_failures": ff, "killed": p.returncode == 1, "error": p.returncode not in (0, 1)}
    json.dump(r, open(out, "x"), indent=1); print(json.dumps(r))
