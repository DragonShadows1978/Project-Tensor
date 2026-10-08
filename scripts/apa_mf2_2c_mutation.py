"""APA-MF2.2c mutation run (registered M1-M5): scratch package copies under /tmp (the tree is never
edited); engine artifacts/apa_mf2_2a/build (read-only, no build). Kill = >= 1 failure in
test_apa_mf2_2c_*.py + test_apa_mf2_2_n2.py + test_apa_mf2_1_definition.py.
Prior art: DeMillo, Lipton & Sayward (1978) -- unverified, lead to check. Harness: mine."""
import glob, hashlib, json, os, shutil, subprocess, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "/tmp/claude-1000/apa_mf2_2c_mut"; E = os.path.join(W, "artifacts/apa_mf2_2a/build")
TESTS = sorted(glob.glob(os.path.join(W, "tensor_cuda/tests/test_apa_mf2_2c_*.py"))) + \
    [os.path.join(W, "tensor_cuda/tests/test_apa_mf2_2_n2.py"), os.path.join(W, "tensor_cuda/tests/test_apa_mf2_1_definition.py")]
MEAN = "    mean = _row_sum(u) / cnt                 # APA-MF2.2 Amendment 3 (V7): bounded-depth sums\n"
VAR = "    var = _row_sum(dev_ * dev_) / cnt\n"
REM = "        total = total + x.slice(last, nb * _ROW_SUM_BLOCK, rem).sum([-1], True)\n"
MUT = {"M1": [(MEAN, "    mean = u.sum([-1], True) / cnt\n"), (VAR, "    var = (dev_ * dev_).sum([-1], True) / cnt\n")],
       "M2": [(REM, "        total = total\n")],
       "M3": [(MEAN, "    mean = u.sum([-1], True) / cnt\n")],
       "M4": [(VAR, "    var = (dev_ * dev_).sum([-1], True) / cnt\n")],
       "M5": [(VAR, "    var = _row_sum(u * u) / cnt - mean * mean\n")]}
if sys.argv[1] == "prep":
    man = {}
    src = open(os.path.join(W, "tensor_cuda/tensor_cuda/quant.py")).read()
    for k, subs in MUT.items():
        pkg = os.path.join(M, "pkg_" + k)
        shutil.copytree(os.path.join(W, "tensor_cuda/tensor_cuda"), os.path.join(pkg, "tensor_cuda"), ignore=shutil.ignore_patterns("__pycache__"))
        mut = src
        for old, new in subs:
            assert mut.count(old) == 1, (k, old); mut = mut.replace(old, new)
        open(os.path.join(pkg, "tensor_cuda/quant.py"), "w").write(mut)
        man[k] = {"subs": subs, "sha256": hashlib.sha256(mut.encode()).hexdigest()}
    os.makedirs(os.path.join(W, "artifacts/apa_mf2_2c/mutation"), exist_ok=True)
    json.dump(man, open(os.path.join(W, "artifacts/apa_mf2_2c/mutation/mutants.json"), "x"), indent=1); print("prepared", list(man))
else:
    mid, out = sys.argv[2:4]; pkg = os.path.join(M, "pkg_" + mid)
    env = dict(os.environ, PYTHONPATH=f"{E}:{pkg}:{W}", PYTHONDONTWRITEBYTECODE="1")
    chk = subprocess.run([sys.executable, "-c", "import tensor_cuda.quant as Q; print(Q.__file__)"], env=env, capture_output=True, text=True, cwd=W)
    assert chk.stdout.strip().startswith(pkg), chk.stdout + chk.stderr
    p = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-x"], env=env, capture_output=True, text=True, cwd=W, timeout=560)
    last = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
    ff = [l for l in p.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")][:2]
    r = {"mutant": mid, "package": pkg, "pytest_rc": p.returncode, "summary": last[-1] if last else p.stdout[-300:],
         "first_failures": ff, "killed": p.returncode == 1, "error": p.returncode not in (0, 1)}
    json.dump(r, open(out, "x"), indent=1); print(json.dumps(r))
