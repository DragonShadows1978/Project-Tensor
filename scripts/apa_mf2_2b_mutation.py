"""APA-MF2.2b mutation run (registered B1-B7): scratch package copies under /tmp (the tree is never
edited); engine = artifacts/apa_mf2_2a/build (read-only, no build). Kill = >= 1 failure in
tensor_cuda/tests/test_apa_mf2_2b_*.py. Prior art: DeMillo, Lipton & Sayward (1978) -- unverified,
lead to check. Harness: mine."""
import glob, hashlib, json, os, shutil, subprocess, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "/tmp/claude-1000/apa_mf2_2b_mut"; E = os.path.join(W, "artifacts/apa_mf2_2a/build")
TESTS = sorted(glob.glob(os.path.join(W, "tensor_cuda/tests/test_apa_mf2_2b_*.py")))
FN, QN = "functional.py", "quant.py"
MUT = {
    "B1": (FN, """        weights = _guarded_softmax(scores, lambda s: tc.causal_softmax(
            s.reshape([B_, H_ * L, 1, S])).reshape([B_, H_, L, S]))""",
           """        weights = tc.causal_softmax(
            scores.reshape([B_, H_ * L, 1, S])).reshape([B_, H_, L, S])"""),
    "B2": (FN, "        weights = _guarded_softmax(scores, lambda s: s.softmax(-1))", "        weights = scores.softmax(-1)"),
    "B3": (FN, """        return tc.matmul(_guarded_softmax(scores, tc.causal_softmax,
                                          _causal_mask(L, S, query.device.split(":")[0], query.dtype)),
                         value)""", "        return tc.matmul(tc.causal_softmax(scores), value)"),
    "B4": (QN, "    weights = F._guarded_softmax(scores, lambda s: s.softmax(-1))", "    weights = scores.softmax(-1)"),
    "B5": (FN, "    return weights * (blocked * -1.0 + 1.0)", "    return weights"),
    "B6": (FN, "    return ind.min([-1], True)", "    return ind.max([-1], True)"),
    "B7": (FN, "        ind = tc._C.where(invisible, _guard_const(1.0, ind), ind)", "        ind = ind"),
}
if sys.argv[1] == "prep":
    man = {}
    for k, (f, old, new) in MUT.items():
        pkg = os.path.join(M, "pkg_" + k)
        shutil.copytree(os.path.join(W, "tensor_cuda/tensor_cuda"), os.path.join(pkg, "tensor_cuda"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        src = open(os.path.join(W, "tensor_cuda/tensor_cuda", f)).read()
        assert src.count(old) == 1, k
        mut = src.replace(old, new); open(os.path.join(pkg, "tensor_cuda", f), "w").write(mut)
        man[k] = {"file": f, "replace": [old, new], "sha256": hashlib.sha256(mut.encode()).hexdigest()}
    os.makedirs(os.path.join(W, "artifacts/apa_mf2_2b/mutation"), exist_ok=True)
    json.dump(man, open(os.path.join(W, "artifacts/apa_mf2_2b/mutation/mutants.json"), "x"), indent=1); print("prepared", list(man))
else:
    mid, out = sys.argv[2:4]; pkg = os.path.join(M, "pkg_" + mid)
    env = dict(os.environ, PYTHONPATH=f"{E}:{pkg}:{W}", PYTHONDONTWRITEBYTECODE="1")
    chk = subprocess.run([sys.executable, "-c", "import tensor_cuda.functional as F, tensor_cuda.quant as Q; print(F.__file__); print(Q.__file__)"],
                         env=env, capture_output=True, text=True, cwd=W)
    assert chk.stdout.split() and all(x.startswith(pkg) for x in chk.stdout.split()), chk.stdout + chk.stderr
    p = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-x"], env=env,
                       capture_output=True, text=True, cwd=W, timeout=560)
    last = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
    ff = [l for l in p.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")][:2]
    r = {"mutant": mid, "package": pkg, "pytest_rc": p.returncode, "summary": last[-1] if last else p.stdout[-300:],
         "first_failures": ff, "killed": p.returncode == 1, "error": p.returncode not in (0, 1)}
    json.dump(r, open(out, "x"), indent=1); print(json.dumps(r))
