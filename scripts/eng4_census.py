"""ENG4 M1 census -> artifacts/eng4/CENSUS.json.

Roles (L90): product (in-repo C++ and Python callers of the forward tc::matmul), test (existing tests), consumer
(Project-Voice/engine and GRAPA-Native-LLM, read-only grep), harness (this round's own scripts: n/a).

- Product and N1 rows are hand-written from reading the code (the M1 grep of L82 plus `rg` over src/ and the package);
  each row's file:line is re-checked here: the line must contain the row's snippet, else the script stops.
- Test rows: every direct call of tc.matmul / _C.matmul / Tensor @ in tensor_cuda/tests and tests/ (grep), with the
  rank combination where the code fixes it (hand annotations below), else "dynamic"; indirect test paths reach the
  forward through product entries (nn.Linear, Conv2D, ConvTranspose2D, SDPA, apa_quant_attention, int4/intn linear)
  and carry those entries' classes.
- Consumer rows: every `matmul(` and ` @ ` line in the two roots (rg, *.py), classified mechanically:
  "not TensorCUDA" when the file does not import tensor_cuda or the call is np./numpy./torch.; else the hand
  annotation for that site when one exists, else "dynamic". Copies under artifacts/ are counted and listed by file.

Classes (order M2/M3): a = B rank 2; b = equal ranks with equal batch shapes; c = unequal ranks, B rank > 2, all
leading B extents 1. Refused = anything else.

Usage: python3 -B -P scripts/eng4_census.py
Prior art: no prior art known to me beyond grep/regex call-site inventories (ENG2/ENG3 census scripts in this repo).
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE = "/mnt/ForgeRealm/Project-Voice/engine"
GRAPA = "/mnt/ForgeRealm/GRAPA-Native-LLM"

# (file, line, snippet, ranks / class, note)
PRODUCT = [
    ("tensor_cuda/src/bindings.cpp", 2075, 'm.def("matmul", &ops::matmul', "dynamic (caller's operands)",
     "Python _C.matmul -> ops::matmul (ops.cpp) -> tc::matmul"),
    ("tensor_cuda/src/bindings.cpp", 2050, '"__matmul__"', "dynamic (caller's operands)",
     "Tensor @ Tensor -> ops::matmul(a, b) -> tc::matmul"),
    ("tensor_cuda/src/ops.cpp", 826, "NDArray out = tc::matmul(a.data(), b.data(), alpha, trans_b);",
     "dynamic", "ops::matmul (autograd op): the forward call every Python path takes"),
    ("tensor_cuda/src/ops.cpp", 785, "if (a.dtype != DType::Float32) return tc::matmul(a, b, alpha, trans_b);",
     "a or b (backward operands: g with A's batch shape and B or its rank-2 view; A^T/g pairs share A's batch)",
     "matmul_backward's lower-precision path; operands built by the Tensor matmul closure"),
    ("tensor_cuda/src/ops.cpp", 864, "x.v->accumulate_grad(tc::matmul(g, w_kn, 1.f, /*trans_b=*/true));",
     "class a (w_kn rank 2, (K,N))", "int4 linear backward"),
    ("tensor_cuda/src/ops.cpp", 903, "x.v->accumulate_grad(tc::matmul(g, w_kn, 1.f, /*trans_b=*/true));",
     "class a (w_kn rank 2)", "intn linear backward"),
    ("tensor_cuda/src/kernels.cu", 4336, "return matmul(x, w_kn);", "class a (w_kn rank 2); x rank >= 1 (rank 1 "
     "refused by the pre-existing rank check)", "int4_linear"),
    ("tensor_cuda/src/kernels.cu", 4818, "return matmul(x, w_kn);", "class a (w_kn rank 2)", "intn_linear"),
    ("tensor_cuda/src/gemm_apa.cu", 779, "NDArray outg = matmul(pg, v, 1.f, false);",
     "class b: pg [B,KVH,M,S] and v [B,KVH,S,VD], both rank 4; validate() (gemm_apa.cu:75-98) fixes v's B and KVH",
     "APA SB2 selective attention P@V"),
    ("tensor_cuda/tensor_cuda/__init__.py", 343, "return _C.matmul(a, b, alpha, trans_b)", "dynamic",
     "public tc.matmul"),
    ("tensor_cuda/tensor_cuda/nn.py", 107, "out = tc.matmul(x, w, trans_b=True)",
     "class a (weight rank 2: (out, in)); x any rank >= 2", "nn.Linear"),
    ("tensor_cuda/tensor_cuda/nn.py", 386, "out = tc.matmul(cols.transpose(1, 2), W2.transpose(0, 1))",
     "class a (rank 3 cols x rank 2 W2)", "nn.Conv2D (im2col)"),
    ("tensor_cuda/tensor_cuda/nn.py", 448, "oc = tc.matmul(x2, Wt).transpose(1, 2)", "class a (rank 3 x rank 2)",
     "nn.ConvTranspose2D"),
    ("tensor_cuda/tensor_cuda/functional.py", 171, "scores = tc.matmul(query, key, alpha=scale, trans_b=True)",
     "dynamic; documented (B,H,L,D) for q/k/v -> class b when the caller follows it (no GQA: callers repeat KV)",
     "scaled_dot_product_attention QK^T"),
    ("tensor_cuda/tensor_cuda/functional.py", 177, "return tc.matmul(_guarded_softmax(scores, tc.causal_softmax,",
     "class b if q/k/v share (B,H) (scores carry q's batch)", "SDPA fused-softmax path P@V"),
    ("tensor_cuda/tensor_cuda/functional.py", 200, "return tc.matmul(weights, value)",
     "class b if q/k/v share (B,H)", "SDPA P@V"),
    ("tensor_cuda/tensor_cuda/quant.py", 125, "rotated = tc.matmul(unit, Rb.transpose(-2, -1))",
     "class b: unit (B,H,S,D) unpacked rank 4; Rb expanded to [B,H,D,D]", "_quantize_keys rotate"),
    ("tensor_cuda/tensor_cuda/quant.py", 127, "recon = tc.matmul(centroids, Rb) * norms",
     "class b ([B,H,S,D] x [B,H,D,D])", "_quantize_keys rotate back"),
    ("tensor_cuda/tensor_cuda/quant.py", 248, "bulk32 = tc.matmul(qf, kqf, alpha=scale, trans_b=True)",
     "rank 4 x rank 4 (qf unpacked B,H,L,_); class b if the caller's key shares (B,H)", "_definition_selection"),
    ("tensor_cuda/tensor_cuda/quant.py", 291, "bulk = tc.matmul(query, key_quant, alpha=scale, trans_b=True)",
     "rank 4 x rank 4 (query unpacked; key_quant from _quantize_keys of key); class b if key shares (B,H)",
     "apa_quant_attention bulk"),
    ("tensor_cuda/tensor_cuda/quant.py", 292, "ranking = tc.matmul(query, key, alpha=scale, trans_b=True)",
     "rank 4 x dynamic; class b if key shares (B,H) (documented (B,H,L,D))", "apa_quant_attention ranking"),
    ("tensor_cuda/tensor_cuda/quant.py", 331, "return tc.matmul(weights, value)", "class b if value shares (B,H)",
     "apa_quant_attention P@V"),
    ("tensor_cuda/tensor_cuda/apa_packed.py", 389, "return (tc.matmul(cent, Rb) * norm).detach()",
     "class b: cent (B,KV,S,D) built here; Rb expanded to [B,KV,D,D]", "PackedSpKeys._dense_reconstruction"),
]

# N1: callers of numpy_dtype / tensor_from_numpy (C++ name), and whether the Python side casts first.
N1 = [
    ("tensor_cuda/src/bindings.cpp", 88, "DType dt = numpy_dtype(arr);", "the only caller of numpy_dtype",
     "tensor_from_numpy"),
    ("tensor_cuda/src/bindings.cpp", 2071, 'm.def("tensor", &tensor_from_numpy', "the only binding of "
     "tensor_from_numpy: Python name _C.tensor (there is no _C.tensor_from_numpy)", "binding"),
    ("tensor_cuda/tensor_cuda/__init__.py", 63, "result = _C.tensor(arr, device, False).astype(dtype).detach()",
     "casts first: arr is np.float32 (bfloat16 device-cast path)", "_tensor_from_device_cast"),
    ("tensor_cuda/tensor_cuda/__init__.py", 75, "return _C.tensor(arr, device, requires_grad)",
     "casts first: np.asarray(data, dtype=_np_dtype(dtype)), a listed dtype (ENG2 F1 refuses others)", "tc.tensor"),
    ("tensor_cuda/tensor_cuda/__init__.py", 90, "return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)",
     "casts first: astype(_np_dtype(dtype))", "zeros/ones/randn/rand factories"),
    ("tensor_cuda/tensor_cuda/__init__.py", 101, "return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)",
     "NO cast: the caller's dtype goes straight to numpy_dtype (unlisted dtypes: float32 garbage before N1, "
     "TypeError after)", "tc.from_numpy"),
    ("tensor_cuda/tensor_cuda/__init__.py", 1394,
     'idx = _C.tensor(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)), "cuda", False)', "casts first: int64",
     "embedding"),
    ("tensor_cuda/tensor_cuda/__init__.py", 1424, "return _C.cross_entropy(logits, _C.tensor(onehot, device, False))",
     "casts first: onehot built as np.float32", "cross_entropy"),
    ("tensor_cuda/tests/test_bwd_r1_core.py", 48, "xs = tuple(tc.from_numpy(a, requires_grad=True) for a in "
     "case.operands)", "test; operands are float32 (oracle f32() / np.float32 arrays)", "test"),
    ("tensor_cuda/tests/test_bwd_r1_core.py", 51, "(forward * tc.from_numpy(case.seed)).sum().backward()",
     "test; seed float32 (seed_with_zeros of f32 data)", "test"),
    ("tensor_cuda/tests/test_eng1.py", 80, 'return tc.from_numpy(np.zeros(shape, dtype=dtype), device="cpu")',
     "test; dtype parameter (default float32)", "test"),
    ("tensor_cuda/tests/test_eng1.py", 622, "cpu = lambda s, d=np.float32: tc.from_numpy(np.zeros(s, dtype=d), "
     "device='cpu')", "test (subprocess source text); default float32", "test"),
]

# Hand annotations for consumer sites that import tensor_cuda (rel path, line) -> ranks/class.
CONSUMER_NOTES = {
    ("Project-Voice/engine/kokoro_tc/model.py", 725): "class a if W is rank 2 (linear weight via self.f32(W))",
    ("Project-Voice/engine/kokoro_tc/model.py", 728): "class a if W is rank 2 (trans_b linear)",
    ("Project-Voice/engine/kokoro_tc/model.py", 810): "class b: q, k [1, 12, T, 64] per the inline comment [1,12,T,T]",
    ("Project-Voice/engine/kokoro_tc/model.py", 812): "class b: p [1,12,T,T] x v [1,12,T,64]",
    ("Project-Voice/engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py", 95):
        "class a (rank 3 cols x rank 2 W2), the nn.Conv2D form",
    ("Project-Voice/engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py", 122):
        "class a (rank 3 x rank 2), the ConvTranspose form",
    ("GRAPA-Native-LLM/grapa/model.py", 91): "class a: emb.weight rank 2 (vocab, d)",
    ("GRAPA-Native-LLM/grapa/model_mla.py", 109): "class a: emb weight rank 2",
    ("GRAPA-Native-LLM/grapa/fwd_precision.py", 470): "class a: emb weight rank 2",
    ("GRAPA-Native-LLM/grapa/codebooks.py", 79): "class c: u (B,1,S,D) rank 4 x R (1,D,D) rank 3, B batch [1]",
    ("GRAPA-Native-LLM/grapa/codebooks.py", 85): "class c: q (B,1,S,D) rank 4 x R (1,D,D) rank 3",
    ("GRAPA-Native-LLM/grapa/attention.py", 81): "class b: q (B,H,L,D) x k after _repeat_kv -> (B,H,S,D)",
    ("GRAPA-Native-LLM/grapa/attention.py", 89): "class b: q x kq after _repeat_kv",
    ("GRAPA-Native-LLM/grapa/attention.py", 115): "class b: weights (B,H,L,S) x v after _repeat_kv",
}

# Direct test call sites: (rel path, line) -> ranks/class where the code fixes them.
TEST_NOTES = {
    ("tensor_cuda/tests/test_bwd_r1_matmul.py", 147): "rank 2 x rank 2 (class b, empty batch)",
    ("tensor_cuda/tests/test_bwd_r1_matmul.py", 164): "REFUSED by M3 (M7 test, changed): A (2,3)|(2,1,2,3)|"
                                                       "(2,2,3) x B (2,3,4)|(2,3,4)|(1,2,3,4)",
    ("tensor_cuda/tests/test_bwd_r1_matmul.py", 179): "class c: A (2,2,3) x B (1,1,3,4)|(1,1,4,3)",
    ("tensor_cuda/tests/test_bwd_r1_core.py", 24): "via oracle Cases: small() plain/trans rank 2 x 2; batch_shared "
                                                    "(2,2,3,5) x (5,4) class a; batched (2,2,..) x (2,2,..) class b; "
                                                    "kokoro rank 2 x 2; strided canary (2,128,128) x (2,128,128) "
                                                    "class b; TF32 canaries rank 2",
    ("tensor_cuda/tests/test_engine.py", 36): "rank 2 x rank 2", ("tensor_cuda/tests/test_engine.py", 43): "rank 2 x 2",
}


def line_of(path, n):
    with open(path, errors="replace") as f:
        for i, l in enumerate(f, 1):
            if i == n:
                return l.rstrip("\n")
    return None


B_TREE = "/tmp/eng4_src_B"  # git archive of HEAD (f7282b1d + the order commit): Build B's sources


def e_lines(path, snip):
    with open(path, errors="replace") as f:
        return [i for i, l in enumerate(f, 1) if snip in l]


def checked(rows):
    """Engine rows are checked at Build B's line numbers (the order cites production lines); tests at the worktree.
    Each row also records where the same code sits in Build E's file."""
    out = []
    for f, n, snip, ranks, note in rows:
        base = ROOT if f.startswith(("tensor_cuda/tests/", "tests/")) else B_TREE
        got = line_of(os.path.join(base, f), n)
        if got is None or snip not in got:
            raise SystemExit(f"census row does not match its line: {f}:{n}: {got!r} lacks {snip!r}")
        out.append({"site_build_B": f"{f}:{n}", "lines_in_build_E": e_lines(os.path.join(ROOT, f), snip),
                    "code": got.strip(), "ranks_class": ranks, "note": note})
    return out


def rg(args):
    r = subprocess.run(["rg", "-n", "--no-heading"] + args, capture_output=True, text=True, timeout=300)
    if r.returncode not in (0, 1):
        raise SystemExit(f"rg failed rc={r.returncode}: {r.stderr[:500]}")
    return r.stdout.splitlines()


def imports_tc(path, cache={}):
    if path not in cache:
        with open(path, errors="replace") as f:
            cache[path] = bool(re.search(r"import tensor_cuda|from tensor_cuda|tensor_cuda as", f.read()))
    return cache[path]


def consumer_rows(root):
    label = os.path.relpath(root, "/mnt/ForgeRealm")
    rows, copies = [], {}
    for l in rg(["-g", "*.py", "-e", r"matmul\(", "-e", " @ ", root]):
        path, n, code = l.split(":", 2)
        rel = os.path.relpath(path, "/mnt/ForgeRealm")
        if "/artifacts/" in path:
            copies[rel] = copies.get(rel, 0) + 1
            continue
        if not imports_tc(path):
            cls = "not TensorCUDA (file does not import tensor_cuda)"
        elif re.search(r"\b(np|numpy|torch)\.matmul\(", code):
            cls = "not TensorCUDA (NumPy/PyTorch call)"
        elif (rel, int(n)) in CONSUMER_NOTES:
            cls = CONSUMER_NOTES[(rel, int(n))]
        elif re.search(r"matmul\(", code) or " @ " in code:
            cls = "dynamic (file imports tensor_cuda; operands not fixed at the site, or a NumPy @ in a mixed file)"
        rows.append({"site": f"{rel}:{n}", "code": code.strip()[:200], "ranks_class": cls})
    return label, rows, copies


def main():
    product = checked(PRODUCT)
    n1 = checked(N1)
    tests = []
    for l in rg(["-g", "*.py", "-e", r"\b(tc|tensor_cuda|_C)\.matmul\(", "-e", "__matmul__",
                 os.path.join(ROOT, "tensor_cuda/tests"), os.path.join(ROOT, "tests")]):
        path, n, code = l.split(":", 2)
        rel = os.path.relpath(path, ROOT)
        tests.append({"site": f"{rel}:{n}", "code": code.strip()[:200],
                      "ranks_class": TEST_NOTES.get((rel, int(n)), "dynamic or rank 2 (no batch dimension fixed at "
                                                                  "the site; reaches the forward as a product call)")})
    consumers = {}
    copies = {}
    for root in (VOICE, GRAPA):
        label, rows, cp = consumer_rows(root)
        consumers[label] = rows
        copies[label] = cp
    tc_sites = {k: [r for r in v if not r["ranks_class"].startswith("not TensorCUDA")] for k, v in consumers.items()}
    refused_product = [r for r in product if r["ranks_class"].startswith("REFUSED")]
    rec = {
        "order": "orders/ENG4_MATMUL_CONTRACT.md M1, N1, L90",
        "classes": {"a": "B rank 2 (B stride 0; rank 2 x rank 2 keeps the old stride K*N with batch 1)",
                    "b": "equal ranks, equal batch shapes (B stride K*N)",
                    "c": "unequal ranks, B rank > 2, every leading B extent 1 (B stride 0)",
                    "refused": "every other pair (M3), different devices, M/N/K or batch above INT_MAX (M4)"},
        "role_counts": {"product": len(product), "test_direct": len(tests),
                        "consumer_sites": {k: len(v) for k, v in consumers.items()},
                        "consumer_tensorcuda_sites": {k: len(v) for k, v in tc_sites.items()},
                        "consumer_artifact_copy_sites": {k: sum(v.values()) for k, v in copies.items()},
                        "harness": "n/a"},
        "product": product,
        "product_refused": refused_product,
        "product_verdict": ("no product call path fixes a refused combination; the dynamic entries (tc.matmul, "
                            "Tensor @, ops::matmul) accept what their callers pass, and every caller that fixes "
                            "ranks passes class a or b"),
        "tests_direct": tests,
        "tests_with_batch_dimensions": [t for t in tests if t["ranks_class"].startswith(("class", "REFUSED", "via"))],
        "tests_indirect": "tests of nn.Linear/Conv2D/ConvTranspose2D (class a), SDPA, apa_quant_attention, "
                          "PackedSpKeys, gemm_apa selective attention (class b with documented equal (B,H)), "
                          "int4/intn linear (class a); per-ID outcomes are G4's evidence",
        "consumers": consumers,
        "consumer_tensorcuda_sites": tc_sites,
        "consumer_artifact_copies_by_file": copies,
        "n1_callers": n1,
        "n1_consumer_callers": "rg for tc.from_numpy / tensor_cuda.from_numpy / _C.tensor in both consumer roots: "
                               "no site",
    }
    out = os.path.join(ROOT, "artifacts/eng4/CENSUS.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["role_counts"], indent=1))
    print("product refused:", len(refused_product))


if __name__ == "__main__":
    main()
