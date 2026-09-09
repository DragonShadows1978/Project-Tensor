"""APA-SPD1 shared contracts; imports and dry-run never initialize CUDA.

Prior art: this harness reuses Project-Tensor E1 and SP1/SP1.1 (David and
implementation seats, 2026): grouped dense composition, frozen delta lookup,
CUDA-event timing and leased one-cell jobs. Statistical summaries use NumPy's
linear quantiles (Hyndman & Fan, 1996, type 7); SHA-256 receipts use FIPS 180-4
(NIST, 2015). Our contribution is measurement/receipt plumbing, no new attention
algorithm. See artifacts/apa_spd1/PRIOR_ART.md for contender citations.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/apa_spd1"
SCOPE = "kernel sweep; this establishes nothing about model quality"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before opening: a nonfinite metric must not destroy a receipt.
    body = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x" if exclusive else "w") as out:
        out.write(body)


def registration(verify_sources=True):
    path = ART / "registration.json"
    if sha(path) != (ART / "registration.sha256").read_text().split()[0]:
        raise RuntimeError("registration hash mismatch")
    reg = json.loads(path.read_text())
    if verify_sources:
        for name, expected in {**reg['source_pins'], **reg['inherited_pins']}.items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f"registered source/receipt drift: {name}")
    return reg


def registry():
    # Vaswani et al. (2017) dense SDPA; E1 (2026) grouped engine entry points.
    rows = []
    for dtype in ("float32", "bfloat16"):
        suffix = "fp32" if dtype == "float32" else "bf16"
        rows.append(dict(id=f"engine_dense_{suffix}", dtype=dtype, owner="engine",
                         entry="tensor_cuda.matmul -> causal_softmax / Tensor.softmax -> matmul",
                         required=True))
        # PyTorch contributors (2023-2026), SDPA backend selection. Each backend
        # is forced alone; there is no successful fallback row under another name.
        rows.append(dict(id=f"torch_math_{suffix}", dtype=dtype, owner="torch",
                         entry="torch.nn.functional.scaled_dot_product_attention; sdpa_kernel(MATH)",
                         backend="MATH", required=True))
        # APA two-pass: existing z-score of |bulk| selector (David, 2026).
        rows.append(dict(id=f"apa_two_pass_{suffix}", dtype=dtype, owner="engine",
                         entry="tensor_cuda.apa_selective_attention; apa_selective_kernel family",
                         required=True))
        # BLASST (Yuan et al., 2025/2026) running-max comparator; APA SP1 (2026)
        # applies it per key to precision, retaining every key. SP1.1 uses
        # partition-local max + online-softmax merge for decode, not global SP.
        rows.append(dict(id=f"apa_sp_{suffix}", dtype=dtype, owner="engine",
                         entry="tensor_cuda._C.apa_selective_attention_sp; prefill / split-K decode; TC_APA_SP=1",
                         required=True))
    for backend in ("EFFICIENT_ATTENTION", "FLASH_ATTENTION"):
        name = "efficient" if backend.startswith("EFFICIENT") else "flash"
        rows.append(dict(id=f"torch_{name}_bf16", dtype="bfloat16", owner="torch",
                         entry=f"torch.nn.functional.scaled_dot_product_attention; sdpa_kernel({backend})",
                         backend=backend, required=False))
    rows.append(dict(id="flash_attn_bf16", dtype="bfloat16", owner="torch",
                     entry="flash_attn.flash_attn_func (optional, version recorded)", required=False))
    rows.append(dict(id="apa_sp2_fp32", dtype="float32", owner="engine",
                     entry="actual SP2 launcher through hashed adapter; epsilon=1e-3 (optional)", required=False))
    return rows


def load_runtime():
    build = ROOT / "artifacts/apa_sp1/build"
    manifest = json.loads((build / "manifest.json").read_text())
    for p, expected in manifest['sources'].items():
        if sha(ROOT / p) != expected:
            raise RuntimeError(f"inherited runtime source mismatch: {p}")
    for p, expected in manifest['modules'].items():
        if sha(build / p) != expected:
            raise RuntimeError(f"inherited runtime binary mismatch: {p}")
    sys.path[:0] = [str(build), str(ROOT / "tensor_cuda")]
    import tensor_cuda as tc
    if Path(tc.__file__).resolve().parent != ROOT / "tensor_cuda/tensor_cuda":
        raise RuntimeError("wrong tensor_cuda Python module")
    if Path(tc._C.__file__).resolve().parent != build:
        raise RuntimeError("wrong tensor_cuda binary; no installed/main fallback")
    return tc


def frozen_sp2():
    base = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2')
    # Only the specifically authorized read-only discovery path in main.
    return sorted(str(p) for p in base.rglob('*') if p.is_file() and 'frozen' in p.name.lower())


def fingerprint():
    paths = [ART / 'registration.json', ART / 'registration.sha256']
    paths += sorted((ROOT / 'scripts').glob('apa_spd1_*'))
    paths += sorted((ROOT / 'tensor_cuda/tests').glob('test_apa_spd1*'))
    adapter = ART / 'sp2_adapter.json'
    if adapter.exists():
        paths.append(adapter)
    result = {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}
    reg = registration(verify_sources=False)
    for name in {**reg['source_pins'], **reg['inherited_pins']}:
        result[name] = sha(ROOT / name)
    # An optional SP2 table appearing later must not silently mix old/new rows.
    result.update({p: sha(p) for p in frozen_sp2()})
    return result


def inventory():
    import torch
    tc = load_runtime()  # Import only: engine "CPU tensors" still require CUDA.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return dict(evidence_class='CPU import/source inspection; GPU availability untested per backend',
                torch_version=torch.__version__, torch_cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), runtime=str(tc._C.__file__),
                runtime_sha256=sha(tc._C.__file__),
                engine_entries={n: hasattr(tc._C, n) for n in
                                ['matmul', 'causal_softmax', 'apa_selective_attention',
                                 'apa_selective_attention_sp', 'set_alloc_pooling']},
                sdpa_enums={n: hasattr(SDPBackend, n) for n in
                            ['MATH', 'EFFICIENT_ATTENTION', 'FLASH_ATTENTION']},
                flash_attn_importable=importlib.util.find_spec('flash_attn') is not None,
                sp2_frozen_files=frozen_sp2(), contenders=registry(),
                note='sm_89 backend support requires a forced call on each real shape/dtype; CPU enum presence is not support')


def inputs(shape, index, seed=20260906):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    B, H, K, L, S, D = (shape[x] for x in ['B', 'H', 'KVH', 'L', 'S', 'D'])
    return tuple(rng.standard_normal(s, dtype=np.float32) for s in
                 [(B, H, L, D), (B, K, S, D), (B, K, S, D)])


def rounded_host(arr, dtype):
    # PyTorch CPU dtype conversion is the common rounding authority. Reuse
    # BF16-expanded fp32 values on both engines to prevent double rounding.
    import torch
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError(f'unregistered dtype {dtype}')
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(getattr(torch, dtype))
    return t.float().numpy().copy()


def valid_mask(L, S, causal):
    # Existing engine/E1 bottom-right causal convention, also PyTorch
    # causal_lower_right (2024-2026); rectangular decode sees the entire cache.
    if not 0 < L <= S:
        raise ValueError('require 0 < L <= S')
    return (np.arange(S)[None, :] <= S - L + np.arange(L)[:, None]
            if causal else np.ones((L, S), bool))


def grouped_dense(tc, q, k, v, shape):
    # E1 _standard_call (2026), adapted only for B and noncausal branch.
    B, H, K, L, S, D = (shape[x] for x in ['B', 'H', 'KVH', 'L', 'S', 'D'])
    qg = q.reshape([B, K, (H // K) * L, D])
    scores = tc.matmul(qg, k, alpha=1 / math.sqrt(D), trans_b=True)
    scores = scores.reshape([B, H, L, S])
    weights = tc.causal_softmax(scores) if shape['causal'] else scores.softmax(-1)
    wg = weights.reshape([B, K, (H // K) * L, S])
    return tc.matmul(wg, v).reshape([B, H, L, D])


def sdpa_call(torch, spec, q, k, v, shape):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right
    # PyTorch SDPA (2023-2026). Efficient lacks native GQA: its explicit
    # repetition is inside the measured call and counted in transient memory.
    gqa = shape['H'] != shape['KVH']
    if spec['backend'] == 'EFFICIENT_ATTENTION' and gqa:
        k = k.repeat_interleave(shape['H'] // shape['KVH'], dim=1)
        v = v.repeat_interleave(shape['H'] // shape['KVH'], dim=1)
        gqa = False
    causal = shape['causal'] and shape['L'] > 1
    bias = None
    if causal and shape['L'] != shape['S']:
        bias = causal_lower_right(shape['L'], shape['S'])
        causal = False
    with sdpa_kernel([getattr(SDPBackend, spec['backend'])]):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=causal, dropout_p=0.0,
            scale=1 / math.sqrt(shape['D']), enable_gqa=gqa)


def metrics(got, want):
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape or got.size == 0:
        raise ValueError('output shape mismatch or empty output')
    bad_got, bad_ref = int((~np.isfinite(got)).sum()), int((~np.isfinite(want)).sum())
    if bad_got or bad_ref:
        return dict(max_abs=None, relative_frobenius=None, nonfinite_got=bad_got,
                    nonfinite_reference=bad_ref, relative_undefined=True)
    diff = got.astype(np.float64) - want.astype(np.float64)
    num, den = float(np.linalg.norm(diff.ravel())), float(np.linalg.norm(want.astype(np.float64).ravel()))
    rel = num / den if den else (0.0 if num == 0 else None)
    return dict(max_abs=float(np.max(np.abs(diff))), relative_frobenius=rel,
                nonfinite_got=0, nonfinite_reference=0, relative_undefined=rel is None)


def summarize_samples(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 7 or not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError('need >=7 finite positive timed calls')
    q1, med, q3 = np.quantile(arr, [.25, .5, .75], method='linear')
    return dict(median_ms=float(med), iqr_ms=float(q3 - q1), q1_ms=float(q1),
                q3_ms=float(q3), samples_ms=arr.tolist(), calls=len(arr))


def matching_fraction(shape, delta_fraction, two_pass_fraction):
    # SP1's fixed .02 criterion is reused, never fitted to this sweep.
    diff = abs(delta_fraction - two_pass_fraction)
    return dict(absolute_difference=diff, within_002=diff <= .02,
                matched_budget_eligible=diff <= .02 and shape['family'] == 'sp1',
                label='CPU z-score sample estimate versus GPU SP mask sample; not exact baseline CUDA selection',
                calibration=shape['delta_status'])


def receipt_valid(receipt, reg, current_fingerprint):
    if receipt.get('fingerprint') != current_fingerprint:
        return False
    cell = receipt.get('shape', {})
    if cell not in reg['shapes']:
        return False
    specs = {s['id']: s for s in registry()}
    rows = receipt.get('rows', [])
    if len(rows) != len(specs) or {r.get('id') for r in rows} != set(specs):
        return False
    for row in rows:
        if row.get('status') not in {'OK', 'UNAVAILABLE', 'BLOCKED', 'BLOCKED_AFTER_ERROR',
                                     'BLOCKED_SP2_INTERFACE', 'RED_NONFINITE', 'ERROR'}:
            return False
        if row.get('status') == 'UNAVAILABLE' and specs[row['id']]['required']:
            return False
        if row.get('dtype') != specs[row['id']]['dtype']:
            return False
        if row.get('status') == 'OK':
            try:
                summarize_samples(row['timing']['samples_ms'])
                if row['peak_allocated_delta_bytes'] < 0:
                    return False
                if row['accuracy']['nonfinite_got'] or row['accuracy']['nonfinite_reference']:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    return True
