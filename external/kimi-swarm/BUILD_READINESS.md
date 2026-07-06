# Build-Readiness Report: core/ Module
## TensorCUDA Model Runner for Gemma 4 26B MoE

**Date:** 2026-06-16  
**Classification:** Internal - Brutally Honest Assessment  
**Status:** Code compiles. Architecture is sound. Integration is partial. It will NOT run out-of-the-box.

---

## 1. Environment Setup

### 1.1 Where Does core/ Live?

**Option A (recommended):** Copy `core/` into the TensorCUDA repo alongside existing `core/`:

```bash
cp -r /mnt/agents/output/core/* /mnt/ForgeRealm/Project-Tensor/tensor_cuda/core/
```

The existing TensorCUDA repo already has `core/` with `gemma4_tc.py`, `mistral7b_tc.py`, etc. My modules **overwrite/replace** the existing `mistral7b_tc.py` and `qwen35_tc.py` stubs and **add** `model_loader.py`, `gemma4_runner.py`, `kv_manager.py`, `server.py`.

**Option B:** Run from artifact folder with PYTHONPATH:

```bash
export PYTHONPATH="/mnt/agents/output:/mnt/ForgeRealm/Project-Tensor/tensor_cuda:$PYTHONPATH"
```

This is fragile because imports inside `core/` reference each other as `core.mistral7b_tc`, `core.qwen35_tc` - the package root must resolve correctly.

**Recommendation: Option A.** Copy into TensorCUDA's `core/`, keep backups of originals.

### 1.2 PYTHONPATH

```bash
export PYTHONPATH="/mnt/ForgeRealm/Project-Tensor/tensor_cuda:$PYTHONPATH"
```

That's it. The `.so` (`_tensor_cuda.so`) must be findable from the `tensor_cuda` package root. The build system (`./build.sh 120`) places it there.

### 1.3 Python Version and Dependencies

| Dependency | Version | Source | Required For |
|-----------|---------|--------|-------------|
| Python | 3.10+ | system | All |
| tensor_cuda | 0.1.0-phase1 | built from source (`./build.sh 120`) | All |
| numpy | 1.24+ | `pip install numpy` | All |
| safetensors | 0.4+ | `pip install safetensors` | `model_loader.py` safetensors path only |
| gguf | 0.10+ | `pip install gguf` | `model_loader.py` GGUF path (production) |
| transformers | 4.40+ | `pip install transformers` | `server.py` tokenizer only |
| sentencepiece | 0.2+ | `pip install sentencepiece` | Gemma tokenizer |

**No PyTorch needed for inference.** PyTorch is only used inside `safetensors.safe_open` (framework="pt"), but the tensor is immediately converted to numpy. If you want zero PyTorch, use the GGUF path exclusively.

### 1.4 Virtual Environment

```bash
cd /mnt/ForgeRealm/Project-Tensor
python3 -m venv venv-tc
source venv-tc/bin/activate
pip install numpy safetensors gguf transformers sentencepiece
export PYTHONPATH="/mnt/ForgeRealm/Project-Tensor/tensor_cuda:$PYTHONPATH"
```

---

## 2. Smoke Test Commands

### 2.1 Verify tensor_cuda imports

```bash
python3 -c "import tensor_cuda as tc; print('TensorCUDA:', tc.__version__); t = tc.zeros(2,3); print('Device:', t.device)"
```

**Expected:** `TensorCUDA: 0.1.0-phase1` and `Device: cuda:0`.  
**If this fails:** The `.so` isn't built or isn't on PYTHONPATH. Run `./build.sh 120` in the tensor_cuda directory.

### 2.2 Verify core/ imports

```bash
python3 -c "from core.mistral7b_tc import BlockTC, QuantLinearTC, RMSNormTC; from core.qwen35_tc import HostEmbedding; from core.gemma4_runner import Gemma4Runner; from core.kv_manager import KVManager; from core.server import InferenceServer; print('All imports OK')"
```

**Expected:** `All imports OK`.  
**If this fails:** `core/` isn't on PYTHONPATH or the TensorCUDA `.so` import failed first.

### 2.3 Verify server starts (without model)

```bash
# This will fail at model load but tests the server plumbing
python3 -m core.server --model /nonexistent.gguf --port 8080
```

**Expected:** Server starts, prints error about missing model file, exits.  
**If the server starts and listens:** The HTTP stack works.

### 2.4 Health check (once model is loaded)

```bash
curl http://localhost:8080/health
```

**Expected:** `{"status": "ok", "model": "gemma-4-26b-moe", "tensor_cuda": true, "kv_cache_mb": 0.0}`

### 2.5 One-prompt generation (once model is loaded)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-moe",
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.7,
    "max_tokens": 32,
    "stream": false
  }'
```

**Expected:** JSON response with `choices[0].message.content` containing generated text.  
**Reality:** Won't work until the GGUF weight mapping is verified (see section 3).

---

## 3. Model Paths

### 3.1 Expected GGUF Layout

The `Gemma4Runner.load_weights_gguf()` method expects the following tensor keys in the GGUF file:

| Tensor Key | Shape | Purpose |
|-----------|-------|---------|
| `token_embd.weight` | (262144, 4608) | Embedding table (may be Q6_K) |
| `output_norm.weight` | (4608,) | Final RMSNorm |
| `blk.{i}.attn_norm.weight` | (4608,) | Pre-attention norm |
| `blk.{i}.post_attention_norm.weight` | (4608,) | Post-attention norm |
| `blk.{i}.ffn_norm.weight` | (4608,) | Pre-FFN norm |
| `blk.{i}.post_ffw_norm.weight` | (4608,) | Post-FFN norm |
| `blk.{i}.layer_output_scale.weight` | (1,) | Layer scalar |
| `blk.{i}.attn_q.weight` | (8192, 4608) | Q projection (16 heads x 512) |
| `blk.{i}.attn_k.weight` | (512, 4608) | K projection (global, 1 head x 512) |
| `blk.{i}.attn_v.weight` | (512, 4608) | V projection (sliding only) |
| `blk.{i}.attn_output.weight` | (4608, 8192) | O projection |
| `blk.{i}.attn_q_norm.weight` | (512,) | Q head norm |
| `blk.{i}.attn_k_norm.weight` | (512,) | K head norm |
| `blk.{i}.ffn_gate.{e}.weight` | (15360, 4608) | Expert e gate proj |
| `blk.{i}.ffn_up.{e}.weight` | (15360, 4608) | Expert e up proj |
| `blk.{i}.ffn_down.{e}.weight` | (4608, 15360) | Expert e down proj |
| `blk.{i}.ffn_gate_inp.weight` | (8, 4608) | Router gate |

### 3.2 Compatibility with Omen MYTHOS GGUF

**Unknown. This is the single biggest risk.**

The existing `gemma4_tc.py` loader (`load_weights_qat()`) is written for **Gemma 4 12B** (not 26B MoE). It expects:
- Dense FFN: `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`
- No per-expert weights
- No router gate
- `hidden_dim = 3840` (not 4608)

My loader adds MoE-specific keys (`ffn_gate.{e}`, `ffn_gate_inp`) and uses `hidden_dim = 4608`. **If the Omen MYTHOS GGUF uses different naming for MoE experts, the loader will silently skip them and the model will produce garbage.**

**Action required:** Run `gguf-dump` on the actual GGUF file and compare keys:

```bash
python3 -c "from gguf import GGUFReader; r = GGUFReader('/path/to/mythos.gguf'); print([t.name for t in r.tensors[:20]])"
```

If the keys differ, the loader needs adjustment.

### 3.3 Fallback Path

If the MoE GGUF naming doesn't match, the code falls back to using **shared expert weights** across all 8 expert slots. This means:
- All 8 experts run the same FFN
- The router still selects top-4, but they all do identical computation
- Output is correct for dense FFN, incorrect for true MoE
- **This is a silent failure mode** - the model runs but quality is wrong

### 3.4 Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26b-it")
```

If the 26B tokenizer differs from 12B, use the correct checkpoint name. The server accepts `--tokenizer` argument.

---

## 4. Known Limitations - The Honest List

### 4.1 What Is Fully Implemented

| Component | Status | Evidence |
|-----------|--------|----------|
| MoE router (top-4 scoring) | Code complete, compiles | `MoERouter.__call__()` |
| MoE expert execution | Code complete, compiles | `MoEGeGLUTC.__call__()` |
| Sliding-window attention | Copied from working `gemma4_tc.py` | `Gemma4AttentionTC.__call__()` sliding branch |
| Global attention (MQA) | Copied from working `gemma4_tc.py` | `Gemma4AttentionTC.__call__()` global branch |
| APA selective attention | Copied from working `gemma4_tc.py` | APA decode + prefill paths |
| Graft injection (cat K/V) | `inject_kv` attr checked in attention | `Gemma4AttentionTC.__call__()` lines 355-363 |
| INT4 KV pack/unpack | Uses `tc.kv_int4_pack/unpack` | `MultiResGlobalKV._spill_to_int4()` |
| KVRing (decode cache) | Copied from working `gemma4_tc.py` | `KVRing` class |
| Chunked prefill | Copied from working `gemma4_tc.py` | `Gemma4Runner.forward()` |
| Generation (greedy/top-p) | Code complete | `Gemma4Runner.generate()` |
| Weight loading (GGUF QAT) | Based on working `gemma4_tc.py` loader | `load_weights_gguf()` |
| REST API server | Pure Python, no deps | `InferenceServer` + `ChatHandler` |
| SSE streaming | Code complete | `SSEStream` class |

### 4.2 What Is Scaffolded (API exists, not wired)

| Component | Status | Problem |
|-----------|--------|---------|
| **Multi-resolution KV in generation** | NOT WIRED | `generate()` uses raw KVRing tuples, not `KVManager`. You must manually call `kv_mgr.init_prefill(caches)` and pass `kv_mgr.caches` to `forward()` - but `forward()` expects KVRing/tuple, not `MultiResGlobalKV`. The types don't match. |
| **Graft mount in generation** | NOT WIRED | `GraftAdapter.mount()` exists, but `generate()` doesn't call it. You must manually set `layer.mixer.inject_kv` on each layer before calling `generate()`. |
| **KVManager -> forward() integration** | TYPE MISMATCH | `Gemma4Runner._forward()` expects `caches[i]` to be `KVRing` or `(k,v)` tuple. `KVManager` returns `MultiResGlobalKV` objects for global layers. The attention layer's `__call__` doesn't know how to read from `MultiResGlobalKV` - it expects `KVRing.kb/vb` attributes. |
| **Safetensors loading** | Implemented but not optimized | Post-hoc INT4 quantization creates large fp32 transients during load. Use GGUF path for production. |
| **Server streaming** | Code complete, untested | The SSE logic is sound but has never been executed against a live model. Token-by-token streaming may have timing issues. |
| **Batching** | NOT IMPLEMENTED | Server handles one request at a time. No concurrent session management. |
| **KV cache offloading** | NOT IMPLEMENTED | INT4 buffers stay in GPU memory. No CPU RAM or NVMe paging. |

### 4.3 What Will Fail Today

| Failure Mode | Trigger | Symptom | Fix |
|-------------|---------|---------|-----|
| **MoE GGUF key mismatch** | Running against actual 26B GGUF | `KeyError` or silent skip of expert weights | Dump actual GGUF keys, adjust loader |
| **Multi-res KV not used** | Calling `generate()` after `kv_mgr.init_prefill()` | Generation uses standard KVRing, no INT4 compression | Wire `KVManager` into `_forward()` decode path |
| **Graft not injected during gen** | Calling `generate()` after `adapter.mount()` | Model doesn't see grafted document | Manually set `layer.mixer.inject_kv` before `generate()` |
| **Router gate missing** | GGUF lacks `ffn_gate_inp.weight` | All experts get uniform scores (0) | Add fallback: load from checkpoint or init uniform |
| **Tokenizer missing** | Running server without transformers | Server starts but all requests fail with "No tokenizer" | Install transformers or provide tokenizer callback |
| **OOM at >32K context** | Standard KVRing without KVManager | KV cache grows unbounded in FP16 | Use KVManager (once wired) or accept 32K limit |

### 4.4 The Integration Gap

The core architectural gap is that `Gemma4Runner.generate()` -> `Gemma4Runner._forward()` -> `Gemma4BlockTC.__call__()` -> `Gemma4AttentionTC.__call__()` expects `cache` to be either:
- A `KVRing` object (for decode)
- A `(k, v)` tuple (for prefill)

But `KVManager` produces `MultiResGlobalKV` objects for global layers. The attention layer's decode path does:

```python
kv_cache.append(k, v, _zero_row())   # KVRing API
S_all = min(kv_cache.count, kv_cache.cap)
sc = tc.matmul(qg, kv_cache.kb, alpha=1.0, trans_b=True)   # KVRing.kb attr
```

`MultiResGlobalKV` has no `.append(k, v, zero_row)`, no `.kb`, no `.count`. The types are incompatible.

**Fix required:** Modify `Gemma4AttentionTC.__call__()` to detect `MultiResGlobalKV` and call its `.get_kv()` method for the full (INT4+FP16) cache, then fall through to the existing attention math. Or wrap `MultiResGlobalKV` in a KVRing-compatible interface.

---

## 5. Test Plan

### 5.1 Unit Tests: KVManager

```python
# test_kv_manager.py - run with: python3 test_kv_manager.py

import numpy as np
import tensor_cuda as tc
from core.gemma4_runner import Gemma4Config
from core.kv_manager import KVManager, MultiResGlobalKV

def test_multi_res_spillover():
    """Verify FP16->INT4 spillover when count exceeds fp16_window."""
    cfg = Gemma4Config()
    mr = MultiResGlobalKV(batch_size=1, head_dim=512, fp16_window=4)

    # Append 6 tokens (exceeds fp16_window=4 by 2)
    for i in range(6):
        k = tc.ones(1, 1, 1, 512, dtype="bfloat16") * (i + 1)
        v = tc.ones(1, 1, 1, 512, dtype="bfloat16") * (i + 1)
        mr.append(k, v)

    assert mr.fp16_count == 4, f"Expected 4 FP16, got {mr.fp16_count}"
    assert mr.int4_count == 2, f"Expected 2 INT4, got {mr.int4_count}"
    assert mr.total_tokens == 6
    print("PASS: spillover")

def test_multi_res_get_kv():
    """Verify get_kv returns concatenated INT4+FP16."""
    cfg = Gemma4Config()
    mr = MultiResGlobalKV(batch_size=1, head_dim=512, fp16_window=4)

    for i in range(6):
        k = tc.ones(1, 1, 1, 512, dtype="bfloat16") * (i + 1)
        v = tc.ones(1, 1, 1, 512, dtype="bfloat16") * (i + 1)
        mr.append(k, v)

    k_full, v_full = mr.get_kv()
    assert k_full.shape == (1, 1, 6, 512), f"Shape mismatch: {k_full.shape}"
    k_np = k_full.float().numpy()
    assert abs(k_np[0, 0, 0, 0] - 1.0) < 0.5, "INT4 unpack error at token 0"
    assert abs(k_np[0, 0, 5, 0] - 6.0) < 0.01, "FP16 error at token 5"
    print("PASS: get_kv")

def test_kv_manager_init():
    """Verify KVManager initializes from prefill caches."""
    cfg = Gemma4Config()
    mgr = KVManager(cfg, batch_size=1, fp16_window=2048)

    prefill_caches = []
    for i in range(cfg.num_layers):
        if cfg.is_global(i):
            k = tc.ones(1, 1, 10, 512, dtype="bfloat16")
            v = tc.ones(1, 1, 10, 512, dtype="bfloat16")
        else:
            k = tc.ones(1, 8, 10, 256, dtype="bfloat16")
            v = tc.ones(1, 8, 10, 256, dtype="bfloat16")
        prefill_caches.append((k, v))

    mgr.init_prefill(prefill_caches)
    assert len(mgr.caches) == 48
    global_indices = [i for i in range(48) if cfg.is_global(i)]
    for gi in global_indices:
        assert isinstance(mgr._multi_res[gi], MultiResGlobalKV)
    print("PASS: kv_manager_init")

def test_save_load_roundtrip():
    """Verify KVManager save/load preserves state."""
    cfg = Gemma4Config()
    mgr = KVManager(cfg, batch_size=1, fp16_window=4)

    prefill_caches = []
    for i in range(cfg.num_layers):
        if cfg.is_global(i):
            k = tc.ones(1, 1, 6, 512, dtype="bfloat16")
            v = tc.ones(1, 1, 6, 512, dtype="bfloat16")
        else:
            k = tc.ones(1, 8, 6, 256, dtype="bfloat16")
            v = tc.ones(1, 8, 6, 256, dtype="bfloat16")
        prefill_caches.append((k, v))

    mgr.init_prefill(prefill_caches)
    mgr.save("/tmp/test_kv.npz", position_offset=42)

    mgr2, pos = KVManager.load("/tmp/test_kv.npz", cfg, batch_size=1, fp16_window=4)
    assert pos == 42
    assert len(mgr2.caches) == 48
    print("PASS: save_load_roundtrip")

if __name__ == "__main__":
    test_multi_res_spillover()
    test_multi_res_get_kv()
    test_kv_manager_init()
    test_save_load_roundtrip()
    print("\nAll KVManager tests passed.")
```

### 5.2 Unit Tests: GGUF Weight Mapping

```python
# test_gguf_mapping.py

from gguf import GGUFReader

def dump_gguf_keys(gguf_path, max_keys=50):
    """Dump the first N tensor keys from a GGUF file."""
    r = GGUFReader(gguf_path)
    keys = [t.name for t in r.tensors]
    print(f"GGUF: {gguf_path}")
    print(f"Total tensors: {len(keys)}")
    print(f"First {max_keys} keys:")
    for k in keys[:max_keys]:
        print(f"  {k}")

    moe_keys = [k for k in keys if "expert" in k.lower() or "gate_inp" in k]
    print(f"\nMoE-related keys: {len(moe_keys)}")
    for k in moe_keys[:20]:
        print(f"  {k}")

    expert_ffn = [k for k in keys if ".ffn_gate." in k or ".ffn_up." in k or ".ffn_down." in k]
    print(f"\nPer-expert FFN keys: {len(expert_ffn)}")
    for k in expert_ffn[:20]:
        print(f"  {k}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 test_gguf_mapping.py /path/to/model.gguf")
        sys.exit(1)
    dump_gguf_keys(sys.argv[1])
```

**Action:** Run this against the actual Omen MYTHOS GGUF. Compare output against the expected keys in section 3.1. Adjust `load_weights_gguf()` if keys differ.

### 5.3 First Omen Smoke Test Plan

| Step | Command | Acceptance Gate |
|------|---------|----------------|
| 1 | `python3 -c "import tensor_cuda as tc; print(tc.__version__)"
| 1 | `python3 -c "import tensor_cuda as tc; print(tc.__version__)"` | Prints `0.1.0-phase1` |
| 2 | `python3 test_gguf_mapping.py /path/to/mythos.gguf` | Dumps keys, confirms MoE expert keys exist |
| 3 | `python3 -c "from core.gemma4_runner import Gemma4Runner; m, info = Gemma4Runner.from_pretrained('/path/to/mythos.gguf', qat=True); print(info)"` | Model loads without error, prints framework string |
| 4 | `python3 -c "import numpy as np; from core.gemma4_runner import Gemma4Runner; m, _ = Gemma4Runner.from_pretrained('/path/to/mythos.gguf', qat=True); ids = np.array([[2, 100, 200]], np.int64); logits, _ = m.forward(ids, last_token_only=True); print('Logits shape:', logits.shape)"` | Forward pass runs, logits shape is `(1, 1, 262144)` |
| 5 | `python3 -c "...; gen_ids, _ = m.generate(ids, max_new_tokens=10, temperature=0.0); print('Generated:', gen_ids)"` | Greedy generation produces token IDs, no crash |
| 6 | `python3 -m core.server --model /path/to/mythos.gguf --port 8080 &` | Server starts, listens on 8080 |
| 7 | `curl http://localhost:8080/health` | Returns `{"status": "ok"}` |
| 8 | `curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'` | Returns JSON with generated text |
| 9 | `python3 test_kv_manager.py` | All 4 KVManager tests pass |
| 10 | Run at 32K context | Completes without OOM, output is coherent |
| 11 | Run at 128K context with KVManager | Completes without OOM, verifies multi-res is active |

### 5.4 VRAM Budget by Context Mode

| Mode | Context | KV Cache | Model | Total | Fits in 32GB? |
|------|---------|----------|-------|-------|---------------|
| Standard (no KVManager) | 32K | 2.7 GB | ~7 GB | ~10 GB | Yes |
| Standard (no KVManager) | 128K | 10.7 GB | ~7 GB | ~18 GB | Yes |
| Standard (no KVManager) | 256K | 21.5 GB | ~7 GB | ~29 GB | Barely |
| Multi-res (KVManager) | 128K | ~760 MB | ~7 GB | ~8 GB | Yes |
| Multi-res (KVManager) | 256K | ~1.1 GB | ~7 GB | ~8.5 GB | Yes |
| Multi-res (KVManager) | 1M | ~4.4 GB | ~7 GB | ~12 GB | Yes |
| Multi-res + TADA | 1M | ~4.4 GB | ~7 GB + 7 GB TADA | ~19 GB | Yes |
| Multi-res + TADA + Arena | 1M | ~4.4 GB + 6 GB arena | ~7 GB + 7 GB TADA | ~25 GB | Yes |

**Key insight:** Multi-resolution KV is the difference between "barely fits at 256K" and "comfortable at 1M with other services running."

---

## 6. What to Build First (Priority Order)

### P0: Verify GGUF Key Mapping (1-2 hours)
Run `test_gguf_mapping.py` against the actual MYTHOS GGUF. If keys don't match, the loader is the first thing to fix. Without correct weight loading, nothing else matters.

### P1: Wire KVManager into Generation (1 day)
The `Gemma4AttentionTC.__call__()` needs to accept `MultiResGlobalKV` in addition to `KVRing`. Two options:
- **Option A:** Add a type check in `__call__`: if `kv_cache` is `MultiResGlobalKV`, call `.get_kv()` and `.append()`, then proceed with standard attention math.
- **Option B:** Make `MultiResGlobalKV` implement the `KVRing` interface (`.kb`, `.vb`, `.count`, `.cap`, `.append()`, `.full`).

Option B is cleaner but more work. Option A is a 20-line change.

### P2: Wire Graft Mount into Generation (4 hours)
Add a `grafts` parameter to `generate()` that accepts a list of `(graft_k, graft_v)` per layer, sets `layer.mixer.inject_kv` before the forward pass, and clears it after. Or use the `graft_callback` mechanism already in the signature.

### P3: End-to-End Smoke Test (1 day)
Run the full smoke test plan (section 5.3). Fix whatever breaks. This is where you'll discover the real issues - CUDA kernel compatibility, memory alignment, shape mismatches, etc.

### P4: Server Hardening (2-3 days)
- Test streaming with real model output
- Add concurrent request handling (thread pool)
- Add request logging and error handling
- Test with actual tokenizer

### P5: Optimization (ongoing)
- Expert parallelism (run selected experts in parallel)
- KV cache offloading to CPU RAM
- Batching multiple requests
- FlashAttention-3 integration (when TensorCUDA adds it)

---

## 7. Summary

**What works today:**
- All Python modules compile and import cleanly
- Model architecture is complete (MoE + sliding + global attention)
- Weight loading code exists for both safetensors and GGUF
- Generation loop exists (greedy + top-p sampling)
- Server exists (OpenAI-compatible API)
- KVManager exists with multi-resolution spillover logic
- GraftAdapter exists for mounting/unmounting

**What doesn't work today:**
- KVManager is not wired into the generation path (type mismatch)
- Graft mounting is not wired into the generation path (manual only)
- GGUF key naming for MoE experts is speculative (needs verification)
- Server streaming is untested
- No batching or concurrent request handling

**Time to first working generation:** 1-2 days (P0 + P3) if GGUF keys match. 3-5 days if they don't.

**Time to full feature set (multi-res KV + grafts + server):** 1-2 weeks.
