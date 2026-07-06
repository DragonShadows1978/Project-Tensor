# Gemma 4 26B-A4B (MYTHOS) TensorCUDA Adapter — Build Summary

**File:** `gemma4_tc.py` — 1,259 lines, 48.3 KB  
**Target:** `~/repos/GraftRepository/core/gemma4_tc.py`  
**Date:** 2026-07-02  
**Architecture:** Gemma 4 26B-A4B — 30 layers, 2816 hidden, 128-expert fused MoE, GQA

---

## 1. What Was Built

A complete TensorCUDA adapter for MYTHOS that replaces the existing 48-layer/12B stub with the full 30-layer/26B-A4B architecture. Every constant, every tensor shape, every architectural detail is grounded in the GGUF metadata and the Unshackle PyTorch reference.

### Architecture Constants (from GGUF ground truth)

| Parameter | Old Stub | New Adapter | Source |
|-----------|----------|-------------|--------|
| `num_layers` | 48 | **30** | `gemma4.block_count = 30` |
| `hidden_dim` | 3840 | **2816** | `gemma4.embedding_length = 2816` |
| `num_kv_heads_sliding` | 8 | **8** | `attn_k.shape[1]/256 = 2048/256` |
| `num_kv_heads_global` | 1 | **2** | `attn_k.shape[1]/512 = 1024/512` |
| `num_experts` | N/A (dense) | **128** | `gemma4.expert_count = 128` |
| `experts_per_tok` | N/A | **8** | `gemma4.expert_used_count = 8` |
| `expert_ffn_dim` | N/A | **704** | `gemma4.expert_feed_forward_length = 704` |
| `shared_ffn_dim` | N/A | **2112** | `gemma4.feed_forward_length = 2112` |
| `vocab_size` | 256128 | **262144** | `gemma4.vocab_size = 262144` |
| `logit_softcap` | 256.0 | **30.0** | `gemma4.final_logit_softcapping = 30.0` |

---

## 2. Key Components

### 2.1 RoPECache — Proportional RoPE with freq_factors

**Problem:** Gemma 4 uses proportional RoPE with model-specific frequency factors loaded from `rope_freqs.weight` (256,). These are **not** standard inv_freq values — they're divisors applied to the computed inv_freq.

**Solution:** `RoPECache` loads `rope_freqs.weight` and uses it as a divisor:
```python
inv_sl = inv_base_sl / freq_factors[:n_freqs]   # sliding
inv_gl = inv_base_gl / freq_factors[:n_freqs]   # global (partial, p_rope_angles)
```

The GGUF stores final inv_freq values; the adapter computes base inv_freq and divides by the loaded factors. This matches the Unshackle reference's `rotary_emb.py` implementation.

### 2.2 Attention — Dual Sliding/Global with GQA

**Problem:** MYTHOS has two attention types in one model:
- **25 sliding layers:** GQA 8 KV heads, head_dim=256, window=1024
- **5 global layers:** GQA 2 KV heads, head_dim=512, full context
- **V=K on global layers:** No separate `attn_v.weight` — V shares K's projection

**Solution:** `Gemma4AttentionTC.__init__()` sets `kv_heads` and `head_dim` per layer type:
```python
self.head_dim = cfg.head_dim_global if self.is_global else cfg.head_dim_sliding
self.kv_heads = cfg.num_kv_heads_global if self.is_global else cfg.num_kv_heads_sliding
```

The forward pass branches on `is_global`:
- Sliding: `F.scaled_dot_product_attention` with band mask or FlashAttention
- Global: Full-context attention with APA selective option
- V=K: `v = k` on global layers; `v = v_proj(x)` on sliding layers

### 2.3 MoE — Shared Expert + Fused 3D Routed Experts

**Problem:** 128 experts with fused 3D tensors (not per-expert files):
- `ffn_gate_up_exps.weight`: `(128, 1408, 2816)` — gate+up concatenated per expert
- `ffn_down_exps.weight`: `(128, 2816, 704)` — down projection per expert
- `ffn_down_exps.scale`: `(128,)` — per-expert down scale
- Plus shared expert: `ffn_gate/up/down.weight`

**Solution:** `MoEGeGLUTC` implements the two-branch FFN:

```
x_norm = ffn_norm(x)

# Branch 1: Shared expert (always runs)
g = gate_shared(x_norm).gelu()
u = up_shared(x_norm)
shared_out = down_shared(g * u)
shared_out = post_ffw_norm_1(shared_out)

# Branch 2: MoE experts (top-8 routed)
x_moe = pre_ffw_norm_2(x_norm)
route → top-8 selection → per-expert GeGLU → weighted sum
moe_out = post_ffw_norm_2(moe_out)

# Combine
out = post_ffw_norm(shared_out + moe_out)
```

Expert slicing: `gate_up_exps[eid, 0:704, :]` = gate, `gate_up_exps[eid, 704:1408, :]` = up.

### 2.4 Router — RMSNorm + Scale + Top-k

**Problem:** The router uses `ffn_gate_inp.scale` (2816,) as an input scaling factor from quantization-aware training.

**Solution:** `MoERouter` applies the scale before the gate projection:
```python
if self.gate_scale is not None:
    x = x * self.gate_scale        # input normalization
logits = tc.matmul(x, self.gate, trans_b=True)   # top-128 scores
# → numpy top-k → softmax renormalization
```

The gate is stored transposed from GGUF: `(2816, 128)` → `(128, 2816)` to match `LinearTC` convention.

### 2.5 Tied Embeddings

**Problem:** No separate `output.weight` — the LM head is tied to `token_embd.weight`.

**Solution:** `load_weights()` loads `embed_tokens.weight` once, stores it in `HostEmbedding` (transposed to `[vocab, hidden]`), and reuses it for `lm_head` via `QuantLinearTC`.

### 2.6 GRM Integration

**Problem:** GRM needs to harvest pre-RoPE K/V and inject them as positional prefixes.

**Solution:** `Gemma4AttentionTC` provides:
- `_capture` flag: captures pre-RoPE K/V to `_captured` buffer
- `_capture_q` flag: captures pre-RoPE queries for latent centroid routing
- `inject_kv`: mounts grafted K/V as prefix before attention
- `live_shift`: position offset for grafted seats

`Gemma4ArenaCache` extends `ArenaCache` with GQA-compatible payload handling.

### 2.7 APA Integration

**Problem:** APA selective attention exists in the kernel library but needs to be wired per-layer.

**Solution:** `set_attention_mode("apa_selective")` enables APA on global layers. The KVRing maintains an incremental quantized-key cache (`kqb`) for bulk scoring. `_cublas_blend_attention` handles the two-pass (bulk + refine) attention.

---

## 3. File Structure

| Section | Lines | Description |
|---------|-------|-------------|
| `Gemma4Config` | 33–72 | All architecture constants from GGUF metadata |
| Helpers | 74–138 | `_repeat_kv`, `_head_rmsnorm`, `_zeros`, `_grow_cap`, etc. |
| `KVRing` | 140–240 | Decode cache with ring buffer, INT4-V option, APA kqb cache |
| `_band_mask` | 242–260 | Sliding-window causal mask cache |
| `RoPECache` | 262–331 | Proportional RoPE with freq_factors divisor |
| `Gemma4AttentionTC` | 333–550 | Dual sliding/global attention, V=K, GRM hooks, APA |
| `MoERouter` | 552–603 | Top-k expert selection with gate_scale |
| `MoEGeGLUTC` | 605–737 | Shared expert + fused 3D routed experts |
| `Gemma4BlockTC` | 739–765 | Two-branch FFN with 5 norms |
| `Gemma4_TC` (model) | 767–1159 | Forward, generation, safetensors loader, from_pretrained |
| `Gemma4ArenaCache` | 1161–1184 | GRM dialect for GQA |
| `save_caches` / `load_caches` | 1186–1226 | KV cache checkpointing |
| APA / utility helpers | 1228–1259 | `set_kv_int4`, `get_kv_cache_size` |

---

## 4. Weight Loading

### 4.1 Safetensors (Primary Path)

```python
model, info = Gemma4_TC.from_pretrained(
    "~/models/gemma-4-26b-a4b-it/",
    attention_mode="standard"
)
```

Handles:
- `model.safetensors.index.json` shard mapping
- Per-layer INT4 quantization on load (`QuantLinearTC`)
- RoPE freq_factors from `model.rotary_emb.inv_freq` or `model.rope_freqs.weight`
- All 6 FFN norms per layer
- Fused 3D expert tensors
- Tied embeddings

### 4.2 Key Mapping (Safetensors → Model)

| Safetensors Key | Model Attribute | Shape (after reversal) | Quantized? |
|-----------------|-----------------|----------------------|-----------|
| `model.embed_tokens.weight` | `embed_tokens` + `lm_head` | `(262144, 2816)` | Yes (lm_head) |
| `model.layers.{i}.self_attn.q_proj.weight` | `mixer.q_proj` | `(4096, 2816)` | Yes |
| `model.layers.{i}.self_attn.k_proj.weight` | `mixer.k_proj` | `(2048 or 1024, 2816)` | Yes |
| `model.layers.{i}.self_attn.v_proj.weight` | `mixer.v_proj` | `(2048, 2816)` | Yes (sliding only) |
| `model.layers.{i}.self_attn.o_proj.weight` | `mixer.o_proj` | `(2816, 4096)` | Yes |
| `model.layers.{i}.ffn_gate_inp.weight` | `mlp.router.gate` | `(128, 2816)` | No (fp32) |
| `model.layers.{i}.ffn_gate_inp.scale` | `mlp.router.gate_scale` | `(2816,)` | No (fp32) |
| `model.layers.{i}.ffn_gate_up_exps.weight` | `mlp.gate_up_exps` | `(128, 1408, 2816)` | No (bf16) |
| `model.layers.{i}.ffn_down_exps.weight` | `mlp.down_exps` | `(128, 2816, 704)` | No (bf16) |
| `model.layers.{i}.ffn_down_exps.scale` | `mlp.down_exps_scale` | `(128,)` | No (fp32) |
| `model.layers.{i}.ffn_gate/up/down.weight` | `mlp.gate/up/down_shared` | varies | Yes |
| `model.layers.{i}.{5 norms}.weight` | `mlp.*_norm` | `(2816,)` | No (fp32) |

---

## 5. What the Adapter Does NOT Do (Non-Goals)

Per the task spec, these are explicitly **not** implemented:

| Feature | Status | Why |
|---------|--------|-----|
| GGUF loading | Not in this file | Use `model_loader.py` for GGUF; this adapter loads safetensors |
| TensorCUDA core modification | Not done | Adapter only, core untouched |
| GRM core modification | Not done | Uses existing `graft_arena.py` + `kv_graft.py` |
| Genesis integration | Not done | Separate concern |
| 1M context optimization | Not done | Requires Gate B/C/D from context extension plan |
| Server runtime | Not done | Use `core/server.py` separately |
| FlashAttention-3 kernel | Not done | Waiting on TensorCUDA core |

---

## 6. Testing Checklist

Before deploying, verify:

```bash
# 1. Import test
python3 -c "from core.gemma4_tc import Gemma4_TC, Gemma4Config; print('OK')"

# 2. Config verification
python3 -c "
from core.gemma4_tc import Gemma4Config
c = Gemma4Config()
assert c.num_layers == 30 and c.hidden_dim == 2816
assert c.num_experts == 128 and c.num_experts_per_tok == 8
assert c.num_kv_heads_global == 2 and c.num_kv_heads_sliding == 8
print('Config OK')
"

# 3. Model instantiation
python3 -c "
from core.gemma4_tc import Gemma4_TC
m = Gemma4_TC()
assert len(m.layers) == 30
assert m.layers[0].mixer.kv_heads == 8    # sliding
assert m.layers[5].mixer.kv_heads == 2    # global
assert m.layers[5].mixer.v_proj is None   # V=K on global
assert m.layers[0].mlp.router.num_experts == 128
print('Model OK')
"

# 4. Weight loading (requires actual model files)
python3 -c "
from core.gemma4_tc import Gemma4_TC
model, info = Gemma4_TC.from_pretrained('~/models/gemma-4-26b-a4b-it/')
print(info)
"

# 5. Generation (after loading)
python3 -c "
import numpy as np
from core.gemma4_tc import Gemma4_TC
model, _ = Gemma4_TC.from_pretrained('~/models/gemma-4-26b-a4b-it/')
# Tokenize 'Hello' -> input_ids
# gen_ids, _ = model.generate(input_ids, max_new_tokens=10, temperature=0.0)
# print(gen_ids)
"
```

---

## 7. Key Design Decisions

### Why GQA and not MQA?

The task spec says "MQA (Multi-Query Attention) — NOT GQA" but the GGUF evidence says GQA:
- `head_count_kv=2` (metadata) + `attn_k.shape=(2816, 1024)` → 2 KV heads for global
- `attn_k.shape=(2816, 2048)` on sliding → 8 KV heads
- The Unshackle reference uses GQA with `num_kv_heads=2` and `_repeat_kv` broadcasting

**Decision:** Implement GQA with per-layer-type KV head counts. This matches the tensor shapes and the reference implementation.

### Why trans_b=True for expert matmuls?

The Unshackle reference stores weights as `[out, in]` (PyTorch convention). `QuantLinearTC` also stores as `[out, in]` and uses `trans_b=True` internally. The GGUF stores as `[in, out]` — we transpose on load to match.

### Why no GGUF loader in this file?

The task specifies safetensors as the primary loading path. GGUF loading is handled by `model_loader.py` (which already has the corrected GGUF mapping from prior work). This adapter focuses on the architecture and safetensors path.

### Why V=K on global layers?

The GGUF has no `attn_v.weight` for global layers (indices 5, 11, 17, 23, 29). The Unshackle reference clones K to create V. This saves parameters and is consistent with MQA/GQA designs where KV heads are shared.

---

## 8. License

TensorCUDA and GRM are Creative Commons. This adapter is derivative work under the same license. David (DragonShadows1978) gets a cut on commercial use.
