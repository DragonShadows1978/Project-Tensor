# MYTHOS Loader Patch — Static Risk Review
## Remote Code Review Against GGUF Ground Truth

**Date:** 2026-06-30  
**Scope:** Find static-code mistakes in the proposed repair before local implementation.  
**Ground Truth:** `MYTHOS_GGUF_KEY_DUMP.json` — 658 tensors, all shapes verified.  
**Method:** Static analysis only. No execution.

---

## 1. Verdict: CONDITIONAL FAIL

**Do NOT apply the patch as written.** Three CRITICAL issues will prevent the model from loading or producing correct shapes. Four additional issues will produce silent wrong-output failures. All are fixable with minimal corrections (below).

**After corrections: the patch should be locally testable.** Runtime validation is still required for 5 blocked items.

---

## 2. CRITICAL Issues (Will Break)

### 2.1 CRITICAL — Embedding Orientation Wrong for `HostEmbedding`

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.6, `load_weights_gguf()` — Embedding section  
**GGUF ground truth:** `token_embd.weight` shape `[2816, 262144]`

**The problem:**

`HostEmbedding.__call__()` does `self.weight[input_ids_np]` which indexes the **first dimension**. If `self.weight` has shape `[2816, 262144]`, then `weight[100]` returns a vector of shape `[262144]` (the vocab size), not `[2816]` (the hidden dimension). Every embedding lookup will produce a 262144-length vector instead of a 2816-length embedding.

**The repair code stores:**
```python
emb_arr = _load_and_mark("token_embd.weight")      # shape [2816, 262144]
emb_scaled = emb_arr * np.float32(cfg.hidden_dim ** 0.5)
self.embed_tokens.weight = np.ascontiguousarray(emb_scaled)  # STILL [2816, 262144]!
```

**The prototype had the same bug** — it also failed to transpose. But the prototype was never tested against a real model, so this bug was latent.

**Correction:**
```python
emb_arr = _load_and_mark("token_embd.weight")      # GGUF: [2816, 262144] = [hidden, vocab]
emb_vh = emb_arr.T                                 # [262144, 2816] = [vocab, hidden]
emb_scaled = emb_vh * np.float32(cfg.hidden_dim ** 0.5)
self.embed_tokens.weight = np.ascontiguousarray(emb_scaled)  # NOW [262144, 2816]

# lm_head is tied to (unscaled) embeddings, transposed back to [hidden, vocab] for QuantLinearTC
# QuantLinearTC expects weight as (out_features, in_features)
# For lm_head: out=vocab_size (262144), in=hidden_dim (2816)
# So we pass emb_vh directly: shape [262144, 2816] = (out, in)
self.lm_head = QuantLinearTC(np.ascontiguousarray(emb_vh), group_size=128)
```

**Verify after fix:** `model.embed_tokens.weight.shape` should be `(262144, 2816)`.

---

### 2.2 CRITICAL — `trans_b=True` Used Where `trans_b=False` Is Required

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.4, `MoEGeGLUTC._expert_forward()`  
**Also affects:** All `QuantLinearTC` calls if the kernel convention differs from `LinearTC`

**The problem:**

The repair uses `trans_b=True` for all expert matmuls:
```python
gate_out = tc.matmul(x, gate_w, trans_b=True).gelu()  # WRONG
up_out = tc.matmul(x, up_w, trans_b=True)              # WRONG
out = tc.matmul(hidden, down_w, trans_b=True)          # WRONG
```

**GGUF weight orientation:** The GGUF stores weights as `[in_features, out_features]`:
- `ffn_gate_up_exps.weight` slice: `[2816, 704]` = `[hidden_dim, expert_ffn_dim]` = `[in, out]`
- `ffn_down_exps.weight` slice: `[704, 2816]` = `[expert_ffn_dim, hidden_dim]` = `[in, out]`

For matmul `y = x @ W` where `x` is `(B, L, in)` and `W` is `(in, out)`:
- Result: `(B, L, out)` — **no transpose needed** (`trans_b=False`)

With `trans_b=True`, the computation becomes `x @ W^T`:
- `x (B, L, 2816) @ W^T (704, 2816)` — **shape mismatch** (2816 ≠ 704)

**The `LinearTC` vs `QuantLinearTC` convention confusion:**

| Class | Weight Storage | Matmul Call | Effective Computation |
|-------|---------------|-------------|----------------------|
| `LinearTC` | `(out, in)` | `matmul(x, W, trans_b=True)` | `x @ W^T = x (..., in) @ (in, out)` |
| `QuantLinearTC` | `(out, in)` from `__init__` | `int4_linear_fused(x, packed, ...)` | **Unknown** — kernel-internal |

The `LinearTC` stores weight as `(out, in)` and uses `trans_b=True` to get `x @ W^T`. But GGUF stores as `(in, out)`. The `QuantLinearTC.__init__` sets:
```python
self.out_features, self.in_features = weight_fp32.shape
```

For `attn_q.weight` [2816, 4096]: `out_features=2816, in_features=4096`. This is **backwards** from the intended `(out=4096, in=2816)`.

**Two possibilities:**

A. `int4_linear_fused` computes `x @ W` (no transpose) internally. Then the [2816, 4096] orientation means `x (..., 2816) @ W (2816, 4096) = (..., 4096)` — **correct by accident** because the kernel doesn't transpose.

B. `int4_linear_fused` computes `x @ W^T` like `LinearTC`. Then `x (..., 4096) @ W^T (4096, 2816)` — **shape mismatch** with x being (..., 2816).

**The prototype's existing attention code uses `self.q_proj(x)` without questioning this.** If the prototype was never tested, we don't know which case is true.

**Correction for `_expert_forward`:** Remove `trans_b=True` (assume GGUF weights are `[in, out]` and kernel does `x @ W`):

```python
def _expert_forward(self, x, eid):
    cfg = self.cfg
    # gate_up_exps: (2816, 1408, 128) — slice expert eid, reshape to 2D
    expert_3d = self.gate_up_exps.slice(-1, eid, 1)          # (2816, 1408, 1)
    expert_2d = expert_3d.reshape([cfg.hidden_dim, 2 * cfg.expert_ffn_dim])  # (2816, 1408)
    gate_w = expert_2d.slice(1, 0, cfg.expert_ffn_dim)       # (2816, 704)
    up_w = expert_2d.slice(1, cfg.expert_ffn_dim, cfg.expert_ffn_dim)  # (2816, 704)
    
    down_3d = self.down_exps.slice(-1, eid, 1)               # (704, 2816, 1)
    down_w = down_3d.reshape([cfg.expert_ffn_dim, cfg.hidden_dim])      # (704, 2816)
    
    # matmul: x (B, L, in) @ W (in, out) = (B, L, out) — NO trans_b
    gate_out = tc.matmul(x, gate_w).gelu()   # (B, L, 2816) @ (2816, 704) = (B, L, 704)
    up_out = tc.matmul(x, up_w)               # (B, L, 2816) @ (2816, 704) = (B, L, 704)
    hidden = gate_out * up_out                 # (B, L, 704)
    
    if self.down_exps_scale is not None:
        scale = self.down_exps_scale.slice(0, eid, 1)  # (1,)
        hidden = hidden * scale
    
    out = tc.matmul(hidden, down_w)           # (B, L, 704) @ (704, 2816) = (B, L, 2816)
    return out
```

**BLOCKED:** If this produces shape errors at runtime, try `trans_b=True` instead — the kernel's internal convention is unknown from static analysis.

---

### 2.3 CRITICAL — Expert Tensor Slices Need `reshape` to 2D

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.4, `_expert_forward()`  
**Related to:** Issue 2.2 above

**The problem:**

The repair does:
```python
gate_w = self.gate_up_exps.slice(-1, eid, 1).slice(1, 0, cfg.expert_ffn_dim)
# Result shape: (2816, 704, 1) — STILL 3D!
```

`tc.matmul` expects 2D weight matrices for the `x @ W` pattern. A `(2816, 704, 1)` tensor will cause a shape error or unexpected broadcasting.

**Correction:** See Issue 2.2 correction — the `reshape([cfg.hidden_dim, 2 * cfg.expert_ffn_dim])` call flattens the expert dimension, producing a 2D matrix suitable for matmul.

Same issue for `down_exps`:
```python
down_w = self.down_exps.slice(-1, eid, 1).reshape([cfg.expert_ffn_dim, cfg.hidden_dim])
# Was: (704, 2816, 1) → Now: (704, 2816)
```

---

## 3. HIGH/MEDIUM Issues (Silent Wrong Output)

### 3.1 HIGH — Router Gate `trans_b` Convention

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.3, `MoERouter.__call__()`

**The problem:**

```python
logits = tc.matmul(x, self.gate, trans_b=False)
```

The GGUF `ffn_gate_inp.weight` has shape `[2816, 128]`. Following the same `[in, out]` convention as attention weights:
- `x (B, L, 2816) @ gate (2816, 128)` with `trans_b=False` → `(B, L, 128)` ✓

This appears correct. But if the kernel convention differs (see 2.2), this may need `trans_b=True` with a transposed gate.

**Correction (defensive):** Load the gate transposed to match `LinearTC` convention, use `trans_b=True`:

```python
# In load_weights_gguf:
rg_arr = _load_and_mark(f"{b}.ffn_gate_inp.weight")  # [2816, 128]
if rg_arr is not None:
    # Store as (out, in) = (128, 2816) to match LinearTC convention
    mlp.router.gate = tc.tensor(np.ascontiguousarray(rg_arr.T), dtype="float32")

# In MoERouter.__call__:
logits = tc.matmul(x, self.gate, trans_b=True)  # x (B, L, 2816) @ gate^T (2816, 128) = (B, L, 128)
```

This is equivalent mathematically but matches the `LinearTC` convention consistently.

---

### 3.2 MEDIUM — Router `.scale` Application Order

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.3, `MoERouter.__call__()`

**The problem:**

```python
if self.gate_scale is not None:
    x = x * self.gate_scale  # elementwise on hidden_dim
logits = tc.matmul(x, self.gate, trans_b=True)
```

The `.scale` tensor has shape `(2816,)` — same as hidden_dim. The repair applies it **before** the gate matmul (input scaling).

**Alternative:** The scale might be a **weight scale** from quantization-aware training, meant to be applied to the gate **weights**, not the input:
```python
# Alternative: scale the gate weights
gate_scaled = self.gate * self.gate_scale  # (128, 2816) * (2816,) = (128, 2816)
logits = tc.matmul(x, gate_scaled, trans_b=True)
```

**BLOCKED:** The correct semantics require checking llama.cpp source for `ffn_gate_inp.scale` usage. Both orders are mathematically plausible. Apply to input first (as written), and if routing produces garbage, try applying to weights instead.

---

### 3.3 MEDIUM — RoPE Frequencies Loaded But Unused

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.6, `load_weights_gguf()` — RoPE section

**The problem:**

```python
if "rope_freqs.weight" in tensors_by_name:
    rope_arr = _load_and_mark("rope_freqs.weight")  # shape (256,)
    # BLOCKED: RoPE loading — stored for future use
    del rope_arr
```

The GGUF contains pre-computed RoPE frequencies that may differ from the computed values in `extend_rope()`. If they differ, the model will produce incorrect position encodings and wrong outputs.

**The `extend_rope()` method computes:**
```python
inv_l = 1.0 / (cfg.rope_theta_swa ** (np.arange(0, d, 2, np.float32) / d))
```

This is a standard RoPE frequency computation. The GGUF's `rope_freqs.weight` (256,) might be the actual frequencies used during training.

**Correction:** Store the loaded frequencies and use them instead of computing:

```python
# In load_weights_gguf:
if "rope_freqs.weight" in tensors_by_name:
    rope_arr = _load_and_mark("rope_freqs.weight")  # (256,)
    self._rope_freqs_loaded = rope_arr.copy()
else:
    self._rope_freqs_loaded = None

# In extend_rope():
def extend_rope(self, seq_len: int):
    if seq_len <= self._rope_len:
        return
    cfg = self.config
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    
    # Use loaded frequencies if available
    if hasattr(self, '_rope_freqs_loaded') and self._rope_freqs_loaded is not None:
        # BLOCKED: The loaded freqs are (256,) — need to understand how to apply them
        # This may replace the computed inv_l/inv_g entirely
        inv_l = self._rope_freqs_loaded[:cfg.head_dim_sliding // 2]
        inv_g = self._rope_freqs_loaded[:cfg.head_dim_global // 2]
    else:
        # Fallback: compute from scratch
        inv_l = 1.0 / (cfg.rope_theta_swa ** (np.arange(0, cfg.head_dim_sliding, 2, np.float32) / cfg.head_dim_sliding))
        inv_g = 1.0 / (cfg.rope_theta_global ** (np.arange(0, cfg.head_dim_global, 2, np.float32) / cfg.head_dim_global))
    
    emb_l = np.concatenate([pos * inv_l, pos * inv_l], axis=-1)
    emb_g = np.concatenate([pos * inv_g, pos * inv_g], axis=-1)
    # ... rest unchanged
```

**BLOCKED:** The `(256,)` shape suggests these are base frequencies for 256 dims. But global layers use 512 dims and sliding use 256. The mapping isn't clear from static analysis. If forward produces position-dependent garbage, this is the first place to check.

---

### 3.4 MEDIUM — Re-quantization of Q5_K/Q6_K/Q8_0/Q5_1 to INT4

**Location:** `MYTHOS_LOADER_REPAIR.md` §3.6, all `QuantLinearTC()` calls

**The problem:**

All GGUF weights are dequantized to float32 by `gguf.quants.dequantize()`, then re-quantized to INT4 by `QuantLinearTC._quantize()`. The GGUF weights were originally quantized with type-specific scales (Q5_K, Q6_K, etc.). Re-quantizing with symmetric-8 INT4 (group_size=128) will produce **different scales** and introduce quantization error.

**Impact:** Small per-layer, but accumulates across 30 layers. May cause noticeable quality degradation.

**Mitigation:** This is the standard approach for inference engines that only support INT4. The error is typically small (<1% PPL impact). If quality is poor, consider:
- Using FP16 for attention weights (Q6_K → FP16 is nearly lossless)
- Preserving original GGUF quantization scales (requires per-type dequant kernels)

**Not a blocker** — this is a design trade-off, not a bug.

---

## 4. TensorCUDA API Verification

All TensorCUDA API calls used in the repair:

| API Call | Used In | Available in Prototype? | Risk |
|----------|---------|------------------------|------|
| `tc.zeros(..., dtype=...)` | KVRing, MultiResGlobalKV | ✅ Yes | None |
| `tc.tensor(np_array, dtype=..., device=...)` | Everywhere | ✅ Yes | None |
| `tc.matmul(x, W, trans_b=..., alpha=...)` | Attention, MoE, Router | ✅ Yes | Convention risk (§2.2) |
| `tc.write_rows(dst, src, offset)` | KVRing, MultiResGlobalKV | ✅ Yes | None |
| `tc.slice(dim, start, n)` | _expert_forward, KVRing | ✅ Yes | Returns 3D if input 3D (§2.3) |
| `tc.reshape(shape)` | _expert_forward, attention | ✅ Yes | None |
| `tc.cat([...], dim=...)` | Attention (inject_kv) | ✅ Yes | None |
| `tc.empty_cache()` | Forward, KVRing | ✅ Yes | None |
| `tc.no_grad()` | Everywhere | ✅ Yes | None |
| `tc.is_grad_enabled()` | RMSNorm, RoPE | ✅ Yes | None |
| `tc.int4_linear_fused(x, packed, scales, zeros, group_size)` | QuantLinearTC | ✅ Yes | Internal convention unknown |
| `tc.kv_int4_pack(x, group=...)` | MultiResGlobalKV | ✅ Yes | None |
| `tc.kv_int4_unpack(packed, scales, group=..., lo=..., n=...)` | MultiResGlobalKV | ✅ Yes | None |
| `tensor.float()` | Sampling, KVManager | ✅ Yes | None |
| `tensor.numpy()` | Sampling, Router | ✅ Yes | None |
| `tensor.gelu()` | MoE expert | ✅ Yes | None |
| `tensor.astype(dtype)` | _cast, everywhere | ✅ Yes | None |
| `tensor * scalar` | Down scale, router scale | ✅ Yes | None |
| `tensor.softmax(dim=...)` | Attention, sampling | ✅ Yes | None |
| `F.apply_rotary(q, cos, sin)` | Attention | ✅ Yes | None |
| `F.scaled_dot_product_attention(...)` | Attention (sliding) | ✅ Yes | None |
| `F._causal_mask(L, S, device, dtype)` | Attention | ✅ Yes | None |
| `tc.rms_norm(x, weight, eps)` | RMSNormTC | ✅ Conditional | Falls back to manual if unavailable |
| `tc.rope_apply(q, cos, sin, offset)` | Attention | ✅ Conditional | Falls back to manual if unavailable |
| `tc.causal_softmax(scores)` | Attention | ✅ Conditional | Falls back to manual if unavailable |

**No unsupported APIs identified.** All calls exist in the prototype or have documented fallbacks.

---

## 5. Corrected Code: Complete Replacement Blocks

### 5.1 `load_weights_gguf()` — Embedding Section (replace §3.6 lines)

```python
        # ---- 1. Embeddings (tied output) ----
        emb_arr = _load_and_mark("token_embd.weight")
        if emb_arr is not None:
            # GGUF: token_embd.weight shape [2816, 262144] = [hidden, vocab]
            # HostEmbedding indexes first dimension by token ID
            # So we need [vocab, hidden] = [262144, 2816]
            emb_vh = emb_arr.T                                 # [262144, 2816]
            emb_scaled = emb_vh * np.float32(cfg.hidden_dim ** 0.5)
            self.embed_tokens.weight = np.ascontiguousarray(emb_scaled)

            # lm_head is TIED to embeddings — use unscaled, transposed back
            # QuantLinearTC expects (out_features, in_features)
            # lm_head: out=vocab_size (262144), in=hidden_dim (2816)
            # emb_vh is [262144, 2816] = (out, in) — correct orientation
            self.lm_head = QuantLinearTC(np.ascontiguousarray(emb_vh),
                                         group_size=128)
            del emb_arr, emb_vh, emb_scaled
            gc.collect()
```

### 5.2 `MoEGeGLUTC._expert_forward()` — Complete Replacement

```python
    def _expert_forward(self, x, eid: int):
        """Run a single expert by slicing the fused 3D tensors.

        x: (B, L, 2816)
        eid: expert index 0..127
        Returns: (B, L, 2816)

        NOTE: tc.matmul convention assumes GGUF weights are (in, out).
              If this produces shape errors, try trans_b=True instead.
        """
        cfg = self.cfg

        # Slice expert eid from gate_up_exps, reshape to 2D
        # gate_up_exps: (2816, 1408, 128)
        expert_3d = self.gate_up_exps.slice(-1, eid, 1)     # (2816, 1408, 1)
        expert_2d = expert_3d.reshape([cfg.hidden_dim, 2 * cfg.expert_ffn_dim])  # (2816, 1408)
        gate_w = expert_2d.slice(1, 0, cfg.expert_ffn_dim)          # (2816, 704)
        up_w = expert_2d.slice(1, cfg.expert_ffn_dim,
                               cfg.expert_ffn_dim)                  # (2816, 704)

        # Slice expert eid from down_exps, reshape to 2D
        # down_exps: (704, 2816, 128)
        down_3d = self.down_exps.slice(-1, eid, 1)          # (704, 2816, 1)
        down_w = down_3d.reshape([cfg.expert_ffn_dim, cfg.hidden_dim])           # (704, 2816)

        # GeGLU: matmul with NO trans_b (GGUF weights are [in, out])
        gate_out = tc.matmul(x, gate_w).gelu()   # (B, L, 2816) @ (2816, 704) = (B, L, 704)
        up_out = tc.matmul(x, up_w)               # (B, L, 2816) @ (2816, 704) = (B, L, 704)
        hidden = gate_out * up_out                 # (B, L, 704)

        # Apply per-expert down scale
        if self.down_exps_scale is not None:
            scale = self.down_exps_scale.slice(0, eid, 1)  # (1,)
            hidden = hidden * scale

        out = tc.matmul(hidden, down_w)           # (B, L, 704) @ (704, 2816) = (B, L, 2816)
        return out
```

### 5.3 `MoERouter.__call__()` — Gate Transpose (defensive)

```python
    def __call__(self, x):
        """x: (B, L, hidden_dim) -> (B, L, k) weights, (B, L, k) indices."""
        # Apply input scale if present
        if self.gate_scale is not None:
            x = x * self.gate_scale

        # Gate logits: gate stored as (out, in) = (128, 2816)
        # x (B, L, 2816) @ gate^T (2816, 128) = (B, L, 128)
        logits = tc.matmul(x, self.gate, trans_b=True)

        # Top-k selection via numpy (unchanged)
        ...
```

And in `load_weights_gguf()`, store the gate transposed:
```python
            rg_arr = _load_and_mark(f"{b}.ffn_gate_inp.weight")
            if rg_arr is not None:
                # GGUF: [2816, 128] = [in, out]
                # Store as [128, 2816] = [out, in] to match LinearTC convention
                mlp.router.gate = tc.tensor(np.ascontiguousarray(rg_arr.T),
                                            dtype="float32")
```

### 5.4 `extend_rope()` — RoPE Frequency Override

```python
    def extend_rope(self, seq_len: int):
        if seq_len <= self._rope_len:
            return
        cfg = self.config
        pos = np.arange(seq_len, dtype=np.float32)[:, None]

        # Sliding RoPE
        d_sl = cfg.head_dim_sliding
        if hasattr(self, '_rope_freqs_loaded') and self._rope_freqs_loaded is not None:
            # Use loaded frequencies for first d_sl//2 dims
            n_freqs = min(d_sl // 2, len(self._rope_freqs_loaded))
            inv_l = self._rope_freqs_loaded[:n_freqs]
            if n_freqs < d_sl // 2:
                # Pad remaining with computed values
                inv_l_comp = 1.0 / (cfg.rope_theta_swa ** (np.arange(n_freqs * 2, d_sl, 2, np.float32) / d_sl))
                inv_l = np.concatenate([inv_l, inv_l_comp])
        else:
            inv_l = 1.0 / (cfg.rope_theta_swa ** (np.arange(0, d_sl, 2, np.float32) / d_sl))
        emb_l = np.concatenate([pos * inv_l, pos * inv_l], axis=-1)

        # Global RoPE
        d_gl = cfg.head_dim_global
        if hasattr(self, '_rope_freqs_loaded') and self._rope_freqs_loaded is not None:
            n_freqs = min(d_gl // 2, len(self._rope_freqs_loaded))
            inv_g = self._rope_freqs_loaded[:n_freqs]
            if n_freqs < d_gl // 2:
                inv_g_comp = 1.0 / (cfg.rope_theta_global ** (np.arange(n_freqs * 2, d_gl, 2, np.float32) / d_gl))
                inv_g = np.concatenate([inv_g, inv_g_comp])
        else:
            inv_g = 1.0 / (cfg.rope_theta_global ** (np.arange(0, d_gl, 2, np.float32) / d_gl))
        emb_g = np.concatenate([pos * inv_g, pos * inv_g], axis=-1)

        self.ropes = tuple(
            (_cast(tc.tensor(np.cos(e).astype(np.float32))),
             _cast(tc.tensor(np.sin(e).astype(np.float32))))
            for e in (emb_l, emb_g))
        self._rope_len = seq_len
```

---

## 6. BLOCKED Items (Require Runtime Validation)

| # | Item | Why Blocked | Validation Method |
|---|------|-------------|-------------------|
| 1 | **`int4_linear_fused` transpose convention** | C++ kernel source not visible | Run a single `QuantLinearTC` matmul with known input/output shapes. If output shape is wrong, flip `trans_b` in all call sites. |
| 2 | **Router `.scale` application order** | llama.cpp source needed | Run forward with/without scale. If outputs differ meaningfully, try applying scale to gate weights instead of input. |
| 3 | **RoPE frequency override** | `rope_freqs.weight` semantics unclear | Compare computed RoPE against loaded values. If different, test forward with loaded values. |
| 4 | **Re-quantization quality** | Only measurable with generation | Compare output logits against llama.cpp for same input tokens. |
| 5 | **Shared + MoE output combination** | llama.cpp source needed | Verify `shared_out + moe_out` matches llama.cpp's combination (may involve additional norms or gating). |

---

## 7. Updated Static Validation Plan

After applying corrections, run these checks:

### Step 1: Import and Config
```bash
python3 -c "
from core.gemma4_runner import Gemma4Runner, Gemma4Config
cfg = Gemma4Config()
assert cfg.num_layers == 30 and cfg.hidden_dim == 2816
assert cfg.num_experts == 128 and cfg.num_experts_per_tok == 8
assert cfg.num_kv_heads_global == 2 and cfg.num_kv_heads_sliding == 8
print('Config OK')
"
```

### Step 2: Embedding Orientation
```bash
python3 -c "
from core.gemma4_runner import Gemma4Runner
model = Gemma4Runner()
# embedding weight should be [vocab, hidden] = [262144, 2816]
assert model.embed_tokens.weight.shape == (262144, 2816), f'Wrong embed shape: {model.embed_tokens.weight.shape}'
# lm_head should have in=2816, out=262144
assert model.lm_head.in_features == 2816, f'Wrong lm_head in: {model.lm_head.in_features}'
assert model.lm_head.out_features == 262144, f'Wrong lm_head out: {model.lm_head.out_features}'
print('Embedding/lm_head orientation OK')
"
```

### Step 3: GGUF Load (with corrected code)
```bash
python3 -c "
from core.gemma4_runner import Gemma4Runner
import json
model, info = Gemma4Runner.from_pretrained('/path/to/mythos.gguf', qat=True)
print(json.dumps(info, indent=2))
assert info['loaded'] == 658 or info['unmapped'] <= 5, f'Tensor mismatch: {info}'
print('Load OK')
"
```

### Step 4: Expert Slice Shapes (static check)
```bash
python3 -c "
from core.gemma4_runner import Gemma4Runner, Gemma4Config
import tensor_cuda as tc
import numpy as np

cfg = Gemma4Config()
# Simulate fused expert tensors
gue = tc.tensor(np.zeros((2816, 1408, 128), np.float32))
de = tc.tensor(np.zeros((704, 2816, 128), np.float32))

# Test slice + reshape for expert 0
e0_3d = gue.slice(-1, 0, 1)       # (2816, 1408, 1)
e0_2d = e0_3d.reshape([2816, 1408])  # (2816, 1408)
gate_w = e0_2d.slice(1, 0, 704)      # (2816, 704)
up_w = e0_2d.slice(1, 704, 704)      # (2816, 704)

d0_3d = de.slice(-1, 0, 1)           # (704, 2816, 1)
d0_2d = d0_3d.reshape([704, 2816])    # (704, 2816)

print(f'gate_w: {gate_w.shape}')   # Expected: (2816, 704)
print(f'up_w: {up_w.shape}')       # Expected: (2816, 704)
print(f'down_w: {d0_2d.shape}')    # Expected: (704, 2816)

# Test matmul shapes
x = tc.tensor(np.zeros((1, 1, 2816), np.float32))
g_out = tc.matmul(x, gate_w)        # (1, 1, 2816) @ (2816, 704) = (1, 1, 704)
print(f'gate_out: {g_out.shape}')   # Expected: (1, 1, 704)

h = tc.tensor(np.zeros((1, 1, 704), np.float32))
d_out = tc.matmul(h, d0_2d)         # (1, 1, 704) @ (704, 2816) = (1, 1, 2816)
print(f'down_out: {d_out.shape}')   # Expected: (1, 1, 2816)
print('Expert shapes OK')
"
```

---

## 8. Summary

| Severity | Count | Issues |
|----------|-------|--------|
| **CRITICAL** | 3 | Embedding not transposed; `trans_b=True` wrong for GGUF orientation; expert slices need `reshape` to 2D |
| **HIGH** | 1 | Router gate `trans_b` convention (defensive fix provided) |
| **MEDIUM** | 3 | Router scale order; RoPE unused; re-quantization quality |
| **BLOCKED** | 5 | Kernel transpose convention; scale semantics; RoPE semantics; quality; combination math |

**Total corrections needed: ~15 lines across 4 functions.** All are one-line or few-line changes.

**Time to apply corrections: 10 minutes.**  
**Time to re-validate: 10 minutes (Steps 1–4 above).**
