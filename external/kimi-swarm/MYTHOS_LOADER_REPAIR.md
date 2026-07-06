# MYTHOS Loader Mapping Repair — Concrete Patches
## Static Code Review Against GGUF Ground Truth

**Date:** 2026-06-30  
**Scope:** Loader/config mapping only. No runtime, no generation, no context extension.  
**Ground Truth:** `MYTHOS_GGUF_KEY_DUMP.json` — 658 tensors, all names and shapes verified.  
**Target Files:** `gemma4_runner.py`, `model_loader.py`, `kv_manager.py`

---

## 1. Architecture Mismatch Summary

| Parameter | Prototype Assumes | Actual GGUF | Impact |
|-----------|------------------|-------------|--------|
| `num_layers` | 48 | **30** | Loop iterates 18 extra times → `IndexError` on layer access |
| `hidden_dim` | 4608 | **2816** | All projection shapes wrong → `ShapeError` on matmul |
| `num_kv_heads` | 8 (sliding), 1 (global) | **8 (sliding), 2 (global)** | Global KV cache shape `(B,1,S,512)` vs actual `(B,2,S,512)` |
| `num_experts` | 8 | **128** | Router allocates 8 slots, GGUF has 128 → 120 experts silently dropped |
| `experts_per_tok` | 4 | **8** | Router selects 4, should select 8 |
| `expert_ffn_dim` | 15360 | **704** | FFN intermediate wildly wrong |
| `shared_ffn_dim` | Not present | **2112** | Missing shared expert entirely |
| `expert tensor format` | `ffn_gate.{e}.weight` per-expert | **`ffn_gate_up_exps.weight` 3D fused** | `KeyError` — no per-expert keys exist |
| `router shape` | `(8, 4608)` | **`(2816, 128)` + `.scale (2816,)`** | Wrong matmul dims + missing scale |
| `output.weight` | Expected separate | **Tied to `token_embd.weight`** | `KeyError` on `output.weight`, or wrong lm_head |
| `global layer count` | 8 (i%6==5 of 48) | **5** (i%6==5 of 30) | 3 extra global layers expected |
| `extra norms` | None | **`post_ffw_norm_1/2`, `pre_ffw_norm_2`** | Missing norm layers in FFN path |
| `head_dim_sliding` | 256 | **256** | Correct |
| `head_dim_global` | 512 | **512** | Correct |
| `sliding_window` | 1024 | **1024** | Correct |
| `vocab_size` | 262144 | **262144** | Correct |
| `rope_theta_global` | 1000000.0 | **1000000.0** | Correct |
| `rope_theta_swa` | 10000.0 | **10000.0** | Correct |
| `logit_softcap` | 30.0 | **30.0** | Correct |
| `rope_freqs.weight` | Not expected | **Present, shape (256,)** | Extra tensor — load or ignore |
| `ffn_down_exps.scale` | Not expected | **Present, shape (128,)** | Per-expert down-proj scale |

**Verdict: The prototype cannot load the GGUF. Every layer count, every dimension, every tensor name is wrong.**

---

## 2. Repair Plan by File

### 2.1 `gemma4_runner.py` — Four Areas Need Repair

**A. `Gemma4Config` — Replace entire class**
**B. `MoERouter` — Add `.scale` support, fix shape**  
**C. `MoEGeGLUTC` — Rewrite for fused 3D experts + shared expert**  
**D. `Gemma4BlockTC` — Two-branch FFN forward**  
**E. `load_weights_gguf()` — Complete rewrite for actual GGUF keys**  
**F. `Gemma4AttentionTC` — Fix global KV heads from 1 to 2**

### 2.2 `model_loader.py` — Two Areas Need Repair

**A. `load_gemma4_gguf()` — Map all 25 tensor families**  
**B. `load_gguf()` — Handle all 5 quantization types (not just q4_0)**

### 2.3 `kv_manager.py` — Minor Fixes

**A. Hardcoded `global_head_dim` references**  
**B. `GraftAdapter` loop bound uses old layer count**

---

## 3. Concrete Patches

### 3.1 `Gemma4Config` — Complete Replacement

```python
# ==================================================================
# Gemma 4 Config — MYTHOS (Prism Pro DQ)
# ==================================================================
class Gemma4Config:
    """MYTHOS Gemma 4 variant — 30-layer, 128-expert fused MoE.

    Derived from GGUF metadata:
      gemma4.block_count = 30
      gemma4.embedding_length = 2816
      gemma4.feed_forward_length = 2112          # shared expert
      gemma4.expert_feed_forward_length = 704     # MoE expert
      gemma4.expert_count = 128
      gemma4.expert_used_count = 8
      gemma4.attention.head_count = 16
      gemma4.attention.head_count_kv = 2
      gemma4.attention.key_length = 512           # global
      gemma4.attention.value_length = 512         # global
      gemma4.attention.key_length_swa = 256       # sliding
      gemma4.attention.value_length_swa = 256     # sliding
      gemma4.attention.sliding_window = 1024
      gemma4.rope.freq_base = 1000000.0           # global
      gemma4.rope.freq_base_swa = 10000.0         # sliding
      gemma4.rope.dimension_count = 512           # global
      gemma4.rope.dimension_count_swa = 256       # sliding
      gemma4.attention.layer_norm_rms_epsilon = 1e-6
      gemma4.final_logit_softcapping = 30.0
    """
    vocab_size = 262144
    hidden_dim = 2816
    # Attention
    num_heads = 16
    num_kv_heads = 2              # GQA — 2 KV heads (both sliding AND global)
    # NOTE: sliding layers use 8 KV heads in Q/K/V projections,
    #       but the GGUF says head_count_kv=2 globally.
    #       The sliding-layer K/V weights are shape (2816, 2048) = 8 heads × 256 dim.
    #       The global-layer K/V weights are shape (2816, 1024) = 2 heads × 512 dim.
    #       So: sliding has 8 KV heads (2048/256), global has 2 KV heads (1024/512).
    #       The metadata head_count_kv=2 refers to the global config.
    num_kv_heads_sliding = 8      # 2048 / 256 = 8
    num_kv_heads_global = 2       # 1024 / 512 = 2
    head_dim_sliding = 256
    head_dim_global = 512
    sliding_window = 1024
    # RoPE
    rope_theta_global = 1000000.0
    rope_theta_swa = 10000.0
    p_rope_angles = 64            # partial RoPE angles for global
    # Norm
    rms_norm_eps = 1e-6
    logit_softcap = 30.0
    # Tokens
    eos_token_ids = (1,)          # GGUF: tokenizer.ggml.eos_token_id = 1
    bos_token_id = 2              # GGUF: tokenizer.ggml.bos_token_id = 2
    # MoE
    num_experts = 128
    num_experts_per_tok = 8       # top-8 routing
    expert_ffn_dim = 704          # per-expert intermediate
    shared_ffn_dim = 2112         # shared expert intermediate
    # Architecture
    num_layers = 30
    num_sliding_layers = 25       # all except 5, 11, 17, 23, 29
    num_global_layers = 5         # i % 6 == 5

    @staticmethod
    def is_global(i: int) -> bool:
        return i % 6 == 5

    @staticmethod
    def layer_type(i: int) -> str:
        return "global" if Gemma4Config.is_global(i) else "sliding"
```

### 3.2 `Gemma4AttentionTC.__init__` — Fix KV Heads

The attention class already branches on `is_global`. The fix is minimal — change the KV head count assignment:

```python
class Gemma4AttentionTC:
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        self.cfg = cfg
        self.is_global = Gemma4Config.is_global(layer_idx)
        self.head_dim = cfg.head_dim_global if self.is_global else cfg.head_dim_sliding
        # FIX: was cfg.num_global_kv_heads (1) or cfg.num_kv_heads (8)
        # Now uses the correct count per layer type
        self.kv_heads = cfg.num_kv_heads_global if self.is_global else cfg.num_kv_heads_sliding
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.qkv_proj = None
        self.q_norm_w = self.k_norm_w = None
        # APA dials (unchanged)
        self.attention_mode = "standard"
        self.refine_percentile = 0.15
        self.bulk_bits = 4
        self.attn_block = 1024
        self.apa_min_context = 2048
        # Graft injection (unchanged)
        self.inject_kv = None
        self.graft_seats = 0
```

### 3.3 `MoERouter` — Add Scale Support

```python
class MoERouter:
    """Top-k expert router for 128 experts, top-8 selection.

    GGUF layout:
      ffn_gate_inp.weight: (hidden_dim, num_experts) = (2816, 128)
      ffn_gate_inp.scale:  (hidden_dim,) = (2816,)

    The scale is applied elementwise to the hidden state BEFORE the
    gate projection (input scaling from quantization-aware training).
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 8):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = None          # (hidden_dim, num_experts) — set during load
        self.gate_scale = None    # (hidden_dim,) — set during load, may be None

    def __call__(self, x):
        """x: (B, L, hidden_dim) -> (B, L, k) weights, (B, L, k) indices."""
        # Apply input scale if present
        if self.gate_scale is not None:
            x = x * self.gate_scale

        # Gate logits: (B, L, num_experts)
        logits = tc.matmul(x, self.gate, trans_b=False)
        # NOTE: gate shape is (2816, 128), so x @ gate gives (B, L, 128)
        # No trans_b needed — the GGUF stores (in, out) which is (hidden, experts)

        # Top-k selection via numpy
        logits_np = logits.float().numpy()
        B, L, _ = logits_np.shape
        k = min(self.top_k, self.num_experts)
        weights_np = np.zeros((B, L, k), dtype=np.float32)
        indices_np = np.zeros((B, L, k), dtype=np.int32)

        for b in range(B):
            for pos in range(L):
                row = logits_np[b, pos]
                top_idx = np.argpartition(row, -k)[-k:]
                top_vals = row[top_idx]
                top_vals -= np.max(top_vals)
                exp_vals = np.exp(top_vals)
                weights_np[b, pos] = exp_vals / exp_vals.sum()
                indices_np[b, pos] = top_idx

        weights = tc.tensor(np.ascontiguousarray(weights_np), dtype=x.dtype)
        indices = tc.tensor(np.ascontiguousarray(indices_np), dtype="int32")
        return weights, indices
```

### 3.4 `MoEGeGLUTC` — Complete Rewrite (Fused 3D + Shared Expert)

```python
class MoEGeGLUTC:
    """Two-branch FFN: shared dense expert + MoE experts with fused 3D tensors.

    Architecture (per layer):
      x_norm = ffn_norm(x)
      # Shared expert branch (always runs)
      shared_out = GeGLU(x_norm, gate_shared, up_shared, down_shared)
      shared_out = post_ffw_norm_1(shared_out)
      # MoE expert branch (top-8 of 128)
      x_moe = pre_ffw_norm_2(x_norm)
      moe_out = routed_experts(x_moe, top-8)
      moe_out = post_ffw_norm_2(moe_out)
      # Combine
      out = post_ffw_norm(shared_out + moe_out)

    Fused expert tensors (GGUF):
      ffn_gate_up_exps.weight: (2816, 1408, 128) — gate+up for all 128 experts
      ffn_down_exps.weight:    (704, 2816, 128)   — down for all 128 experts
      ffn_down_exps.scale:     (128,)             — per-expert scale

    Splitting: expert e gets:
      gate_e = gate_up_exps[:, 0:704, e]      # (2816, 704)
      up_e   = gate_up_exps[:, 704:1408, e]   # (2816, 704)
      down_e = down_exps[:, :, e]              # (704, 2816)
    """

    def __init__(self, cfg: Gemma4Config):
        self.cfg = cfg
        self.router = MoERouter(cfg.hidden_dim, cfg.num_experts,
                                top_k=cfg.num_experts_per_tok)

        # Shared expert weights (dense, always runs)
        self.gate_shared = None   # QuantLinearTC (2816, 2112)
        self.up_shared = None     # QuantLinearTC (2816, 2112)
        self.down_shared = None   # QuantLinearTC (2112, 2816)

        # Fused MoE expert tensors — stored as full 3D, sliced per-expert at runtime
        self.gate_up_exps = None  # Tensor (2816, 1408, 128)
        self.down_exps = None     # Tensor (704, 2816, 128)
        self.down_exps_scale = None  # Tensor (128,)

        # Norms
        self.ffn_norm = None              # pre-FFN
        self.post_ffw_norm = None         # post-combined
        self.post_ffw_norm_1 = None       # post-shared
        self.pre_ffw_norm_2 = None        # pre-MoE
        self.post_ffw_norm_2 = None       # post-MoE

    def _expert_forward(self, x, eid: int):
        """Run a single expert by slicing the fused 3D tensors.

        x: (B, L, 2816)
        eid: expert index 0..127
        Returns: (B, L, 2816)
        """
        # Slice expert eid from fused tensors
        gate_w = self.gate_up_exps.slice(-1, eid, 1).slice(1, 0, self.cfg.expert_ffn_dim)
        up_w = self.gate_up_exps.slice(-1, eid, 1).slice(1, self.cfg.expert_ffn_dim,
                                                           self.cfg.expert_ffn_dim)
        down_w = self.down_exps.slice(-1, eid, 1)  # (704, 2816, 1)

        # GeGLU: gate(x) * up(x) @ down
        # These are matmuls with the sliced weights
        gate_out = tc.matmul(x, gate_w, trans_b=True)   # (B, L, 704)
        gate_out = gate_out.gelu()
        up_out = tc.matmul(x, up_w, trans_b=True)       # (B, L, 704)
        hidden = gate_out * up_out                       # (B, L, 704)

        # Apply per-expert down scale
        if self.down_exps_scale is not None:
            scale = self.down_exps_scale.slice(0, eid, 1)  # (1,)
            hidden = hidden * scale

        out = tc.matmul(hidden, down_w, trans_b=True)   # (B, L, 2816)
        return out

    def __call__(self, x):
        """Two-branch FFN forward.

        x: (B, L, hidden_dim)
        Returns: (B, L, hidden_dim)
        """
        cfg = self.cfg
        B, L, D = x.shape

        # Pre-FFN norm
        x_norm = self.ffn_norm(x) if self.ffn_norm is not None else x

        # ---- Shared expert branch (always runs) ----
        if self.gate_shared is not None:
            g = self.gate_shared(x_norm).gelu()
            u = self.up_shared(x_norm)
            shared_out = self.down_shared(g * u)
            shared_out = (self.post_ffw_norm_1(shared_out)
                          if self.post_ffw_norm_1 is not None else shared_out)
        else:
            shared_out = tc.zeros_like(x)

        # ---- MoE expert branch (top-8 routed) ----
        if self.gate_up_exps is not None:
            # Pre-norm for MoE path
            x_moe = (self.pre_ffw_norm_2(x_norm)
                     if self.pre_ffw_norm_2 is not None else x_norm)

            # Router scores
            weights, indices = self.router(x_moe)  # weights: (B, L, 8), indices: (B, L, 8)

            # Accumulate expert outputs
            moe_out = tc.zeros_like(x)
            idx_np = indices.numpy().astype(np.int32)    # (B, L, 8)
            w_np = weights.float().numpy()               # (B, L, 8)

            for ki in range(cfg.num_experts_per_tok):
                for eid in range(cfg.num_experts):
                    # Find positions that selected this expert for slot ki
                    mask = (idx_np[:, :, ki] == eid)
                    if not mask.any():
                        continue

                    # Run expert eid on all positions (simpler than gather/scatter)
                    expert_out = self._expert_forward(x_moe, eid)  # (B, L, D)

                    # Apply mask and weight
                    mask_t = tc.tensor(np.ascontiguousarray(mask.astype(np.float32)),
                                       dtype=x.dtype)
                    w_k = weights.slice(-1, ki, 1)  # (B, L, 1)
                    moe_out = moe_out + expert_out * mask_t * w_k

            moe_out = (self.post_ffw_norm_2(moe_out)
                       if self.post_ffw_norm_2 is not None else moe_out)
        else:
            moe_out = tc.zeros_like(x)

        # Combine and final norm
        out = shared_out + moe_out
        out = (self.post_ffw_norm(out)
               if self.post_ffw_norm is not None else out)
        return out
```

### 3.5 `Gemma4BlockTC` — Two-Branch FFN

```python
class Gemma4BlockTC:
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        self.is_global = Gemma4Config.is_global(layer_idx)
        e = cfg.rms_norm_eps

        # Attention norms
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, e)

        # FFN norms — two-branch architecture
        self.ffn_norm = RMSNormTC(cfg.hidden_dim, e)              # pre-FFN (both branches)
        self.post_ffw_norm = RMSNormTC(cfg.hidden_dim, e)         # post-combined
        self.post_ffw_norm_1 = RMSNormTC(cfg.hidden_dim, e)       # post-shared expert
        self.pre_ffw_norm_2 = RMSNormTC(cfg.hidden_dim, e)        # pre-MoE experts
        self.post_ffw_norm_2 = RMSNormTC(cfg.hidden_dim, e)       # post-MoE experts

        self.layer_scalar = 1.0
        self.mixer = Gemma4AttentionTC(cfg, layer_idx)
        self.mlp = MoEGeGLUTC(cfg)

    def __call__(self, x, ropes, position_offset=0, cache=None):
        cos, sin = ropes[1] if self.is_global else ropes[0]

        # Attention path
        a, new_cache = self.mixer(_cast(self.input_layernorm(x)), cos, sin,
                                  position_offset, cache)
        h = x + _cast(self.post_attention_layernorm(a))

        # FFN path — MoEGeGLUTC handles both shared and routed branches
        mo = self.mlp(h)
        h = h + mo

        # Layer output scale
        return h * self.layer_scalar, new_cache
```

### 3.6 `load_weights_gguf()` — Complete Rewrite

```python
    def load_weights_gguf(self, gguf_path: str, progress: bool = True):
        """Load MYTHOS weights from GGUF (mixed Q5_K/Q6_K/Q8_0/Q5_1/F32).

        Ground truth: MYTHOS_GGUF_KEY_DUMP.json — 658 tensors, 25 families.
        Critical: output is tied to token_embd (no separate output.weight).
        """
        from core.model_loader import load_gguf_mythos
        from gguf import GGUFReader
        from gguf.quants import dequantize
        import gc

        cfg = self.config
        reader = GGUFReader(gguf_path)

        # Build tensor lookup by name
        tensors_by_name = {t.name: t for t in reader.tensors}

        # Tracking
        loaded = 0
        skipped = 0
        critical_missing = []
        unmapped = set(tensors_by_name.keys())

        def _deq(name: str) -> np.ndarray:
            """Dequantize a GGUF tensor to float32 numpy."""
            t = tensors_by_name[name]
            return np.asarray(dequantize(t.data, int(t.tensor_type))).astype(np.float32)

        def _load_and_mark(name: str) -> np.ndarray:
            """Load tensor and remove from unmapped set."""
            nonlocal loaded
            if name not in tensors_by_name:
                critical_missing.append(name)
                return None
            arr = _deq(name)
            unmapped.discard(name)
            loaded += 1
            return arr

        # ---- 1. Embeddings (tied output) ----
        emb_arr = _load_and_mark("token_embd.weight")
        if emb_arr is not None:
            # Embeddings: multiply by sqrt(hidden_dim) per Gemma convention
            emb_scaled = emb_arr * np.float32(cfg.hidden_dim ** 0.5)
            self.embed_tokens.weight = np.ascontiguousarray(emb_scaled)

            # Output projection is TIED to embeddings — transpose for (vocab, hidden) -> (hidden, vocab)
            # lm_head expects (out_features, in_features) = (vocab_size, hidden_dim)
            # token_embd is (hidden_dim, vocab_size), so we transpose
            lm_w = np.ascontiguousarray(emb_arr.T)  # (262144, 2816)
            self.lm_head = QuantLinearTC(lm_w, group_size=128)
            del emb_arr, emb_scaled, lm_w
            gc.collect()

        # ---- 2. Output norm ----
        norm_arr = _load_and_mark("output_norm.weight")
        if norm_arr is not None:
            self.norm.weight = tc.tensor(np.ascontiguousarray(norm_arr), dtype="float32")

        # ---- 3. RoPE frequencies (load but don't use yet) ----
        if "rope_freqs.weight" in tensors_by_name:
            rope_arr = _load_and_mark("rope_freqs.weight")
            # BLOCKED: RoPE loading — stored for future use
            # The runner currently computes RoPE from scratch.
            # If rope_freqs differs from computed values, override in extend_rope().
            del rope_arr

        # ---- 4. Per-layer weights ----
        for i in range(cfg.num_layers):
            Lr = self.layers[i]
            g = Lr.mixer.is_global
            b = f"blk.{i}"
            mlp = Lr.mlp

            # ---- 4a. Attention projections ----
            # Q projection
            q_arr = _load_and_mark(f"{b}.attn_q.weight")
            if q_arr is not None:
                Lr.mixer.q_proj = QuantLinearTC(np.ascontiguousarray(q_arr), group_size=128)

            # K projection
            k_arr = _load_and_mark(f"{b}.attn_k.weight")
            if k_arr is not None:
                Lr.mixer.k_proj = QuantLinearTC(np.ascontiguousarray(k_arr), group_size=128)

            # V projection
            v_arr = _load_and_mark(f"{b}.attn_v.weight")
            if v_arr is not None:
                Lr.mixer.v_proj = QuantLinearTC(np.ascontiguousarray(v_arr), group_size=128)

            # O projection
            o_arr = _load_and_mark(f"{b}.attn_output.weight")
            if o_arr is not None:
                Lr.mixer.o_proj = QuantLinearTC(np.ascontiguousarray(o_arr), group_size=128)

            # ---- 4b. Attention head norms ----
            qn_arr = _load_and_mark(f"{b}.attn_q_norm.weight")
            if qn_arr is not None:
                Lr.mixer.q_norm_w = tc.tensor(np.ascontiguousarray(qn_arr), dtype="float32")

            kn_arr = _load_and_mark(f"{b}.attn_k_norm.weight")
            if kn_arr is not None:
                Lr.mixer.k_norm_w = tc.tensor(np.ascontiguousarray(kn_arr), dtype="float32")

            # ---- 4c. Attention norms ----
            an_arr = _load_and_mark(f"{b}.attn_norm.weight")
            if an_arr is not None:
                Lr.input_layernorm.weight = tc.tensor(np.ascontiguousarray(an_arr), dtype="float32")

            pan_arr = _load_and_mark(f"{b}.post_attention_norm.weight")
            if pan_arr is not None:
                Lr.post_attention_layernorm.weight = tc.tensor(np.ascontiguousarray(pan_arr), dtype="float32")

            # ---- 4d. MoE Router ----
            rg_arr = _load_and_mark(f"{b}.ffn_gate_inp.weight")
            if rg_arr is not None:
                # Shape: (2816, 128) — stored as numpy, convert to tensor
                mlp.router.gate = tc.tensor(np.ascontiguousarray(rg_arr), dtype="float32")

            rs_arr = _load_and_mark(f"{b}.ffn_gate_inp.scale")
            if rs_arr is not None:
                mlp.router.gate_scale = tc.tensor(np.ascontiguousarray(rs_arr), dtype="float32")

            # ---- 4e. Shared Expert FFN ----
            sg_arr = _load_and_mark(f"{b}.ffn_gate.weight")
            if sg_arr is not None:
                mlp.gate_shared = QuantLinearTC(np.ascontiguousarray(sg_arr), group_size=128)

            su_arr = _load_and_mark(f"{b}.ffn_up.weight")
            if su_arr is not None:
                mlp.up_shared = QuantLinearTC(np.ascontiguousarray(su_arr), group_size=128)

            sd_arr = _load_and_mark(f"{b}.ffn_down.weight")
            if sd_arr is not None:
                mlp.down_shared = QuantLinearTC(np.ascontiguousarray(sd_arr), group_size=128)

            # ---- 4f. Fused MoE Expert Tensors (3D) ----
            # gate_up_exps: (2816, 1408, 128) — dequantized to float32
            gue_arr = _load_and_mark(f"{b}.ffn_gate_up_exps.weight")
            if gue_arr is not None:
                mlp.gate_up_exps = tc.tensor(np.ascontiguousarray(gue_arr),
                                              dtype=BlockTC.COMPUTE_DTYPE)

            # down_exps: (704, 2816, 128)
            de_arr = _load_and_mark(f"{b}.ffn_down_exps.weight")
            if de_arr is not None:
                mlp.down_exps = tc.tensor(np.ascontiguousarray(de_arr),
                                          dtype=BlockTC.COMPUTE_DTYPE)

            # down_exps scale: (128,)
            des_arr = _load_and_mark(f"{b}.ffn_down_exps.scale")
            if des_arr is not None:
                mlp.down_exps_scale = tc.tensor(np.ascontiguousarray(des_arr), dtype="float32")

            # ---- 4g. FFN Norms ----
            fn_arr = _load_and_mark(f"{b}.ffn_norm.weight")
            if fn_arr is not None:
                mlp.ffn_norm.weight = tc.tensor(np.ascontiguousarray(fn_arr), dtype="float32")

            pfn_arr = _load_and_mark(f"{b}.post_ffw_norm.weight")
            if pfn_arr is not None:
                mlp.post_ffw_norm.weight = tc.tensor(np.ascontiguousarray(pfn_arr), dtype="float32")

            pfn1_arr = _load_and_mark(f"{b}.post_ffw_norm_1.weight")
            if pfn1_arr is not None:
                mlp.post_ffw_norm_1.weight = tc.tensor(np.ascontiguousarray(pfn1_arr), dtype="float32")

            pfn2_arr = _load_and_mark(f"{b}.post_ffw_norm_2.weight")
            if pfn2_arr is not None:
                mlp.post_ffw_norm_2.weight = tc.tensor(np.ascontiguousarray(pfn2_arr), dtype="float32")

            pff2_arr = _load_and_mark(f"{b}.pre_ffw_norm_2.weight")
            if pff2_arr is not None:
                mlp.pre_ffw_norm_2.weight = tc.tensor(np.ascontiguousarray(pff2_arr), dtype="float32")

            # ---- 4h. Layer output scale ----
            los_arr = _load_and_mark(f"{b}.layer_output_scale.weight")
            if los_arr is not None:
                Lr.layer_scalar = float(los_arr[0])

            gc.collect()
            if progress and i % 5 == 4:
                print(f"    layer {i + 1}/{cfg.num_layers}", flush=True)

        # ---- 5. Report ----
        print(f"\n[loader] Loaded: {loaded} tensors")
        if critical_missing:
            print(f"[loader] CRITICAL MISSING ({len(critical_missing)}):")
            for name in critical_missing:
                print(f"  - {name}")
            raise RuntimeError(f"{len(critical_missing)} critical tensors missing from GGUF")
        if unmapped:
            print(f"[loader] Unmapped ({len(unmapped)} tensors — expected for rope_freqs, metadata):")
            for name in sorted(unmapped)[:20]:
                print(f"  - {name}")
            if len(unmapped) > 20:
                print(f"  ... and {len(unmapped) - 20} more")

        return {
            "loaded": loaded,
            "missing": len(critical_missing),
            "unmapped": len(unmapped),
            "framework": f"tensor_cuda MYTHOS ({cfg.num_layers}L, {cfg.hidden_dim}H, {cfg.num_experts}E/{cfg.num_experts_per_tok}A, GQA {cfg.num_kv_heads}KV)",
        }
```

### 3.7 `model_loader.py` — Dequantization Fix

The existing `load_gguf()` only handles types 0 (F32), 1 (F16), and 2 (Q4_0). The actual GGUF uses Q5_K (13), Q6_K (14), Q8_0 (8), and Q5_1 (7). The fix is to use `gguf.quants.dequantize` as a universal fallback for all types:

```python
def load_gguf(
    gguf_path: str,
    weight_callback: Callable[[str, np.ndarray], None],
    q40_callback=None,
    progress: bool = True,
):
    """Iterate over all tensors in a GGUF file, dequantizing to float32.

    Uses gguf.quants.dequantize as universal handler for all quant types.
    """
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(gguf_path)
    for t in r.tensors:
        name = t.name
        tt = int(t.tensor_type)

        if tt == 2 and q40_callback is not None:  # Q4_0 — exact repack path
            data = np.asarray(t.data)
            N = data.shape[0] if data.ndim > 0 else 1
            K = int(np.prod(data.shape[1:])) if data.ndim > 1 else 32
            from core.model_loader import _q40_repack
            packed, scales = _q40_repack(data, N, K)
            q40_callback(name, packed, scales, N, K)
        else:
            # Universal dequantize: F32, F16, Q8_0, Q5_K, Q6_K, Q5_1, etc.
            arr = np.asarray(dequantize(t.data, tt)).astype(np.float32)
            weight_callback(name, arr)
```

### 3.8 `kv_manager.py` — Hardcoded Dim Fixes

Three locations need updating:

**A. `MultiResGlobalKV.__init__` docstring (cosmetic):**
```python
# Change: "D = global_head_dim (512 for Gemma 4)"
# To: "D = global_head_dim (512 for MYTHOS global layers)"
```

**B. `KVManager.init_prefill` — KV head count for sliding layers:**
```python
# In init_prefill(), the sliding-layer K/V shape needs to match
# the actual attention config:
# Sliding: (B, 8, S, 256) — 8 KV heads × 256 dims = 2048
# Global:  (B, 2, S, 512) — 2 KV heads × 512 dims = 1024

# The existing code creates:
#   k = tc.ones(1, 8, 10, 256) for sliding
#   k = tc.ones(1, 1, 10, 512) for global (WAS WRONG — should be 2 not 1)
# Fix: change global to (1, 2, 10, 512)
```

**C. `GraftAdapter.load_graft` loop bound:**
```python
# Change: for i in range(self.cfg.num_layers):  # was 48, now 30
# This is already correct if cfg is updated — no code change needed.
```

### 3.9 `Gemma4Runner.__init__` — Layer Count

```python
def __init__(self, cfg=None):
    self.config = cfg or Gemma4Config()
    cfg = self.config
    self.embed_tokens = HostEmbedding()
    self.layers = [Gemma4BlockTC(cfg, i) for i in range(cfg.num_layers)]  # 30
    self.norm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)  # 2816
    self.lm_head = None
    self._rope_len = 0
    self.extend_rope(4096)
    self.kv_manager = None
```

---

## 4. Full File: `gemma4_runner.py` — Key Sections Only

Below is the complete text for the sections that change. Unchanged sections (KVRing, band mask, _head_rmsnorm, _cublas_blend_attention, forward/generate/sample, from_pretrained) are omitted.

```python
# ==================================================================
# Gemma 4 Config — MYTHOS (Prism Pro DQ)
# ==================================================================
class Gemma4Config:
    vocab_size = 262144
    hidden_dim = 2816
    num_heads = 16
    num_kv_heads_sliding = 8       # 2048 / 256
    num_kv_heads_global = 2        # 1024 / 512
    head_dim_sliding = 256
    head_dim_global = 512
    sliding_window = 1024
    rope_theta_global = 1000000.0
    rope_theta_swa = 10000.0
    p_rope_angles = 64
    rms_norm_eps = 1e-6
    logit_softcap = 30.0
    eos_token_ids = (1,)
    bos_token_id = 2
    num_experts = 128
    num_experts_per_tok = 8
    expert_ffn_dim = 704
    shared_ffn_dim = 2112
    num_layers = 30
    num_sliding_layers = 25
    num_global_layers = 5

    @staticmethod
    def is_global(i: int) -> bool:
        return i % 6 == 5

    @staticmethod
    def layer_type(i: int) -> str:
        return "global" if Gemma4Config.is_global(i) else "sliding"


# ==================================================================
# MoE Router
# ==================================================================
class MoERouter:
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 8):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = None
        self.gate_scale = None

    def __call__(self, x):
        if self.gate_scale is not None:
            x = x * self.gate_scale
        logits = tc.matmul(x, self.gate, trans_b=False)
        logits_np = logits.float().numpy()
        B, L, _ = logits_np.shape
        k = min(self.top_k, self.num_experts)
        weights_np = np.zeros((B, L, k), dtype=np.float32)
        indices_np = np.zeros((B, L, k), dtype=np.int32)
        for b in range(B):
            for pos in range(L):
                row = logits_np[b, pos]
                top_idx = np.argpartition(row, -k)[-k:]
                top_vals = row[top_idx]
                top_vals -= np.max(top_vals)
                exp_vals = np.exp(top_vals)
                weights_np[b, pos] = exp_vals / exp_vals.sum()
                indices_np[b, pos] = top_idx
        weights = tc.tensor(np.ascontiguousarray(weights_np), dtype=x.dtype)
        indices = tc.tensor(np.ascontiguousarray(indices_np), dtype="int32")
        return weights, indices


# ==================================================================
# MoE GeGLU — Fused 3D + Shared Expert
# ==================================================================
class MoEGeGLUTC:
    def __init__(self, cfg: Gemma4Config):
        self.cfg = cfg
        self.router = MoERouter(cfg.hidden_dim, cfg.num_experts,
                                top_k=cfg.num_experts_per_tok)
        self.gate_shared = None
        self.up_shared = None
        self.down_shared = None
        self.gate_up_exps = None
        self.down_exps = None
        self.down_exps_scale = None
        self.ffn_norm = None
        self.post_ffw_norm = None
        self.post_ffw_norm_1 = None
        self.pre_ffw_norm_2 = None
        self.post_ffw_norm_2 = None

    def _expert_forward(self, x, eid: int):
        gate_w = self.gate_up_exps.slice(-1, eid, 1).slice(1, 0, self.cfg.expert_ffn_dim)
        up_w = self.gate_up_exps.slice(-1, eid, 1).slice(1, self.cfg.expert_ffn_dim, self.cfg.expert_ffn_dim)
        down_w = self.down_exps.slice(-1, eid, 1)
        gate_out = tc.matmul(x, gate_w, trans_b=True).gelu()
        up_out = tc.matmul(x, up_w, trans_b=True)
        hidden = gate_out * up_out
        if self.down_exps_scale is not None:
            scale = self.down_exps_scale.slice(0, eid, 1)
            hidden = hidden * scale
        out = tc.matmul(hidden, down_w, trans_b=True)
        return out

    def __call__(self, x):
        cfg = self.cfg
        B, L, D = x.shape
        x_norm = self.ffn_norm(x) if self.ffn_norm is not None else x

        # Shared expert
        if self.gate_shared is not None:
            g = self.gate_shared(x_norm).gelu()
            u = self.up_shared(x_norm)
            shared_out = self.down_shared(g * u)
            shared_out = (self.post_ffw_norm_1(shared_out)
                          if self.post_ffw_norm_1 is not None else shared_out)
        else:
            shared_out = tc.zeros_like(x)

        # MoE experts
        if self.gate_up_exps is not None:
            x_moe = (self.pre_ffw_norm_2(x_norm)
                     if self.pre_ffw_norm_2 is not None else x_norm)
            weights, indices = self.router(x_moe)
            moe_out = tc.zeros_like(x)
            idx_np = indices.numpy().astype(np.int32)
            for ki in range(cfg.num_experts_per_tok):
                for eid in range(cfg.num_experts):
                    mask = (idx_np[:, :, ki] == eid)
                    if not mask.any():
                        continue
                    expert_out = self._expert_forward(x_moe, eid)
                    mask_t = tc.tensor(np.ascontiguousarray(mask.astype(np.float32)), dtype=x.dtype)
                    w_k = weights.slice(-1, ki, 1)
                    moe_out = moe_out + expert_out * mask_t * w_k
            moe_out = (self.post_ffw_norm_2(moe_out)
                       if self.post_ffw_norm_2 is not None else moe_out)
        else:
            moe_out = tc.zeros_like(x)

        out = shared_out + moe_out
        out = (self.post_ffw_norm(out)
               if self.post_ffw_norm is not None else out)
        return out


# ==================================================================
# Attention — Fixed KV Heads
# ==================================================================
class Gemma4AttentionTC:
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        self.cfg = cfg
        self.is_global = Gemma4Config.is_global(layer_idx)
        self.head_dim = cfg.head_dim_global if self.is_global else cfg.head_dim_sliding
        self.kv_heads = cfg.num_kv_heads_global if self.is_global else cfg.num_kv_heads_sliding
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.qkv_proj = None
        self.q_norm_w = self.k_norm_w = None
        self.attention_mode = "standard"
        self.refine_percentile = 0.15
        self.bulk_bits = 4
        self.attn_block = 1024
        self.apa_min_context = 2048
        self.inject_kv = None
        self.graft_seats = 0
    # ... rest of __call__ unchanged ...


# ==================================================================
# Block — Two-Branch FFN
# ==================================================================
class Gemma4BlockTC:
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        self.is_global = Gemma4Config.is_global(layer_idx)
        e = cfg.rms_norm_eps
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.ffn_norm = RMSNormTC(cfg.hidden_dim, e)
        self.post_ffw_norm = RMSNormTC(cfg.hidden_dim, e)
        self.post_ffw_norm_1 = RMSNormTC(cfg.hidden_dim, e)
        self.pre_ffw_norm_2 = RMSNormTC(cfg.hidden_dim, e)
        self.post_ffw_norm_2 = RMSNormTC(cfg.hidden_dim, e)
        self.layer_scalar = 1.0
        self.mixer = Gemma4AttentionTC(cfg, layer_idx)
        self.mlp = MoEGeGLUTC(cfg)

    def __call__(self, x, ropes, position_offset=0, cache=None):
        cos, sin = ropes[1] if self.is_global else ropes[0]
        a, new_cache = self.mixer(_cast(self.input_layernorm(x)), cos, sin,
                                  position_offset, cache)
        h = x + _cast(self.post_attention_layernorm(a))
        mo = self.mlp(h)
        h = h + mo
        return h * self.layer_scalar, new_cache
```

---

## 5. Static Validation Plan

### 5.1 Step 1: Import Test

```bash
cd ~/TensorCUDA
source venv-tc/bin/activate
python3 -c "from core.gemma4_runner import Gemma4Runner, Gemma4Config; print('Import OK')"
```

**Pass:** No import errors.  
**Fail:** Any `ImportError` or `AttributeError`.

### 5.2 Step 2: Config Verification

```python
python3 -c "
from core.gemma4_runner import Gemma4Config
cfg = Gemma4Config()
assert cfg.num_layers == 30, f'Expected 30, got {cfg.num_layers}'
assert cfg.hidden_dim == 2816, f'Expected 2816, got {cfg.hidden_dim}'
assert cfg.num_experts == 128, f'Expected 128, got {cfg.num_experts}'
assert cfg.num_experts_per_tok == 8, f'Expected 8, got {cfg.num_experts_per_tok}'
assert cfg.num_kv_heads_global == 2, f'Expected 2, got {cfg.num_kv_heads_global}'
assert cfg.num_kv_heads_sliding == 8, f'Expected 8, got {cfg.num_kv_heads_sliding}'
assert cfg.expert_ffn_dim == 704, f'Expected 704, got {cfg.expert_ffn_dim}'
assert cfg.shared_ffn_dim == 2112, f'Expected 2112, got {cfg.shared_ffn_dim}'
global_layers = [i for i in range(30) if cfg.is_global(i)]
assert global_layers == [5, 11, 17, 23, 29], f'Wrong global layers: {global_layers}'
print('All config checks passed')
"
```

**Pass:** All assertions pass.  
**Fail:** Any assertion fails.

### 5.3 Step 3: Model Instantiation

```python
python3 -c "
from core.gemma4_runner import Gemma4Runner
model = Gemma4Runner()
cfg = model.config
print(f'Layers: {len(model.layers)}')
print(f'Layer 0 (sliding): global={model.layers[0].mixer.is_global}, kv_heads={model.layers[0].mixer.kv_heads}, head_dim={model.layers[0].mixer.head_dim}')
print(f'Layer 5 (global):  global={model.layers[5].mixer.is_global}, kv_heads={model.layers[5].mixer.kv_heads}, head_dim={model.layers[5].mixer.head_dim}')
print(f'MoE experts: {model.layers[0].mlp.router.num_experts}, top_k={model.layers[0].mlp.router.top_k}')
print(f'Shared FFN dim: {cfg.shared_ffn_dim}')
print(f'Expert FFN dim: {cfg.expert_ffn_dim}')
"
```

**Expected output:**
```
Layers: 30
Layer 0 (sliding): global=False, kv_heads=8, head_dim=256
Layer 5 (global):  global=True, kv_heads=2, head_dim=512
MoE experts: 128, top_k=8
Shared FFN dim: 2112
Expert FFN dim: 704
```

### 5.4 Step 4: GGUF Load (Static Mapping)

```python
python3 -c "
import json
from core.gemma4_runner import Gemma4Runner

model, info = Gemma4Runner.from_pretrained(
    '/home/omen/models/MYTHOS-26B-A4B-PRISM-PRO-DQ-GGUF/mythos-26b-a4b-prism-pro-dq.gguf',
    qat=True
)
print(json.dumps(info, indent=2))

# Verify all layers loaded
for i in range(model.config.num_layers):
    L = model.layers[i]
    assert L.mixer.q_proj is not None, f'Layer {i}: q_proj missing'
    assert L.mixer.k_proj is not None, f'Layer {i}: k_proj missing'
    assert L.mixer.v_proj is not None, f'Layer {i}: v_proj missing'
    assert L.mixer.o_proj is not None, f'Layer {i}: o_proj missing'
    assert L.mlp.gate_shared is not None, f'Layer {i}: shared gate missing'
    assert L.mlp.up_shared is not None, f'Layer {i}: shared up missing'
    assert L.mlp.down_shared is not None, f'Layer {i}: shared down missing'
    assert L.mlp.gate_up_exps is not None, f'Layer {i}: fused experts missing'
    assert L.mlp.down_exps is not None, f'Layer {i}: down_exps missing'
    assert L.mlp.router.gate is not None, f'Layer {i}: router gate missing'

# Verify tied output
assert model.lm_head is not None, 'lm_head missing'
print(f'lm_head in_features: {model.lm_head.in_features}')
print(f'lm_head out_features: {model.lm_head.out_features}')
assert model.lm_head.in_features == 2816, 'lm_head in_features wrong'
assert model.lm_head.out_features == 262144, 'lm_head out_features wrong'

print('All loading checks passed')
"
```

**Pass:** `info["loaded"]` equals 658 (or close), no assertion failures, `lm_head` has correct shape.  
**Fail:** Any `KeyError`, `AssertionError`, or shape mismatch.

### 5.5 Step 5: Tensor Family Coverage

```python
python3 << 'PYEOF'
from gguf import GGUFReader
import sys

r = GGUFReader('/home/omen/models/MYTHOS-26B-A4B-PRISM-PRO-DQ-GGUF/mythos-26b-a4b-prism-pro-dq.gguf')
names = [t.name for t in r.tensors]

# Expected patterns
expected_patterns = [
    "token_embd.weight",
    "output_norm.weight",
    "rope_freqs.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "attn_norm.weight", "post_attention_norm.weight",
    "ffn_gate_inp.weight", "ffn_gate_inp.scale",
    "ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_down_exps.scale",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_norm.weight", "post_ffw_norm.weight",
    "post_ffw_norm_1.weight", "post_ffw_norm_2.weight", "pre_ffw_norm_2.weight",
    "layer_output_scale.weight",
]

missing = []
for pattern in expected_patterns:
    found = any(pattern in n for n in names)
    if not found:
        missing.append(pattern)

if missing:
    print(f"MISSING PATTERNS ({len(missing)}):")
    for p in missing:
        print(f"  - {p}")
    sys.exit(1)
else:
    print(f"All {len(expected_patterns)} tensor families found in GGUF")
PYEOF
```

**Pass:** All 20 patterns found.  
**Fail:** Any pattern missing.

---

## 6. Remaining Blockers (Post-Repair)

| # | Blocker | Why | Resolution Path |
|---|---------|-----|----------------|
| 1 | **Forward pass not tested** | I cannot execute code | Run Step 4 validation locally |
| 2 | **RoPE frequency mismatch** | `rope_freqs.weight` in GGUF vs computed RoPE | If forward produces garbage, compare `rope_freqs.weight` against computed values and override |
| 3 | **Dequantization correctness** | Q5_K, Q6_K, Q8_0, Q5_1 dequant via `gguf.quants` | If outputs are wrong, verify `dequantize()` produces correct float32 values — compare against llama.cpp's output for same input |
| 4 | **Fused expert slicing dim order** | `(2816, 1408, 128)` — gate at `[0:704,:]` or `[:,0:704,:]`? | If expert outputs are wrong, try flipping the slice dims in `_expert_forward()` |
| 5 | **Router scale application order** | `.scale` before or after gate matmul? | If routing is wrong, try applying scale after matmul instead of before |
| 6 | **Shared + MoE combination** | `shared_out + moe_out` — correct operation? | Compare against llama.cpp source for Gemma 4 MoE FFN combination |
| 7 | **KVManager type mismatch** | `MultiResGlobalKV` not compatible with `KVRing` | Separate mission — wire KVManager into generation |
| 8 | **Server streaming untested** | No live model to stream against | Separate mission — test after forward pass works |
| 9 | **Graft mounting unwired** | `inject_kv` set manually, not by `generate()` | Separate mission — wire graft adapter into generation loop |
| 10 | **GGUF dequant memory** | Dequantizing 18GB GGUF to float32 needs ~70GB RAM | The loader dequantizes one tensor at a time; if OOM, add `gc.collect()` between layers and use `del` aggressively |

---

## 7. Summary of Changes

| File | Lines Changed | What |
|------|--------------|------|
| `gemma4_runner.py` | ~150 | Config (all constants), Attention (KV heads), MoERouter (scale), MoEGeGLUTC (fused 3D + shared), BlockTC (2-branch norms), load_weights_gguf (all 20 tensor families) |
| `model_loader.py` | ~20 | `load_gguf()` — universal dequantize fallback for all quant types |
| `kv_manager.py` | ~5 | Global KV head count in `init_prefill()`, docstrings |

**Total estimated patch size: ~175 lines across 3 files.**

**Time to apply: 30 minutes.**  
**Time to validate: 15 minutes (Steps 1–5 above).**  
**Time to first forward pass: 1–2 hours (if Step 4 passes).**
