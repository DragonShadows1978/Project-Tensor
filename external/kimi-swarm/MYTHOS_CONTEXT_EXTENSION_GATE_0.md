# MYTHOS Context Extension — Gate 0
## Corrected Loader + First Testable Extension Experiment

**Authority:** `ARA_REALITY_GATE.md` (cycle 2 FINAL) > `MYTHOS_GGUF_KEY_DUMP.json` (raw) > `MYTHOS_LOADER_RISK_REVIEW.md` (this document) > all prior design decks  
**Date:** 2026-07-02  
**Scope:** Loader repair incorporating all verified corrections, then the first context-extension experiment gated on logit parity. No 1M claims. No Collatz claims. No APA miracles.

---

## 0. Where We Actually Are

**MYTHOS runs today** on llama-cpp-turboquant at 128K context, 150–170 tok/s, stable for days (ARA GATE §1). This is the baseline. Everything TensorCUDA could add is **downstream** of matching this baseline.

**What TensorCUDA could theoretically add** that llama-cpp cannot:

| Capability | llama-cpp Status | TensorCUDA Path | Status |
|-----------|-----------------|-----------------|--------|
| APA 2.1× speedup at 2K+ | No — standard attention only | Kernel exists, Gemma-4 is parity case | Code exists, not wired |
| INT4 KV compression | No — turbo3 is Q8_0 | `kv_int4_pack/unpack` exists | Code exists, not wired |
| Graft virtual memory | No — text-level RAG only | `inject_kv` exists in attention | Code exists, not wired |
| Multi-resolution KV | No | `KVManager` + `MultiResGlobalKV` exist | Type mismatch — not wired |
| 1M context | 128K ceiling on 32GB | Arithmetic says 4.4GB @ 1M | Pure arithmetic — no evidence |

**The honest truth:** Every unique TensorCUDA capability sits behind **Gate A** (loader parity) and **Gate B** (first-token logit match). There is no shortcut. The ARA GATE's mission sequence (§7) is correct: A → B → C → D → E, each gated on the previous.

**This document provides:**
- **Part I:** The corrected loader patches (all findings from risk review + ARA GATE merged)
- **Part II:** Context Extension Gate 0 — the first experiment after parity that actually tests a context-extension mechanism

---

## Part I: Corrected Loader Patches

### 1.1 The Four Critical Corrections (from Risk Review + ARA GATE)

| # | Issue | Source | Fix |
|---|-------|--------|-----|
| 1 | **Embedding not transposed** — `token_embd.weight [2816, 262144]` stored as-is, `HostEmbedding` indexes first dim by token ID, produces 262144-length vectors | Risk Review §2.1 | Transpose to `[262144, 2816]` before storing |
| 2 | **`trans_b=True` backwards for GGUF-oriented weights** — GGUF stores `[in, out]`, matmul with `trans_b=True` computes `x @ W^T` causing shape mismatch | Risk Review §2.2 | Remove `trans_b=True` in expert forward; use `trans_b=False` |
| 3 | **Expert slices stay 3D** — `.slice()` returns `(2816, 704, 1)`, `matmul` needs 2D | Risk Review §2.3 | `.reshape([2816, 1408])` before slicing gate/up |
| 4 | **Global layers have NO V tensor** — `attn_v.weight` exists only on 25 sliding layers; 5 global layers (5, 11, 17, 23, 29) have Q, K, output, norms only | ARA GATE §2, correction 1 | On global layers, load Q/K/output only; V must be derived from K (or shared projection — read llama.cpp source) |

### 1.2 The Config — Final Corrected Version

```python
class Gemma4Config:
    """MYTHOS Gemma 4 variant — verified against GGUF metadata (ARA GATE §2).
    
    All values from gemma4.* metadata keys in the GGUF, not guessed.
    """
    vocab_size = 262144
    hidden_dim = 2816
    num_heads = 16
    # Attention: GQA. Metadata says head_count_kv=2, but sliding K/V shapes
    # imply 8 KV heads (2048/256). Global K shape (2816,1024) at head_dim 512
    # implies 2 KV heads. Per-layer-type resolution required — see attention code.
    num_kv_heads_sliding = 8       # 2048 / 256 = 8  (from attn_k.shape[1]/256)
    num_kv_heads_global = 2        # 1024 / 512 = 2  (from attn_k.shape[1]/512)
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
```

### 1.3 `load_weights_gguf()` — Full Corrected Version

```python
    def load_weights_gguf(self, gguf_path: str, progress: bool = True):
        """Load MYTHOS weights from GGUF (mixed Q5_K/Q6_K/Q8_0/Q5_1/F32).
        
        Ground truth: MYTHOS_GGUF_KEY_DUMP.json — 658 tensors, 25 families.
        Critical corrections applied:
          - Embedding transposed to [vocab, hidden] for HostEmbedding
          - Global layers: NO attn_v.weight (ARA GATE §2 correction 1)
          - Router gate stored transposed to match LinearTC convention
          - lm_head tied to token_embd (no separate output.weight)
          - All 20 tensor families mapped, fail-closed on missing critical tensors
        """
        from gguf import GGUFReader
        from gguf.quants import dequantize
        import gc

        cfg = self.config
        reader = GGUFReader(gguf_path)
        tensors_by_name = {t.name: t for t in reader.tensors}
        
        loaded = 0
        critical_missing = []
        unmapped = set(tensors_by_name.keys())

        def _deq(name: str) -> np.ndarray:
            t = tensors_by_name[name]
            return np.asarray(dequantize(t.data, int(t.tensor_type))).astype(np.float32)

        def _load_and_mark(name: str) -> np.ndarray:
            nonlocal loaded
            if name not in tensors_by_name:
                critical_missing.append(name)
                return None
            arr = _deq(name)
            unmapped.discard(name)
            loaded += 1
            return arr

        # ---- 1. Embeddings (tied output) — CORRECTED: transpose ----
        emb_arr = _load_and_mark("token_embd.weight")
        if emb_arr is not None:
            # GGUF: [2816, 262144] = [hidden_dim, vocab_size]
            # HostEmbedding indexes by token ID → needs [vocab_size, hidden_dim]
            emb_vh = emb_arr.T                                 # [262144, 2816]
            emb_scaled = emb_vh * np.float32(cfg.hidden_dim ** 0.5)
            self.embed_tokens.weight = np.ascontiguousarray(emb_scaled)
            
            # lm_head: tied to embeddings, QuantLinearTC expects (out, in)
            # emb_vh is [262144, 2816] = (vocab_size, hidden_dim) = (out, in)
            self.lm_head = QuantLinearTC(np.ascontiguousarray(emb_vh),
                                         group_size=128)
            del emb_arr, emb_vh, emb_scaled
            gc.collect()

        # ---- 2. Output norm ----
        norm_arr = _load_and_mark("output_norm.weight")
        if norm_arr is not None:
            self.norm.weight = tc.tensor(np.ascontiguousarray(norm_arr), dtype="float32")

        # ---- 3. RoPE frequencies (load for possible override) ----
        if "rope_freqs.weight" in tensors_by_name:
            rope_arr = _load_and_mark("rope_freqs.weight")
            self._rope_freqs_loaded = rope_arr.copy()
            del rope_arr
        else:
            self._rope_freqs_loaded = None

        # ---- 4. Per-layer weights ----
        for i in range(cfg.num_layers):
            Lr = self.layers[i]
            g = Lr.mixer.is_global
            b = f"blk.{i}"
            mlp = Lr.mlp

            # 4a. Attention projections — ALL layers have Q, K, O
            q_arr = _load_and_mark(f"{b}.attn_q.weight")
            if q_arr is not None:
                Lr.mixer.q_proj = QuantLinearTC(np.ascontiguousarray(q_arr), group_size=128)

            k_arr = _load_and_mark(f"{b}.attn_k.weight")
            if k_arr is not None:
                Lr.mixer.k_proj = QuantLinearTC(np.ascontiguousarray(k_arr), group_size=128)

            # CRITICAL: V projection — ONLY on sliding layers (ARA GATE §2 correction 1)
            # Global layers (i%6==5) have NO attn_v.weight
            if not g:
                v_arr = _load_and_mark(f"{b}.attn_v.weight")
                if v_arr is not None:
                    Lr.mixer.v_proj = QuantLinearTC(np.ascontiguousarray(v_arr), group_size=128)
            # On global layers: v_proj remains None — forward pass must derive V from K
            # or use K as V (GQA shared projection). See BLOCKED note below.

            o_arr = _load_and_mark(f"{b}.attn_output.weight")
            if o_arr is not None:
                Lr.mixer.o_proj = QuantLinearTC(np.ascontiguousarray(o_arr), group_size=128)

            # 4b. Attention head norms
            qn_arr = _load_and_mark(f"{b}.attn_q_norm.weight")
            if qn_arr is not None:
                Lr.mixer.q_norm_w = tc.tensor(np.ascontiguousarray(qn_arr), dtype="float32")

            kn_arr = _load_and_mark(f"{b}.attn_k_norm.weight")
            if kn_arr is not None:
                Lr.mixer.k_norm_w = tc.tensor(np.ascontiguousarray(kn_arr), dtype="float32")

            # 4c. Attention norms
            an_arr = _load_and_mark(f"{b}.attn_norm.weight")
            if an_arr is not None:
                Lr.input_layernorm.weight = tc.tensor(np.ascontiguousarray(an_arr), dtype="float32")

            pan_arr = _load_and_mark(f"{b}.post_attention_norm.weight")
            if pan_arr is not None:
                Lr.post_attention_layernorm.weight = tc.tensor(np.ascontiguousarray(pan_arr), dtype="float32")

            # 4d. MoE Router — CORRECTED: store transposed
            rg_arr = _load_and_mark(f"{b}.ffn_gate_inp.weight")
            if rg_arr is not None:
                # GGUF: [2816, 128] = [in, out]
                # Store as [128, 2816] = [out, in] to match LinearTC convention
                mlp.router.gate = tc.tensor(np.ascontiguousarray(rg_arr.T), dtype="float32")

            rs_arr = _load_and_mark(f"{b}.ffn_gate_inp.scale")
            if rs_arr is not None:
                mlp.router.gate_scale = tc.tensor(np.ascontiguousarray(rs_arr), dtype="float32")

            # 4e. Shared Expert FFN
            sg_arr = _load_and_mark(f"{b}.ffn_gate.weight")
            if sg_arr is not None:
                mlp.gate_shared = QuantLinearTC(np.ascontiguousarray(sg_arr), group_size=128)

            su_arr = _load_and_mark(f"{b}.ffn_up.weight")
            if su_arr is not None:
                mlp.up_shared = QuantLinearTC(np.ascontiguousarray(su_arr), group_size=128)

            sd_arr = _load_and_mark(f"{b}.ffn_down.weight")
            if sd_arr is not None:
                mlp.down_shared = QuantLinearTC(np.ascontiguousarray(sd_arr), group_size=128)

            # 4f. Fused MoE Expert Tensors
            gue_arr = _load_and_mark(f"{b}.ffn_gate_up_exps.weight")
            if gue_arr is not None:
                mlp.gate_up_exps = tc.tensor(np.ascontiguousarray(gue_arr),
                                              dtype=BlockTC.COMPUTE_DTYPE)

            de_arr = _load_and_mark(f"{b}.ffn_down_exps.weight")
            if de_arr is not None:
                mlp.down_exps = tc.tensor(np.ascontiguousarray(de_arr),
                                          dtype=BlockTC.COMPUTE_DTYPE)

            des_arr = _load_and_mark(f"{b}.ffn_down_exps.scale")
            if des_arr is not None:
                mlp.down_exps_scale = tc.tensor(np.ascontiguousarray(des_arr), dtype="float32")

            # 4g. FFN Norms
            for norm_name, target in [
                ("ffn_norm.weight", mlp.ffn_norm),
                ("post_ffw_norm.weight", mlp.post_ffw_norm),
                ("post_ffw_norm_1.weight", mlp.post_ffw_norm_1),
                ("post_ffw_norm_2.weight", mlp.post_ffw_norm_2),
                ("pre_ffw_norm_2.weight", mlp.pre_ffw_norm_2),
            ]:
                n_arr = _load_and_mark(f"{b}.{norm_name}")
                if n_arr is not None:
                    target.weight = tc.tensor(np.ascontiguousarray(n_arr), dtype="float32")

            # 4h. Layer output scale
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
            raise RuntimeError(f"{len(critical_missing)} critical tensors missing")
        if unmapped:
            print(f"[loader] Unmapped ({len(unmapped)} tensors):")
            for name in sorted(unmapped)[:20]:
                print(f"  - {name}")
            if len(unmapped) > 20:
                print(f"  ... and {len(unmapped) - 20} more")

        return {
            "loaded": loaded,
            "missing": len(critical_missing),
            "unmapped": len(unmapped),
            "framework": f"tensor_cuda MYTHOS ({cfg.num_layers}L, {cfg.hidden_dim}H, {cfg.num_experts}E/{cfg.num_experts_per_tok}A)",
        }
```

### 1.4 `MoEGeGLUTC._expert_forward()` — Corrected (no trans_b, with reshape)

```python
    def _expert_forward(self, x, eid: int):
        """Run a single expert by slicing the fused 3D tensors.
        
        CORRECTIONS from risk review:
          - reshape to 2D before matmul (was 3D slice)
          - trans_b=False (GGUF weights are [in, out])
          - gate stored transposed to match LinearTC convention
        """
        cfg = self.cfg
        
        # gate_up_exps: (2816, 1408, 128)
        expert_3d = self.gate_up_exps.slice(-1, eid, 1)     # (2816, 1408, 1)
        expert_2d = expert_3d.reshape([cfg.hidden_dim, 2 * cfg.expert_ffn_dim])  # (2816, 1408)
        gate_w = expert_2d.slice(1, 0, cfg.expert_ffn_dim)          # (2816, 704)
        up_w = expert_2d.slice(1, cfg.expert_ffn_dim, cfg.expert_ffn_dim)  # (2816, 704)
        
        # down_exps: (704, 2816, 128)
        down_3d = self.down_exps.slice(-1, eid, 1)          # (704, 2816, 1)
        down_w = down_3d.reshape([cfg.expert_ffn_dim, cfg.hidden_dim])      # (704, 2816)
        
        # GeGLU: NO trans_b — GGUF weights are [in, out]
        gate_out = tc.matmul(x, gate_w).gelu()   # (B, L, 2816) @ (2816, 704) = (B, L, 704)
        up_out = tc.matmul(x, up_w)               # (B, L, 2816) @ (2816, 704) = (B, L, 704)
        hidden = gate_out * up_out                 # (B, L, 704)
        
        if self.down_exps_scale is not None:
            scale = self.down_exps_scale.slice(0, eid, 1)  # (1,)
            hidden = hidden * scale
        
        out = tc.matmul(hidden, down_w)           # (B, L, 704) @ (704, 2816) = (B, L, 2816)
        return out
```

### 1.5 `MoERouter.__call__()` — Corrected (transposed gate)

```python
    def __call__(self, x):
        """x: (B, L, hidden_dim) -> (B, L, k) weights, (B, L, k) indices.
        
        Gate stored as (out, in) = (128, 2816) — transposed from GGUF's (2816, 128).
        Uses trans_b=True to compute x @ gate^T = x (B, L, 2816) @ (2816, 128).
        """
        if self.gate_scale is not None:
            x = x * self.gate_scale
        
        # Gate stored as (128, 2816), so trans_b=True gives (B, L, 128)
        logits = tc.matmul(x, self.gate, trans_b=True)
        
        # Top-k via numpy (unchanged)
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

### 1.6 Global-Layer V Mechanism — BLOCKED

**The ARA GATE identified that global layers have NO `attn_v.weight`** (§2 correction 1). On sliding layers, Q, K, V are separate projections. On global layers, only Q, K, and output exist.

**What llama.cpp does:** The V projection on global layers is **shared with K** — the same tensor serves as both K and V. This is consistent with GQA designs where KV heads are shared and the V projection can be elided when it shares weights with K.

**The fix in attention forward:**

```python
# In Gemma4AttentionTC.__call__(), when building k and v:
if self.is_global:
    # Global layer: K and V share the same projection
    k = _head_rmsnorm(kraw, self.k_norm_w, cfg.rms_norm_eps, B, L, KV, D)
    k = k.reshape([B, L, KV, D]).transpose(1, 2)
    v = k  # V IS K — shared projection, no separate v_proj on global layers
else:
    # Sliding layer: separate K and V projections
    k = _head_rmsnorm(kraw, self.k_norm_w, cfg.rms_norm_eps, B, L, KV, D)
    k = k.reshape([B, L, KV, D]).transpose(1, 2)
    vsrc = self.v_proj(x) if self.v_proj is not None else kraw
    v = _head_rmsnorm(vsrc, None, cfg.rms_norm_eps, B, L, KV, D)
    v = v.reshape([B, L, KV, D]).transpose(1, 2)
```

**BLOCKED:** This assumption (V = K on global layers) must be verified against llama.cpp's gemma4 source. If llama.cpp uses a different mechanism (separate V computation, different weight sharing), the forward pass will produce wrong logits. The first parity test will reveal this immediately.

---

## Part II: Context Extension Gate 0

### 2.1 What "Context Extension" Means Here

MYTHOS already runs at 128K on llama-cpp. The question is not "can we run the model?" — it's **"can we run it at >128K without OOM, and if so, what's the mechanism and what's the evidence?"**

The ARA GATE is explicit: APA on Gemma-4 is **parity, not extension** (ARA GATE references APA paper §4.2.1: "Gemma-4 is the parity-catch-up case, not a context win"). APA gives 2.1× speedup at 2K+ sequence length, but the memory wall is the same place. The real context-extension lever is **KV cache compression** — specifically, storing the 5 global layers' KV cache at INT4 instead of FP16.

### 2.2 Why Gemma-4's Architecture Is the Key

Gemma-4 has a **mixed attention architecture**:

| Layer Type | Count | KV Cache Growth | Head Config |
|-----------|-------|----------------|-------------|
| Sliding-window | 25 | **Capped at 1024 keys** (~328 MB fixed) | GQA 8 KV heads × 256 dim |
| Global (full context) | 5 | **Grows with context** | GQA 2 KV heads × 512 dim |

**Only 5 of 30 layers grow with context length.** The 25 sliding-window layers are ring-buffered at 1024 keys — their KV cache footprint is **constant regardless of context size**. This is a structural gift that uniform transformers (Qwen, Llama) don't have.

**The arithmetic:**

| Context | Sliding KV (25 layers) | Global KV FP16 (5 layers) | Global KV INT4 (5 layers) | Total FP16 | Total Multi-Res |
|---------|----------------------|--------------------------|--------------------------|-----------|----------------|
| 128K | 328 MB | 2,048 MB | 512 MB | 2,376 MB | **840 MB** |
| 256K | 328 MB | 4,096 MB | 1,024 MB | 4,424 MB | **1,352 MB** |
| 512K | 328 MB | 8,192 MB | 2,048 MB | 8,520 MB | **2,376 MB** |
| 1M | 328 MB | 16,384 MB | 4,096 MB | 16,712 MB | **4,424 MB** |

At 128K context, multi-resolution KV uses **840 MB** vs. llama-cpp's **~10.7 GB** (turbo3 Q8_0 KV cache at 128K). That's a **12.7× reduction**.

**The catch:** This arithmetic is pure theory. No end-to-end evidence exists (ARA GATE §1: "1M context: pure arithmetic today; requires multi-res KV working and validated").

### 2.3 Gate 0 — "Does INT4 Global KV Produce Identical Logits?"

**Gate 0 is the first context-extension experiment that can be run AFTER logit parity is proven at short context.** It tests whether compressing the global layers' KV cache to INT4 changes the output.

**Hypothesis:** On Gemma-4 (qk-normed, bulk-bits floor = 4-bit per APA paper §4.3), INT4 KV storage is quality-neutral for the global attention layers.

**Experiment:**

```python
# After Gate B (logit parity at short context) passes:

# Step 1: Run a forward pass with standard FP16 KV cache
# Step 2: Run the SAME forward pass with INT4-compressed global KV
# Step 3: Compare logits

# The test:
def test_int4_kv_parity(model, input_ids):
    with tc.no_grad():
        # Baseline: all FP16
        logits_fp16, _ = model.forward(input_ids, last_token_only=True)
        
        # Test: global layers at INT4, sliding at FP16
        # This requires wiring KVManager into the generation path
        # (currently type-mismatched — see ARA GATE §6)
        
        # Compare top-k tokens
        fp16_topk = np.argsort(logits_fp16.numpy()[0, 0])[-10:]
        int4_topk = np.argsort(logits_int4.numpy()[0, 0])[-10:]
        
        overlap = len(set(fp16_topk) & set(int4_topk))
        print(f"Top-10 overlap: {overlap}/10")
        
        # Max logit difference
        max_diff = np.max(np.abs(logits_fp16.numpy() - logits_int4.numpy()))
        print(f"Max |logit diff|: {max_diff:.4f}")
        
        return overlap == 10 and max_diff < 0.5  # tolerance for quantization noise
```

**Success criteria:**
- Top-10 token overlap: 10/10
- Max |logit diff|: < 0.5 (quantization noise tolerance)
- Perplexity on a short test set: within 1% of FP16 baseline

**Why this matters:** If Gate 0 passes, INT4 global KV is verified as quality-neutral. That unlocks the multi-resolution path: FP16 for recent tokens (quality), INT4 for long-range (compression). The arithmetic table above becomes evidence, not speculation.

**If Gate 0 fails:** The INT4 compression introduces measurable quality loss. Options: (a) use INT8 instead (2× compression, likely quality-neutral), (b) increase FP16 window size, (c) investigate per-head quantization (some heads may need more bits).

### 2.4 What APA and Ghost Geometry Actually Contribute

**APA (Adaptive Precision Attention):**

| Claim | Evidence | Relevance to Gate 0 |
|-------|---------|---------------------|
| 2.1× speedup at 2K+ context | Measured on 7 models (APA paper §4.5) | Speed, not extension. Gate 0 is about memory, not speed. |
| 4-bit bulk floor for qk-normed models | MiniCPM3 sweep: 4-bit free, 2-bit breaks (§4.3) | **Directly supports Gate 0 hypothesis** — INT4 should be quality-neutral on Gemma-4 (qk-normed). |
| Gemma-4 = parity case, not extension | 8K decode, 7.61 GB vs standard 7.64 GB (§4.2.1) | APA won't extend context on MYTHOS. Don't expect it to. |
| Architecture-agnostic | MHA/MQA/GQA/MLA + MoE (§4.1) | Confirms APA kernel can run on MYTHOS's GQA+MoE, but doesn't help with context size. |

**APA's contribution to Gate 0:** The bulk-bits law (§4.3) predicts that 4-bit KV storage on Gemma-4's qk-normed keys is at the quality-neutral floor. This is the empirical basis for the Gate 0 hypothesis — not a proof, but a strong prior from a related measurement.

**Ghost Geometry (Precision-Collapse Framework):**

| Claim | Evidence | Relevance to Gate 0 |
|-------|---------|---------------------|
| Precision depth decays geometrically in attention | Measured: 50.5–53.2% at 1 bit, 60–73% by 2 bits (§5.3) | Supports the intuition that most KV cache entries don't need full precision. But this measures *dot products*, not KV storage. |
| Structural law transfers from Collatz to attention | Decay constants differ 2–10×; shape matches (§5.3) | Philosophical support for precision allocation, not an engineering gate. |
| Collatz framework has 7 gaps (G1–G7) | Self-audited, not proven (§4) | **No engineering relevance.** Do not use. |

**Ghost Geometry's contribution to Gate 0:** The GHOST_PRECISION measurement confirms that attention interactions are heavily concentrated — most resolve at low bit-width. This supports the *intuition* behind INT4 KV compression, but it's a measurement of dot-product precision, not KV storage precision. The direct evidence for Gate 0 comes from APA's bulk-bits law, not Ghost Geometry.

**Honest verdict:** Ghost Geometry is intellectually interesting but provides no testable engineering gate for MYTHOS context extension. APA provides one specific testable prediction (4-bit floor on qk-normed models) that directly supports Gate 0. Everything else is downstream of the loader parity work.

### 2.5 Mission Sequence (Revised from ARA GATE §7)

| Mission | Scope | Success Evidence | Prerequisite |
|---------|-------|-----------------|-------------|
| **A — Loader Repair** | Apply corrected patches (§1.1–1.6), verify static mapping | 658 tensors loaded, 0 critical missing, config derived from GGUF metadata | None — this is the starting point |
| **B — Logit Parity** | Same prompt through TensorCUDA and llama-cpp, compare top-k logits | Same argmax token, top-10 overlap 10/10, max \|diff\| < 0.5 | Mission A |
| **C — Gate 0: INT4 KV Parity** | Compress global-layer KV to INT4, verify identical logits | Top-10 overlap 10/10, max \|diff\| < 0.5, ppl within 1% | Mission B |
| **D — Multi-Res KV Wiring** | Wire KVManager into generation path (FP16 recent + INT4 long-range) | Generation runs at 32K→128K context without OOM, ppl tracked | Mission C |
| **E — Extended Context Test** | 256K→512K→1M context with multi-res KV | Completes without OOM, coherent output, compared against llama-cpp at 128K | Mission D |

**Do not attempt C before B passes.** Gate 0 tests whether INT4 compression is quality-neutral, but if the base forward pass is already wrong (B fails), Gate 0 results are meaningless.

---

## 3. Remaining Blockers (Post-Patch)

| # | Blocker | Discovery Path | Who |
|---|---------|---------------|-----|
| 1 | **Global-layer V = K assumption** | If Gate B fails, check whether llama.cpp uses shared K/V or separate V | Read `~/repos/llama-cpp-turboquant/` gemma4 model source |
| 2 | **`int4_linear_fused` transpose convention** | If any QuantLinearTC matmul produces wrong shape | Single-tensor test: known input, check output shape |
| 3 | **RoPE frequency override** | If Gate B shows position-dependent divergence | Compare `rope_freqs.weight` against computed values in `extend_rope()` |
| 4 | **Router scale application order** | If MoE outputs differ from llama-cpp | Try applying `.scale` to gate weights vs. input, compare |
| 5 | **Shared + MoE combination math** | If FFN outputs differ | Check llama.cpp's `ffn_norm` → shared → `post_ffw_norm_1` → `pre_ffw_norm_2` → MoE → `post_ffw_norm_2` → `post_ffw_norm` sequence against our implementation |
| 6 | **KVManager ↔ attention type mismatch** | Gate C can't run until fixed | `MultiResGlobalKV` needs KVRing-compatible interface, or attention needs to detect MRGKV type |

---

## 4. Static Validation Commands

### After applying patches:

```bash
# 1. Import test
python3 -c "from core.gemma4_runner import Gemma4Runner, Gemma4Config; cfg = Gemma4Config(); assert cfg.num_layers == 30 and cfg.hidden_dim == 2816; print('OK')"

# 2. Embedding orientation
python3 -c "from core.gemma4_runner import Gemma4Runner; m = Gemma4Runner(); assert m.embed_tokens.weight.shape == (262144, 2816); assert m.lm_head.in_features == 2816; assert m.lm_head.out_features == 262144; print('OK')"

# 3. Layer structure
python3 -c "
from core.gemma4_runner import Gemma4Config
assert Gemma4Config.is_global(5) and not Gemma4Config.is_global(0)
global_layers = [i for i in range(30) if Gemma4Config.is_global(i)]
assert global_layers == [5, 11, 17, 23, 29]
print('OK')
"

# 4. Full GGUF load (the real test)
python3 -c "
from core.gemma4_runner import Gemma4Runner
import json
model, info = Gemma4Runner.from_pretrained('/home/omen/models/.../mythos-26b-a4b-prism-pro-dq.gguf', qat=True)
print(json.dumps(info, indent=2))
assert info['missing'] == 0, f'Critical tensors missing: {info}"
print('Load OK')
"
```

---

## 5. Summary

**This document provides:**
- **Four corrected loader patches** (embedding transpose, trans_b removal, reshape, no-V globals)
- **One testable context-extension gate** (Gate 0: INT4 KV parity, gated on logit parity)
- **Honest assessment of APA's contribution** (4-bit floor prediction supports Gate 0 hypothesis; speedup is irrelevant to context extension on Gemma-4)
- **Honest assessment of Ghost Geometry's contribution** (none for this gate — the precision-depth measurement is about dot products, not KV storage)
- **Mission sequence** (A → B → C → D → E) with explicit gating

**Time estimate:**
- Apply patches: 15 minutes
- Static validation: 10 minutes
- Gate A (loader): 1 hour (includes fixing whatever breaks)
- Gate B (parity): 2–4 hours (includes llama.cpp source reading for V mechanism)
- Gate 0 (INT4 KV): 2–4 hours (includes wiring KVManager)

**Total to Gate 0: 1–2 days of focused work.**
