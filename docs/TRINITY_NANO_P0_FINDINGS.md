# Trinity Nano P0 Architecture Scout — Findings DRAFT

**Status:** DRAFT for lead fold-in to `docs/TRINITY_NANO_PORT_LEDGER.md`.  
**Order:** P0 architecture scout (read-only except this file). No GPU. No git commit.  
**Sources:** HuggingFace `arcee-ai/Trinity-Nano-Preview` raw main, fetched 2026-07-08 via atlasforge-web-proxy + direct curl for exact line numbers.  
**Evidence classes:** `[SOURCED file:line]` = verbatim from model code/config; `[REASONING]` = inference; `[CONFIG]` = config.json receipt.

**Primary files:**
- `modeling_afmoe.py` (680 lines; line numbers below match raw/main)
- `configuration_afmoe.py` (133 lines)
- `config.json` (Nano checkpoint values)
- `model.safetensors.index.json` + HF tree API sizes
- `LICENSE` (OpenMDW-1.1), `README.md` license section

---

## Verdict (load-bearing first)

| Question | Answer |
|---|---|
| **(a) full_attention RoPE or NoPE?** | **NoPE.** RoPE applied **only** when `is_local_attention` (i.e. `layer_types[i] == "sliding_attention"`). Full-attention layers skip `apply_rotary_pos_emb` entirely. |
| **T1 condition** | **CONFIRMED** by custom code (not blog). Full layers = NoPE. Sliding layers = standard RoPE. |
| **Port-blocking surprises** | **None that void T1–T3 premises.** Several **parity-critical** details for P2 (gated attn epilogue, dual residual norms, muP embed scale, expert_bias in top-k). |
| **LICENSE** | **OpenMDW-1.1** — permissive use/mod/redistribution of Model Materials; retain license + notices; patent/copyright lawsuit termination clause. OK for research port + commercial use of materials; outputs unrestricted. |
| **Download size** | **Index `total_size` = 12,240,020,480 B (~11.40 GiB tensor payload).** Three shards sum (LFS) = **12,242,694,352 B (~11.40 GiB / 12.24 GB)**. |

---

## (a) full_attention: RoPE vs NoPE — LOAD-BEARING

**Verdict: full_attention layers are NoPE. Sliding layers apply RoPE.**

Layer locality flag from config `layer_types`:

```326:327:modeling_afmoe.py
self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
self.sliding_window = config.sliding_window if self.is_local_attention else None
```

**[SOURCED modeling_afmoe.py:326-327]**

Decisive application gate — RoPE runs **only** inside `if self.is_local_attention`:

```374:376:modeling_afmoe.py
if self.is_local_attention:
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

**[SOURCED modeling_afmoe.py:374-376]**

There is **no** `else` branch applying RoPE for full attention. `position_embeddings` are still computed once per forward and passed to every layer, but full layers ignore them. **[SOURCED modeling_afmoe.py:573-583]** (rotary always computed; layer loop always passes `position_embeddings`).

Nano `layer_types` (config.json): full at indices **3,7,11,...,55** (14 full / 42 sliding). Formula when unset:

```108:109:configuration_afmoe.py
"sliding_attention" if bool((i + 1) % global_attn_every_n_layers) else "full_attention"
```

**[SOURCED configuration_afmoe.py:108-109]** **[CONFIG]** `global_attn_every_n_layers=4`.

**Implication for T1:** GRM arena RoPE-hole law is predicted **not** to apply on the 14 full (NoPE) layers. Sliding layers **do** use RoPE (θ=10000) — hole law may still apply there; T1 scopes full layers only. **[REASONING]**

---

## (b) Gated attention — form and location

**Form:** elementwise (per-channel) sigmoid gate on the **concatenated multi-head attention output**, **before** `o_proj`.

Gate projection source = same pre-attention residual stream (post-`input_layernorm` hidden), width = `num_heads * head_dim` (matches attn output, not `hidden_size` alone for Nano: 8×128=1024 == hidden, coincidence of dims):

```345:347:modeling_afmoe.py
self.gate_proj = nn.Linear(
    config.hidden_size, self.num_heads * self.head_dim, bias=False
)
```

**[SOURCED modeling_afmoe.py:345-347]**

Forward — project gate from hidden; apply **after** attention reshape, **before** o_proj:

```362:365:modeling_afmoe.py
query_states = self.q_proj(hidden_states).view(hidden_shape)
key_states = self.k_proj(hidden_states).view(hidden_shape)
value_states = self.v_proj(hidden_states).view(hidden_shape)
gate_states = self.gate_proj(hidden_states)
```

```400:402:modeling_afmoe.py
output = output.view(*input_shape, -1).contiguous()
output = output * F.sigmoid(gate_states)
return self.o_proj(output)
```

**[SOURCED modeling_afmoe.py:362-365,400-402]**

| Detail | Value | Evidence |
|---|---|---|
| Activation | `F.sigmoid` | [SOURCED :401] |
| Scope | Elementwise on full `heads*head_dim` vector | gate_proj out dim = heads*head_dim [SOURCED :345-347,401] |
| Per-head only? | No — per channel across all heads | [REASONING from shapes] |
| Source | `hidden_states` into attention (after input LN) | [SOURCED :365] |
| vs o_proj | Gate **before** o_proj | [SOURCED :401-402] |
| Bias on gate_proj | False | [SOURCED :346] |

**Port note:** new attention epilogue kernel (or fused path) required: `o_proj(sigmoid(gate(x)) ⊙ attn_out)`. **[REASONING]**

---

## (c) Sliding layers: RoPE + window mask

| Item | Nano value | Evidence |
|---|---|---|
| RoPE theta | `10000` | **[CONFIG]** `rope_theta: 10000` |
| rope_scaling | `null` → type `"default"` | **[CONFIG]**; **[SOURCED modeling_afmoe.py:36-39]** |
| max_position | 131072 | **[CONFIG]** |
| Window | 2048 | **[CONFIG]** `sliding_window: 2048` |
| Who gets window | sliding only (`sliding_window=None` on full) | **[SOURCED :326-327]** |

RoPE implementation: standard `AfmoeRotaryEmbedding` + `apply_rotary_pos_emb` (half-rotate). No yarn/NTK unless rope_scaling set. **[SOURCED modeling_afmoe.py:31-90,100-122]**

Window masking: dual mask maps built in model forward:

```563:566:modeling_afmoe.py
causal_mask_mapping = {
    "full_attention": create_causal_mask(**mask_kwargs),
    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
}
```

**[SOURCED modeling_afmoe.py:563-566]**

Per-layer mask selected by `decoder_layer.attention_type`. Also passes `sliding_window=self.sliding_window` into `attention_interface` for SDPA/flash backends. **[SOURCED :396,575-576]**

---

## (d) MoE routing

**Config (Nano):** `score_func=sigmoid`, `route_norm=true`, `route_scale=2.826`, `num_experts=128`, `num_experts_per_tok=8`, `num_shared_experts=1`, `moe_intermediate_size=256`. **[CONFIG]**

### Router forward (complete math)

```220:244:modeling_afmoe.py
def forward(self, hidden_states, expert_bias: torch.Tensor | None):
    _, _, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    scores = self.gate(hidden_states)

    # Apply scoring function in float32 for stability
    if self.score_func == "sigmoid":
        scores = torch.sigmoid(scores.to(torch.float32))
    else:
        scores = F.softmax(scores.to(torch.float32), dim=-1)

    if expert_bias is not None:
        _, selected_experts = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
        top_scores = scores.gather(dim=1, index=selected_experts)
    else:
        top_scores, selected_experts = torch.topk(scores, k=self.top_k, dim=1)

    # Normalize weights if using sigmoid
    if self.score_func == "sigmoid" and self.route_norm:
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator

    top_scores = top_scores * self.route_scale
    return top_scores, selected_experts
```

**[SOURCED modeling_afmoe.py:220-244]**

| Step | Math | Notes |
|---|---|---|
| Scores | `sigmoid(W_g h)` in fp32 | not softmax |
| Top-8 selection | `topk(scores + expert_bias)` | bias used for **selection only** |
| Weights | `gather(scores)` (no bias) | DeepSeek-style aux-free bias |
| route_norm | `top_scores / (sum + 1e-20)` | only if sigmoid AND route_norm |
| route_scale | `top_scores *= route_scale` | **after** norm; Nano = **2.826** |

`expert_bias`: `nn.Parameter(zeros(num_experts), requires_grad=False)` — present in checkpoints (54 tensors). **[SOURCED modeling_afmoe.py:262]** **[SOURCED weight_map: `mlp.expert_bias`]**

### Shared + routed combine

```272:307:modeling_afmoe.py
if self.shared_experts is not None:
    shared_output = self.shared_experts(hidden_states_flat)
...
routed_output = (
    routed_output.to(torch.float32) * top_scores_sorted.unsqueeze(-1)
).to(hidden_states.dtype)

# Scatter back to original positions
output = shared_output.scatter_add(
    dim=0, index=token_indices_expanded, src=routed_output
)
```

**[SOURCED modeling_afmoe.py:272-307]**

- Shared expert: **one** `AfmoeMLP` with `intermediate = moe_intermediate_size * num_shared_experts` (=256×1=256 on Nano). **[SOURCED :253-256]**
- Combine: `shared + sum_k (w_k * expert_k(h))` via `scatter_add`.
- Routed experts: ModuleList of 128 `AfmoeMLP(inter=256)`.
- `n_group` / `topk_group` exist in config but **are never referenced in modeling_afmoe.py** — dead config for this codepath. **[REASONING]**

---

## (e) Attention norms / softmax tweaks

### QK-Norm: **YES**, standard RMSNorm (NOT zero-centered)

```342:343:modeling_afmoe.py
self.q_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

Applied after q/k proj, before transpose / RoPE:

```367:368:modeling_afmoe.py
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
```

**[SOURCED modeling_afmoe.py:342-343,367-368]**

`AfmoeRMSNorm` (docstring: "equivalent to T5LayerNorm"):

```151:155:modeling_afmoe.py
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

- Weight init: `torch.ones` — **not** zero-centered (not Gemma-2 style `(1+w)*x`). **[SOURCED :147,151-155]**
- No mean subtraction. **[SOURCED]**

### Softmax / sink / softcap

```175:179:modeling_afmoe.py
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
...
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
    query.dtype
)
```

**[SOURCED modeling_afmoe.py:175-179]**

- Softmax in fp32: yes.
- Attention sink: **absent** in custom code. **[SOURCED — no sink tokens/logits]**
- Logit softcap: **absent** (no tanh/softcap in attention or lm_head). **[SOURCED — search null]**

---

## (f) Dense layers 0–1

```423:427:modeling_afmoe.py
self.moe_enabled = layer_idx >= config.num_dense_layers
if self.moe_enabled:
    self.mlp = AfmoeMoE(config)
else:
    self.mlp = AfmoeMLP(config)
```

**[SOURCED modeling_afmoe.py:423-427]** **[CONFIG]** `num_dense_layers: 2` → layers **0 and 1** are dense.

`AfmoeMLP` = standard SwiGLU-style:

```204:204:modeling_afmoe.py
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**[SOURCED modeling_afmoe.py:192-204]**

| Dense MLP param | Value |
|---|---|
| intermediate_size | **3072** **[CONFIG]** |
| hidden_act | **silu** **[CONFIG]** |
| biases | all False **[SOURCED :198-200]** |

Weight map confirms layers 0–1 have `mlp.{gate,up,down}_proj` only (no experts/router/shared). Layers ≥2 have full MoE. **[SOURCED weight_map]**

---

## (g) Other from-scratch CUDA port must-knows

### Attention scale
`self.scaling = self.head_dim**-0.5` → **1/√128**. **[SOURCED modeling_afmoe.py:324]**  
Passed into attention_interface as `scaling=self.scaling`. **[SOURCED :395]**

### muP (inference-critical)
`mup_enabled: true` **[CONFIG]**. Only forward effect in this code:

```570:572:modeling_afmoe.py
# Apply muP input scaling if enabled
if self.config.mup_enabled:
    hidden_states = hidden_states * (self.config.hidden_size**0.5)
```

**[SOURCED modeling_afmoe.py:570-572]**  
→ embed scale by **√1024 = 32**. No other muP factors (no attn logit rescale, no output scale) in this file. **[SOURCED — only this site]**  
**Parity will fail if omitted.** **[REASONING]**

### Embedding / lm_head
- Separate `lm_head` Linear(hidden, vocab, bias=False). **[SOURCED :604]**
- `tie_word_embeddings: false` **[CONFIG]**; distinct `lm_head.weight` and `model.embed_tokens.weight` in weight_map → **untied**. **[SOURCED weight_map]**
- Class has `_tied_weights_keys = ["lm_head.weight"]` (HF convention when tying enabled) but tying is off. **[SOURCED :596]** **[REASONING]**

### Logit softcap
**None.** `logits = self.lm_head(hidden_states[:, slice_indices, :])` only. **[SOURCED :659]**

### Dual residual norms (Gemma-like)
Each decoder layer:

1. `x = x + post_attn_ln(attn(input_ln(x)))`
2. `x = x + post_mlp_ln(mlp(pre_mlp_ln(x)))`

**[SOURCED modeling_afmoe.py:414-420, 437-458]**  
Four RMSNorms per layer + final `model.norm`. Not standard single pre-norm Llama. **Port-blocking for parity if modeled as Llama-pre-norm.** **[REASONING]**

### GQA geometry
8 Q heads, 2 KV heads, head_dim 128, groups=4. **[CONFIG]**  
`repeat_kv` in eager path. **[SOURCED :128-138,171-172]**

### Cache / KV
Standard `past_key_value.update` after optional RoPE. Full layers cache **unrotated** K (NoPE); sliding cache **RoPE-rotated** K. **[SOURCED :374-380]** **[REASONING]**  
**APA implication:** full-layer KV has no rotary phase — consistent with T2. Sliding APA out of scope (bounded window) per plan.

### Dead / unused config fields in modeling
`n_group`, `topk_group`, `num_expert_groups`, `num_limited_groups`, `load_balance_coeff` (config.json) — no references in modeling_afmoe.py. **[REASONING]**

### Dtype
Checkpoint `dtype: bfloat16`. **[CONFIG]**

### Active params (for sizing only)
README claims ~1B active / 6B total; index `total_parameters: 6_120_003_328`. **[SOURCED index metadata]** Active path ≈ dense MLP + 8×moe_inter experts + 1 shared + attn; not re-derived here. **[REASONING — not re-counted]**

---

## (h) LICENSE + download size

### License verdict
| Item | Value |
|---|---|
| SPDX-ish name | **OpenMDW-1.1** (`license_name: openmdw-1.1`, `license: other`) **[SOURCED README frontmatter]** |
| File | `LICENSE` — "OpenMDW License Agreement, version 1.1" **[SOURCED LICENSE]** |
| Grant | Free of charge to deal in Model Materials without restriction (copyright/patent/database/trade secret) subject to compliance |
| Redistribution | Must retain (1) copy of agreement, (2) applicable copyright/origin notices |
| Patent termination | Filing suit asserting Model Materials infringe patent/copyright terminates grants (except defensive response) |
| Outputs | **No restrictions** on use/mod/sharing of generated outputs |
| Warranty | AS IS; no liability |

**Port verdict:** Compatible with internal research port, redistribution of modified engine code is separate (our code); model weights/artifacts under OpenMDW-1.1 need license retain if redistributed. Not GPL. **[REASONING]**

### Shards (names + sizes)

From HF tree API LFS sizes (download bytes):

| Shard | Size (bytes) | ≈ GiB |
|---|---:|---:|
| `model-00001-of-00003.safetensors` | 5,000,898,920 | 4.657 |
| `model-00002-of-00003.safetensors` | 5,001,006,952 | 4.658 |
| `model-00003-of-00003.safetensors` | 2,240,788,480 | 2.087 |
| **Sum (download)** | **12,242,694,352** | **11.402** |

Index metadata: `total_size: 12,240,020,480` (tensor payload; slight under-sum vs file sizes = headers). `total_parameters: 6_120_003_328`. **[SOURCED model.safetensors.index.json metadata]**

Plus small non-weight files (index ~1.9 MB, code, tokenizer — not load-bearing for VRAM).

---

## T1–T3 premise check (early-stop rail)

| Premise | P0 result | Gate |
|---|---|---|
| T1 NoPE on full_attention | **CONFIRMED** | Proceed; T1 experiment remains valid |
| T2 APA on 14 full layers (kv=2, d=128, unbounded) | Geometry matches; full layers unbounded NoPE | Proceed |
| T3 residency ~12.2GB bf16 weights → INT4 shrink + KV | Weight disk ~11.4 GiB bf16; INT4 path still required for 12GB card | No void; estimate discipline unchanged |
| Early-stop divergence voiding premises | **Not triggered** | P1+ authorized by this scout |

---

## Port map (P2 checklist, minimal)

Must implement for T4 parity:

1. **NoPE full / RoPE sliding** branch on `layer_types` (not “RoPE everywhere”).
2. **QK RMSNorm** (standard, not zero-centered) before RoPE (sliding) / scores (full).
3. **Gated attn:** `o_proj(sigmoid(gate_proj(x)) ⊙ attn_out)` after multi-head concat.
4. **Scale** `1/√head_dim`; softmax fp32.
5. **Dual residual LN** (pre+post attn, pre+post mlp).
6. **muP:** `embed *= √hidden_size` when enabled.
7. **Dense L0–L1** SwiGLU inter=3072; **MoE L2–L55** sigmoid-top8 + route_norm + ×2.826 + expert_bias selection + shared scatter_add.
8. **Untied** lm_head; no softcap.
9. Sliding mask window=2048; full causal unbounded.

New kernels likely: gated-attention epilogue; sigmoid router (+ optional fused route_norm/scale). Rest maps onto existing tensor_cuda primitives. **[REASONING]**

---

## Evidence residuals (honest)

- Did **not** execute HF forward or compare numerics (P0 no GPU / no weight download).
- Did **not** inspect tokenizer specials, chat template, or generation defaults beyond architecture.
- Did **not** read transformers' `create_sliding_window_causal_mask` implementation (assumed standard HF sliding causal).
- Line numbers are for `raw/main` as of 2026-07-08; re-verify if upstream retags.

---

## Receipts (fetch)

| URL | Result |
|---|---|
| `.../raw/main/modeling_afmoe.py` | OK (680 lines) |
| `.../raw/main/configuration_afmoe.py` | OK |
| `.../raw/main/config.json` | OK (cache hit) |
| `.../raw/main/model.safetensors.index.json` | OK |
| `.../raw/main/LICENSE` | OpenMDW-1.1 |
| `.../raw/main/README.md` | license_name openmdw-1.1 |
| HF tree API shard sizes | OK |

*End P0 findings DRAFT.*
