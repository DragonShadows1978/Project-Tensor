"""Model-realistic geometries for the Phase 0.1 kernel microbench harness.

Extracted by code inspection (2026-07-07) from the live driver configs in
GraftRepository/core/ — NOT invented numbers. Each geometry cites the class
it was read from so a reviewer can re-derive it.

Sources:
  - Qwen3.5-9B:  GraftRepository/core/qwen35_tc.py:Qwen35Config
  - Gemma 4:     GraftRepository/core/gemma4_tc.py:Gemma4Config
  - GPT-OSS-20B: GraftRepository/core/gpt_oss20b_tc.py:GptOss20BConfig
    (ledger 2026-07-07 09:20: extension directed by David, sink-aware APA
    blend + resident_packed_mxfp4 expert mode)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttnGeometry:
    """One attention-layer shape family for a model."""

    model: str
    layer_kind: str          # "standard", "sliding", "global-mqa", "sink-sliding", "sink-full"
    num_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool = True
    sinks: bool = False


@dataclass(frozen=True)
class FFNGeometry:
    """Dense FFN (SwiGLU) shape for a model."""

    model: str
    hidden_dim: int
    intermediate_dim: int


@dataclass(frozen=True)
class MoEGeometry:
    """MXFP4 expert shape (GPT-OSS)."""

    model: str
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    experts_per_tok: int
    group_size: int = 32   # MXFP4 block size (16 bytes -> 32 fp4 values)


# ---------------------------------------------------------------------------
# Qwen3.5-9B (GraftRepository/core/qwen35_tc.py:Qwen35Config)
# vocab 248320, hidden 4096, intermediate 12288, 32 layers,
# 16 q heads / 4 kv heads, head_dim 256. GQA throughout (no MQA/global split).
# ---------------------------------------------------------------------------
QWEN35 = {
    "attn": AttnGeometry("qwen35", "standard", num_heads=16, num_kv_heads=4, head_dim=256),
    "ffn": FFNGeometry("qwen35", hidden_dim=4096, intermediate_dim=12288),
}

# ---------------------------------------------------------------------------
# Gemma 4 (text-only 12B) (GraftRepository/core/gemma4_tc.py:Gemma4Config)
# vocab 262144, hidden 3840, intermediate 15360, 48 layers:
#   40 sliding-window (window 1024, GQA 16q/8kv, head_dim 256)
#   8 global (i % 6 == 5) (MQA 16q/1kv, head_dim 512, K=V shared -> num_kv_heads=1)
# ---------------------------------------------------------------------------
GEMMA4 = {
    "attn_sliding": AttnGeometry("gemma4", "sliding", num_heads=16, num_kv_heads=8, head_dim=256),
    "attn_global": AttnGeometry("gemma4", "global-mqa", num_heads=16, num_kv_heads=1, head_dim=512),
    "ffn": FFNGeometry("gemma4", hidden_dim=3840, intermediate_dim=15360),
    "sliding_window": 1024,
}

# ---------------------------------------------------------------------------
# GPT-OSS-20B (GraftRepository/core/gpt_oss20b_tc.py:GptOss20BConfig)
# vocab 201088, hidden 2880, intermediate 2880, 24 layers,
# 64 q heads / 8 kv heads, head_dim 64, sliding_window 128 (alternating
# full/sliding layer_types), 32 local experts, 4 active/token, sink logits
# (tc.apa_blend_softmax_sink / tc.apa_selective_attention_sink), MXFP4 expert
# weights: gate_up out_features = 2*intermediate_dim, down out_features =
# hidden_dim, both packed in 32-wide MXFP4 groups (blocks (...,groups,16)).
# ---------------------------------------------------------------------------
GPT_OSS20B = {
    "attn_full_sink": AttnGeometry(
        "gpt_oss20b", "sink-full", num_heads=64, num_kv_heads=8, head_dim=64, sinks=True
    ),
    "attn_sliding_sink": AttnGeometry(
        "gpt_oss20b", "sink-sliding", num_heads=64, num_kv_heads=8, head_dim=64, sinks=True
    ),
    "ffn": FFNGeometry("gpt_oss20b", hidden_dim=2880, intermediate_dim=2880),
    "moe": MoEGeometry(
        "gpt_oss20b", hidden_dim=2880, intermediate_dim=2880,
        num_experts=32, experts_per_tok=4, group_size=32,
    ),
    "sliding_window": 128,
}

ALL_MODELS = {
    "qwen35": QWEN35,
    "gemma4": GEMMA4,
    "gpt_oss20b": GPT_OSS20B,
}

# Decode / prefill shape matrix (plan Phase 0.1). S = cache length at decode
# (KV-cache length the single new token attends over); L = prefill chunk
# length (L == S, square).
DECODE_S = (512, 2048, 8192)
PREFILL_L = (512, 2048)

# GPT-OSS near-full-context shape (ledger 09:20: "near-full context in VRAM
# on the 12 GB card, no OOM" claim for the sink-attention + resident mxfp4
# path). GPT-OSS-20B's native context is 131072; the fast blend path is
# gated to fast_max_seq=4096 (Mistral/Qwen driver default carried over) so
# large S here always exercises the O(L) apa_selective_kernel path, per the
# ledger's own prediction. Bounded to what fits in ~12GB at bench time.
GPT_OSS_NEAR_FULL_S = 32768
