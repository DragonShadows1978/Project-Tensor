#!/usr/bin/env bash
# APA-SP5 amendment 2 item 2 -- streamed-mode 16K rung, reproducing the H4
# ladder's construction (same script, same prompt bytes, same settings,
# same bulk_bits=8, --skip-lm-head, resident_packed_mxfp4).
#
# arm B = stock two-pass, exactly the H4 command.
# arm C = identical EXCEPT the single-pass entry is injected in-process via
#         artifacts/apa_sp5/a2_inject/sitecustomize.py, because the streamed
#         path has NO single-pass seam: the ladder emits only
#         --attention-mode {standard,apa_selective}, and the port's sole
#         apa_selective branch is hard-wired to the two-pass entry at
#         core/gpt_oss20b_tc.py:744. The port and the smoke are READ-ONLY and
#         are NOT edited. The smoke also hard-codes the stock engine at
#         smoke:24, which has no SP entry, so sitecustomize preloads the SP5
#         engine first. Label results accordingly.
set -uo pipefail
ARM=${1:?arm B|C}; TOK=${2:-16384}
GR=/mnt/ForgeRealm/GraftRepository
WT=/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5
PROMPT=$GR/artifacts/gpt_oss_20b/h4_context_ladder_apa_16k_sampled/prompts/prompt_16384.txt
OUT=$WT/artifacts/apa_sp5/ladder_a2/ladder_stream_${ARM}_${TOK}.json
mkdir -p "$(dirname "$OUT")"
cd "$GR"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false TC_APA_SP=1
export PYTHONPATH=$WT/artifacts/apa_sp5/a2_inject:$GR
if [[ "$ARM" == C ]]; then
  export APA_SP5_SP_DELTA=3.16
  export APA_SP5_ENGINE_ROOT=$WT/tensor_cuda
  export APA_SP5_ENGINE_BUILD=$WT/artifacts/apa_sp5/build
else
  unset APA_SP5_SP_DELTA
fi
python3 scripts/gpt_oss20b_stream_forward_smoke.py \
  --prompt-file "$PROMPT" \
  --max-tokens "$TOK" \
  --attention-mode apa_selective \
  --apa-layer-scope full \
  --refine-percentile 0.15 \
  --bulk-bits 8 \
  --expert-mode resident_packed_mxfp4 \
  --skip-lm-head \
  --output "$OUT"
rc=$?
echo "LADDER_DONE arm=$ARM tokens=$TOK rc=$rc out=$OUT"
exit $rc
