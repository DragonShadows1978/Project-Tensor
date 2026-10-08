| path | special row | cells | fd32b6d | MF2 | MF2.1 | MF2.2 | after1 | after2 | fd32b6d finite & after2 NaN |
|---|---|---|---|---|---|---|---|---|---|
| fused_sdpa | all_blocked | 48 | 48 | 48 | 0 | 0 | 48 | 48 | 0 |
| int4_all | all_blocked | 48 | 48 | 48 | 0 | 0 | 48 | 48 | 0 |
| int4_partial | all_blocked | 48 | 48 | 48 | 0 | 0 | 48 | 48 | 0 |
| sdpa_infer_noncausal | all_blocked | 72 | 0 | 0 | 0 | 0 | 0 | 72 | 0 |
| sdpa_infer_causal | all_blocked | 54 | 0 | 0 | 0 | 0 | 0 | 54 | 0 |
| sdpa_infer_causal_fused | all_blocked | 54 | 0 | 0 | 0 | 0 | 0 | 54 | 0 |
| apa_quant_p1.0_causalFalse | all_blocked | 72 | 0 | 0 | 0 | 0 | 0 | 72 | 0 |
| apa_quant_p1.0_causalTrue | all_blocked | 54 | 0 | 0 | 0 | 0 | 0 | 54 | 0 |
| apa_quant_p0.15_causalFalse | all_blocked | 72 | 0 | 0 | 0 | 0 | 0 | 72 | 0 |
| apa_quant_p0.15_causalTrue | all_blocked | 54 | 0 | 0 | 0 | 0 | 0 | 54 | 0 |
| sdpa_grad_causal | all_blocked | 54 | 0 | 0 | 0 | 0 | 0 | 51 | 0 |
| sdpa_infer_masked | all_masked | 72 | 0 | 0 | 0 | 0 | 0 | 72 | 0 |
| sdpa_grad_masked | all_masked | 72 | 0 | 0 | 0 | 0 | 0 | 72 | 0 |
| fused_sdpa | S=1 blocked | 12 | 12 | 12 | 0 | 0 | 12 | 12 | 0 |
| int4_all | S=1 blocked | 12 | 12 | 12 | 0 | 0 | 12 | 12 | 0 |
| int4_partial | S=1 blocked | 12 | 12 | 12 | 0 | 0 | 12 | 12 | 0 |
| sdpa_infer_noncausal | S=1 blocked | 18 | 0 | 0 | 0 | 0 | 0 | 18 | 0 |
| sdpa_infer_causal | S=1 blocked | 9 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| sdpa_infer_causal_fused | S=1 blocked | 9 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| apa_quant_p1.0_causalFalse | S=1 blocked | 18 | 0 | 0 | 0 | 0 | 0 | 18 | 0 |
| apa_quant_p1.0_causalTrue | S=1 blocked | 9 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| apa_quant_p0.15_causalFalse | S=1 blocked | 18 | 0 | 0 | 0 | 0 | 0 | 18 | 0 |
| apa_quant_p0.15_causalTrue | S=1 blocked | 9 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| sdpa_grad_causal | S=1 blocked | 9 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| fused_sdpa | single_visible | 48 | 48 | 48 | 12 | 48 | 48 | 48 | 0 |
| int4_all | single_visible | 48 | 48 | 48 | 12 | 48 | 48 | 48 | 0 |
| int4_partial | single_visible | 48 | 48 | 48 | 12 | 48 | 48 | 48 | 0 |
| sdpa_infer_noncausal | single_visible | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 0 |
| sdpa_infer_causal | single_visible | 54 | 54 | 36 | 36 | 36 | 36 | 54 | 0 |
| sdpa_infer_causal_fused | single_visible | 54 | 36 | 36 | 36 | 36 | 36 | 54 | 0 |
| apa_quant_p1.0_causalFalse | single_visible | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 0 |
| apa_quant_p1.0_causalTrue | single_visible | 54 | 54 | 36 | 36 | 36 | 36 | 54 | 0 |
| apa_quant_p0.15_causalFalse | single_visible | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 0 |
| apa_quant_p0.15_causalTrue | single_visible | 54 | 54 | 36 | 36 | 36 | 36 | 54 | 0 |
| sdpa_grad_causal | single_visible | 54 | 54 | 36 | 36 | 36 | 36 | 54 | 0 |
| fused_sdpa | neg_inf_first | 36 | 36 | 36 | 0 | 36 | 36 | 36 | 0 |
| int4_all | neg_inf_first | 36 | 36 | 36 | 0 | 36 | 36 | 36 | 0 |
| int4_partial | neg_inf_first | 36 | 36 | 36 | 0 | 36 | 36 | 36 | 0 |
| sdpa_infer_noncausal | neg_inf_first | 54 | 54 | 54 | 54 | 54 | 54 | 54 | 0 |
| sdpa_infer_causal | neg_inf_first | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 |
| sdpa_infer_causal_fused | neg_inf_first | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 |
| apa_quant_p1.0_causalFalse | neg_inf_first | 54 | 54 | 54 | 54 | 54 | 54 | 54 | 0 |
| apa_quant_p1.0_causalTrue | neg_inf_first | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 |
| apa_quant_p0.15_causalFalse | neg_inf_first | 54 | 54 | 54 | 54 | 54 | 54 | 54 | 0 |
| apa_quant_p0.15_causalTrue | neg_inf_first | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 |
| sdpa_grad_causal | neg_inf_first | 45 | 33 | 33 | 33 | 33 | 33 | 33 | 0 |
| eager_softmax | all_blocked | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| causal_softmax | all_blocked | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| eager_softmax | S=1 blocked | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| causal_softmax | S=1 blocked | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| eager_softmax | single_visible | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 0 |
| causal_softmax | single_visible | 18 | 12 | 12 | 12 | 12 | 12 | 12 | 0 |
| eager_softmax | neg_inf_first | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 0 |
| causal_softmax | neg_inf_first | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 0 |
