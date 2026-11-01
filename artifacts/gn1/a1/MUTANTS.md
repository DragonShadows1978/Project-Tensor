# A1 mutant registration

Prior art: DeMillo, Lipton and Sayward (1978), unverified — lead to check. Re-derived on D1. One semantic decision per mutant, including both affine branches or dtype plumbing where necessary. No GPU executions or actual kills claimed. Shape/dtype errors during a test call are valid structural kills; device, collection or setup failures and timeouts are invalid. Every actual killer must be recorded by the parent after child exit. No equivalences claimed.

| ID | Decision | Child / parent expected | Predicted witness (not observed) |
|---|---|---|---|
| M01 | unbiased_variance | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M02 | epsilon_outside_sqrt | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M03 | epsilon_ignored | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M04 | wrong_reduction_axis | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M05 | wrong_group_extent | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M06 | interleaved_channels | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M07 | weight_omitted | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M08 | bias_omitted | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M09 | affine_swapped | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M10 | refusal_removed | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_nondivisible_refusal` |
| M11 | broadcast_extra_dimension | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M12 | normalization_omitted | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M13 | wrong_power | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M14 | default_epsilon_changed | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_constructor_defaults` |
| M15 | input_dtype_compute | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-half-float16]` |
| M16 | output_not_cast_back | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-default-float16]` |
| M17 | parameters_not_cast_for_fp32_input | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_fp32_input_half_parameters[forward]` |
| M18 | input_upcast_without_parameter_casts | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-half-float16]` |
