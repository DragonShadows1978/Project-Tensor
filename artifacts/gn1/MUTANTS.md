# GN1 mutant registration

Written before execution; no mutant has run on GPU. Prior art: DeMillo, Lipton and Sayward (1978), mutation testing; unverified — lead to check. Every nn.py is the candidate with one registered literal replacement inside GroupNorm. No equivalences claimed. Runtime shape errors can kill structural mutants; collection/setup errors, timeouts or no-device errors never count. Numeric mutants have numerical witnesses. M01 also errors at extent one; its registered witness has extent 3.

| ID | Single decision changed | Expected pytest exit | Predicted witness (not observed killer) |
|---|---|---|---|
| M01 | unbiased_variance | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M02 | epsilon_outside_sqrt | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M03 | epsilon_ignored | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M04 | wrong_reduction_axis | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M05 | wrong_group_extent | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M06 | interleaved_channels | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M07 | weight_omitted | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M08 | bias_omitted | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M09 | affine_swapped | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M10 | refusal_removed | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_nondivisible_refusal` |
| M11 | broadcast_extra_dimension | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M12 | normalization_omitted | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M13 | wrong_power | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M14 | default_epsilon_changed | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_constructor_defaults` |

Actual killers will be populated by the lead runner from the exact pytest report node IDs, after the forge child exits.
