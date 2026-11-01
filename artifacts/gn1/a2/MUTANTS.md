# A2 mutant table

Prior art: DeMillo/Lipton/Sayward (1978), mutation adequacy; Offutt/Voas RIP model, unverified — lead to check. All 41 load, differ and reach their single edit site. No multiline edit exception is needed. Witness prefix: `tensor_cuda/tests/test_groupnorm_gn1.py::`. 40 CPU assertion kills with explicit NumPy engine stand-in; one V3 equivalence. No actual GPU kills claimed.

| Mutant | Witness (prefix below) | Edited → reached lines | CPU outcome | Trace |
|---|---|---|---|---|
| D5_bias_dtype | `test_dtype_refusal[int64-bias]` | [543] → [543] | AssertionError | [trace](mutants/D5_bias_dtype/trace.json) |
| D5_c_bool | `test_positive_integer_refusal[true-num_channels]` | [520] → [520] | AssertionError | [trace](mutants/D5_c_bool/trace.json) |
| D5_c_int | `test_positive_integer_refusal[float-num_channels]` | [520] → [520] | AssertionError | [trace](mutants/D5_c_int/trace.json) |
| D5_c_positive | `test_positive_integer_refusal[negative-num_channels]` | [521] → [521] | AssertionError | [trace](mutants/D5_c_positive/trace.json) |
| D5_channels | `test_shape_refusal[channels]` | [537] → [537] | AssertionError | [trace](mutants/D5_channels/trace.json) |
| D5_g_bool | `test_positive_integer_refusal[true-num_groups]` | [517] → [517] | AssertionError | [trace](mutants/D5_g_bool/trace.json) |
| D5_g_int | `test_positive_integer_refusal[float-num_groups]` | [517] → [517] | AssertionError | [trace](mutants/D5_g_int/trace.json) |
| D5_g_positive | `test_positive_integer_refusal[negative-num_groups]` | [518] → [518] | AssertionError | [trace](mutants/D5_g_positive/trace.json) |
| D5_input_dtype | `test_dtype_refusal[int64-input]` | [539] → [539] | AssertionError | [trace](mutants/D5_input_dtype/trace.json) |
| D5_rank | `test_shape_refusal[rank1]` | [535] → [535] | AssertionError | [trace](mutants/D5_rank/trace.json) |
| D5_weight_dtype | `test_dtype_refusal[int64-weight]` | [541] → [541] | AssertionError | [trace](mutants/D5_weight_dtype/trace.json) |
| M01 | `test_forward_values[float32-r2-n1-g2-default]` | [558] → [558] | AssertionError | [trace](mutants/M01/trace.json) |
| M02 | `test_near_epsilon[forward-float32]` | [559] → [559] | AssertionError | [trace](mutants/M02/trace.json) |
| M03 | `test_near_epsilon[forward-float32]` | [559] → [559] | AssertionError | [trace](mutants/M03/trace.json) |
| M04m | `test_forward_values[float32-r3-n2-g2-default]` | [558] → [558] | AssertionError | [trace](mutants/M04m/trace.json) |
| M04v | `test_forward_values[float32-r3-n2-g2-default]` | [558] → [558] | AssertionError | [trace](mutants/M04v/trace.json) |
| M05 | `test_forward_values[float32-r2-n1-g2-default]` | [555] → [555] | AssertionError | [trace](mutants/M05/trace.json) |
| M06in | `test_forward_values[float32-r3-n2-g2-default]` | [555] → [555] | AssertionError | [trace](mutants/M06in/trace.json) |
| M06out | `test_forward_values[float32-r3-n2-g2-default]` | [560] → [560] | AssertionError | [trace](mutants/M06out/trace.json) |
| M07cast | `test_forward_values[float16-r3-n2-g2-default]` | [567] → [567] | AssertionError | [trace](mutants/M07cast/trace.json) |
| M07fast | `test_forward_values[float32-r3-n2-g2-default]` | [564] → [564] | AssertionError | [trace](mutants/M07fast/trace.json) |
| M08cast | `test_forward_values[float16-r3-n2-g2-default]` | [567] → [567] | AssertionError | [trace](mutants/M08cast/trace.json) |
| M08fast | `test_forward_values[float32-r3-n2-g2-default]` | [564] → [564] | AssertionError | [trace](mutants/M08fast/trace.json) |
| M09cast | `test_forward_values[float16-r3-n2-g2-default]` | [567] → [567] | AssertionError | [trace](mutants/M09cast/trace.json) |
| M09fast | `test_forward_values[float32-r3-n2-g2-default]` | [564] → [564] | AssertionError | [trace](mutants/M09fast/trace.json) |
| M10 | `test_nondivisible_refusal` | [523] → [523] | AssertionError | [trace](mutants/M10/trace.json) |
| M11 | `test_forward_values[float32-r3-n2-g2-default]` | [561] → [561] | AssertionError | [trace](mutants/M11/trace.json) |
| M12 | `test_forward_values[float32-r3-n2-g2-default]` | [559] → [559] | AssertionError | [trace](mutants/M12/trace.json) |
| M13 | `test_forward_values[float32-r3-n2-g2-default]` | [559] → [559] | AssertionError | [trace](mutants/M13/trace.json) |
| M14 | `test_constructor_defaults` | [511] → [511] | AssertionError | [trace](mutants/M14/trace.json) |
| M15b | `test_forward_values[float16-r3-n2-g2-default]` | [566] → [566] | AssertionError | [trace](mutants/M15b/trace.json) |
| M15w | `test_forward_values[float16-r3-n2-g2-default]` | [565] → [565] | AssertionError | [trace](mutants/M15w/trace.json) |
| M15x | `test_forward_values[float16-r3-n2-g2-default]` | [550] → [550] | AssertionError | [trace](mutants/M15x/trace.json) |
| M16 | `test_mixed_parameters[forward-default-float16]` | [568] → [568] | AssertionError | [trace](mutants/M16/trace.json) |
| M17b | `test_distinct_parameter_dtypes[forward-w32-b16]` | [566] → [566] | AssertionError | [trace](mutants/M17b/trace.json) |
| M17w | `test_distinct_parameter_dtypes[forward-w16-b32]` | [565] → [565] | AssertionError | [trace](mutants/M17w/trace.json) |
| M18b | `test_mixed_parameters[forward-half-float16]` | [566] → [566] | AssertionError | [trace](mutants/M18b/trace.json) |
| M18w | `test_mixed_parameters[forward-half-float16]` | [565] → [565] | AssertionError | [trace](mutants/M18w/trace.json) |
| V1 | `test_distinct_parameter_dtypes[forward-w32-b16]` | [563] → [563] | AssertionError | [trace](mutants/V1/trace.json) |
| V2 | `test_distinct_parameter_dtypes[forward-w16-b32]` | [562] → [562] | AssertionError | [trace](mutants/V2/trace.json) |
| V3 | `test_forward_values[float16-r3-n2-g2-default]` | [550] → [550] | equivalent (V3 proof) | [trace](mutants/V3/trace.json) |
