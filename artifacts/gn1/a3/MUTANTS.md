# Mutant table

All 52 load and differ at one source site inside GroupNorm. 51 are killed on the CPU stand-in; V3 is the previously registered equivalence. Actual GPU killers remain null. Each row has full phases, call-only trace and source diff under mutants/<ID>/. M14 records default binding rather than import-time reach.

| Mutant | Edit line | Registered witness | Call reach | CPU result | G5 |
| --- | --- | --- | --- | --- | --- |
| D5_bias_dtype | 554 | test_dtype_refusal[int64-bias] | 554 | CPU kill: AssertionError | PASS |
| D5_bias_length | 564 | test_parameter_shape_refusal[short-bias] | 564 | CPU kill: AssertionError | PASS |
| D5_bias_rank | 562 | test_parameter_shape_refusal[matrix61-bias] | 562 | CPU kill: AssertionError | PASS |
| D5_c_bool | 526 | test_positive_integer_refusal[true-num_channels] | 526 | CPU kill: AssertionError | PASS |
| D5_c_index | 529 | test_index_counts[np_int64-num_channels] | 529 | CPU kill: ValueError | PASS |
| D5_c_int | 529 | test_positive_integer_refusal[float-num_channels] | 529 | CPU kill: AssertionError | PASS |
| D5_c_npbool | 526 | test_index_counts[np_int64-num_channels] | 526 | CPU kill: ValueError | PASS |
| D5_c_positive | 532 | test_positive_integer_refusal[negative-num_channels] | 532 | CPU kill: AssertionError | PASS |
| D5_c_store | 529 | test_index_counts[np_int64-num_channels] | 529 | CPU kill: AssertionError | PASS |
| D5_channels | 548 | test_shape_refusal[channels] | 548 | CPU kill: AssertionError | PASS |
| D5_g_bool | 518 | test_positive_integer_refusal[true-num_groups] | 518 | CPU kill: AssertionError | PASS |
| D5_g_index | 521 | test_index_counts[np_int64-num_groups] | 521 | CPU kill: ValueError | PASS |
| D5_g_int | 521 | test_positive_integer_refusal[float-num_groups] | 521 | CPU kill: AssertionError | PASS |
| D5_g_npbool | 518 | test_index_counts[np_int64-num_groups] | 518 | CPU kill: ValueError | PASS |
| D5_g_positive | 524 | test_positive_integer_refusal[negative-num_groups] | 524 | CPU kill: AssertionError | PASS |
| D5_g_store | 521 | test_index_counts[np_int64-num_groups] | 521 | CPU kill: AssertionError | PASS |
| D5_input_dtype | 550 | test_dtype_refusal[int64-input] | 550 | CPU kill: AssertionError | PASS |
| D5_rank | 546 | test_shape_refusal[rank1] | 546 | CPU kill: AssertionError | PASS |
| D5_weight_dtype | 552 | test_dtype_refusal[int64-weight] | 552 | CPU kill: AssertionError | PASS |
| D5_weight_length | 560 | test_parameter_shape_refusal[short-weight] | 560 | CPU kill: AssertionError | PASS |
| D5_weight_rank | 558 | test_parameter_shape_refusal[matrix61-weight] | 558 | CPU kill: AssertionError | PASS |
| M01 | 579 | test_forward_values[float32-r2-n1-g2-default] | 579 | CPU kill: AssertionError | PASS |
| M02 | 580 | test_near_epsilon[forward-float32] | 580 | CPU kill: AssertionError | PASS |
| M03 | 580 | test_near_epsilon[forward-float32] | 580 | CPU kill: AssertionError | PASS |
| M04m | 579 | test_forward_values[float32-r3-n2-g2-default] | 579 | CPU kill: AssertionError | PASS |
| M04v | 579 | test_forward_values[float32-r3-n2-g2-default] | 579 | CPU kill: AssertionError | PASS |
| M05 | 576 | test_forward_values[float32-r2-n1-g2-default] | 576 | CPU kill: AssertionError | PASS |
| M06in | 576 | test_forward_values[float32-r3-n2-g2-default] | 576 | CPU kill: AssertionError | PASS |
| M06out | 581 | test_forward_values[float32-r3-n2-g2-default] | 581 | CPU kill: AssertionError | PASS |
| M07cast | 588 | test_forward_values[float16-r3-n2-g2-default] | 588 | CPU kill: AssertionError | PASS |
| M07fast | 585 | test_forward_values[float32-r3-n2-g2-default] | 585 | CPU kill: AssertionError | PASS |
| M08cast | 588 | test_forward_values[float16-r3-n2-g2-default] | 588 | CPU kill: AssertionError | PASS |
| M08fast | 585 | test_forward_values[float32-r3-n2-g2-default] | 585 | CPU kill: AssertionError | PASS |
| M09cast | 588 | test_forward_values[float16-r3-n2-g2-default] | 588 | CPU kill: AssertionError | PASS |
| M09fast | 585 | test_forward_values[float32-r3-n2-g2-default] | 585 | CPU kill: AssertionError | PASS |
| M10 | 534 | test_nondivisible_refusal | 534 | CPU kill: AssertionError | PASS |
| M11 | 582 | test_forward_values[float32-r3-n2-g2-default] | 582 | CPU kill: AssertionError | PASS |
| M12 | 580 | test_forward_values[float32-r3-n2-g2-default] | 580 | CPU kill: AssertionError | PASS |
| M13 | 580 | test_forward_values[float32-r3-n2-g2-default] | 580 | CPU kill: AssertionError | PASS |
| M14 | 511 | test_constructor_defaults | default eps=0.001 bound in actual call | CPU kill: AssertionError | PASS |
| M15b | 587 | test_forward_values[float16-r3-n2-g2-default] | 587 | CPU kill: AssertionError | PASS |
| M15w | 586 | test_forward_values[float16-r3-n2-g2-default] | 586 | CPU kill: AssertionError | PASS |
| M15x | 571 | test_forward_values[float16-r3-n2-g2-default] | 571 | CPU kill: AssertionError | PASS |
| M16 | 589 | test_mixed_parameters[forward-default-float16] | 589 | CPU kill: AssertionError | PASS |
| M17b | 587 | test_distinct_parameter_dtypes[forward-w32-b16] | 587 | CPU kill: AssertionError | PASS |
| M17w | 586 | test_distinct_parameter_dtypes[forward-w16-b32] | 586 | CPU kill: AssertionError | PASS |
| M18b | 587 | test_mixed_parameters[forward-half-float16] | 587 | CPU kill: AssertionError | PASS |
| M18w | 586 | test_mixed_parameters[forward-half-float16] | 586 | CPU kill: AssertionError | PASS |
| V1 | 584 | test_distinct_parameter_dtypes[forward-w32-b16] | 584 | CPU kill: AssertionError | PASS |
| V2 | 583 | test_distinct_parameter_dtypes[forward-w16-b32] | 583 | CPU kill: AssertionError | PASS |
| V3 | 571 | test_forward_values[float16-r3-n2-g2-default] | 571 | equivalent; passes | PASS |
| V4 | 583 | test_mixed_parameters[forward-default-float16] | 583 | CPU kill: AssertionError | PASS |

Initial instrument/registration failures and their unchanged receipts are retained in initial_mutation/. Matrix61 isolates parameter rank from length; NumPy 2.4.2 already refuses numpy.bool_ in operator.index, so the explicit boolean-clause deletion is equivalent here. The replacement predicate-negation mutant, named inverted, is killed by a valid NumPy count. AMENDMENTS.md records this correction before the final gate.
