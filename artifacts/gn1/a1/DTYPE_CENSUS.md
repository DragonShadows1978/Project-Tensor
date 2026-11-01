# Dtype and parameter configurations (source, collection and reasoning evidence)

| Input | Parameter leaves | Output | x gradient | weight/bias gradients | Coverage |
|---|---|---|---|---|---|
| float32 | float32 | float32 | float32 | float32 | Original 98 IDs, byte-identical math statements, no casts |
| float16 | float16 | float16 | float16 | float16 | Original census, including restored singleton backward IDs |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | Original census |
| float16 | module's untouched float32 | float16 | float16 | float32 | Mixed and low-variance, forward/backward |
| float16 | float16 via Module.half() | float16 | float16 | float16 | Mixed and low-variance, forward/backward |
| bfloat16 | module's untouched float32 | bfloat16 | bfloat16 | float32 | Mixed and low-variance, forward/backward |
| bfloat16 | float16 via Module.half() | bfloat16 | bfloat16 | float16 | Mixed and low-variance, forward/backward |
| float32 | float16 via Module.half() | float32 | float32 | float16 | Two new M17 witnesses, outside original all-FP32 lane |

Assertions are staged, not observed GPU results. Every gradient dtype must equal its own leaf's dtype, even when it differs from x. Cast backward is source-verified at ops.cpp:1060-1063. Integer/bool normalization and float64 engine computation are not supported, as established in round one's source census; NumPy float64 is reference only. Actual collected input/parameter counts are dtype_configuration_census.json. There are 100 engine IDs with FP32 inputs: 98 original FP32/FP32 IDs plus two new FP32/FP16 mixed witnesses. P85 refers to preservation of the original all-FP32 lane, not a claim that those new mixed witnesses do not exist.
