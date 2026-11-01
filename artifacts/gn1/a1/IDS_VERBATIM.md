# Exact new/changed IDs and tolerances

All IDs below are verbatim pytest node IDs. The prefix is tensor_cuda/tests/test_groupnorm_gn1.py::. F32/F16/B16 mean the output dtype, with each gradient using its own leaf's matching dtype entry. Entries give (atol,rtol). Forward checks also execute inside backward IDs. Exact dtype strings and finite arrays are required for every engine comparison.

| Key | Forward (atol,rtol) | Backward (atol,rtol) | Derivation |
|---|---|---|---|
| F32 | (3e-5,3e-5) | (3e-4,3e-4) | Original FP32 budget, unchanged |
| F16 | (3.0044450759887697e-5,0.0005182958984375) | (0.00030017628669738765,0.000788427734375) | Represented input; FP32 budget; one rounding u=2^-11,s=2^-25 |
| B16 | (3.01171875e-5,0.0039363671875) | (0.00030117187499999995,0.004207421875) | Represented input; FP32 budget; one rounding u=2^-8,s=2^-134 |
| FD | (2e-7,2e-7) | n/a | Float64 central differences h=2^-25; truncation/subtraction derivation in REGISTRATION.md |

For F16/B16: A'= (1+u)A+s, R'=(1+u)R+u, A=R=3e-5 forward / 3e-4 backward. Table rows identify output / x-gradient / parameter-gradient keys. A dash means no gradient assertion in that forward ID. All IDs and leaf-specific numeric tolerances also appear in test_id_tolerances.json. The original 98 float32/float32 IDs are unchanged and are not listed here.

| ID | Status | Output / x grad / parameter grads |
|---|---|---|
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float16]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-bfloat16]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-float16]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-bfloat16]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g6-default]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g6-default]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-default-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-default-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-half-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-half-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-default-float16]` | new | F16 / F16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-default-bfloat16]` | new | B16 / B16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-half-float16]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-half-bfloat16]` | new | B16 / B16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-default-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-default-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-half-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-half-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-default-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-default-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-half-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-half-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-default-float16]` | new | F16 / F16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-default-bfloat16]` | new | B16 / B16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-half-float16]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-half-bfloat16]` | new | B16 / B16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-default-float16]` | new | F16 / F16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-default-bfloat16]` | new | B16 / B16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-half-float16]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-half-bfloat16]` | new | B16 / B16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_fp32_input_half_parameters[forward]` | new | F32 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_fp32_input_half_parameters[backward]` | new | F32 / F32 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-half-bfloat16]` | new | FD |
