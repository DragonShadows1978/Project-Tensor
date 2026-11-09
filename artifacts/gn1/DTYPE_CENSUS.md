# GN1 dtype census (source evidence; no GPU execution)

| Dtype | mean/var | pow and elementwise | GN1 coverage |
|---|---|---|---|
| float32 | Supported; FP32 sum/output | Supported | All forward/backward, default and nondefault eps |
| float16 | FP32 reduction accumulator, stored as FP16 before scalar divide; var is composed mean/sub/mul/mean | FP32 arithmetic, store each op as FP16; same-dtype binary operands required | Same-dtype detached leaf affine parameters; all forward, backward except the two default-eps rank-2 G=C cases, recorded before GPU in AMENDMENT_01 |
| bfloat16 | FP32 reduction accumulator, stored as BF16 | FP32 arithmetic, store each op as BF16 | Same-dtype detached leaf affine parameters; all forward/backward |
| float64 | Not an engine DType | Not supported | NumPy reference only. Public tensor(dtype="float64") silently falls back to FP32; it does not test float64 compute |
| int64, uint8, bool | Floating reduction dispatch refuses nonempty tensors | Floating pow/elementwise dispatch refuses | Not a supported normalization path; no numeric comparison claimed |

Citations: kernels.cu:568-574 DISPATCH_FLOAT; :896-913 ew_binary checks identical dtypes; :915-924 ew_scalar; :978-1017 reduce_lastdim_kernel uses float accumulation and st<T> output; :1068-1096 reduction dispatcher; :1200-1204 ew_pow uses DISPATCH_FLOAT. ops.cpp:817-822 mean=sum/count; :915-918 var=mean((x-mean)**2), explicitly population variance ddof=0; :854-859 pow backward materializes x**(p-1). bindings.cpp:1808-1811 requires_grad setter/grad accessor; :100-106 BF16 numpy export casts to FP32. __init__.py:30-38 dtype mapping, :54-61 BF16 cast construction.

Constructor affine tensors remain FP32 as required; mixed FP16/BF16 x with uncast default affine parameters is not supported by ew_binary (dtype mismatch). Tests explicitly supply matching typed parameters and check output dtype. Detach makes BF16 cast outputs genuine leaves, so backward clears no parameter gradient as an intermediate. Autograd reshape, reductions, pow and affine operations reach all three leaves; no parameter gradient is omitted.

The two omitted half backward operands are not claimed fixed or supported: eps**(-1.5) at singleton default eps is ~3.16e7, beyond FP16 max 65504; the class-only repair does not change the engine's pow backward intermediate. Half backward uses eps=0.125 there, and near-eps uses variance=eps=0.25. No GPU results motivated this source-based representability boundary.
