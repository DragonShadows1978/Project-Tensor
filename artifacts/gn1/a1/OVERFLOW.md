# Overflow arithmetic (CPU float64 and represented-storage reasoning)

The engine's pow backward calls ew_pow(a.data(), p-1), ops.cpp:854-859; p=-.5 materializes base^-1.5 in a's dtype before multiplication. FP16 max=65504. Therefore base<65504^(-2/3)=0.0006153961657226473 exceeds its finite range. With default eps=1e-5, variance<0.0006053961657226473 is in this domain.

Nominal variance 1e-4: (1.1e-4)^(-1.5)=866784.1720414475 > 65504.
Nominal variance 1e-6: (1.1e-5)^(-1.5)=27410122.234342143 > 65504.
Singleton variance 0: (1e-5)^(-1.5)=31622776.60168379 > 65504.

The represented variances differ slightly from the decimal targets. Every new low-variance ID, its represented variance, and M15's stepwise stored var/base/power are in overflow_arithmetic.json. Default and .half() affine configuration and forward/backward do not change variance. Unique dtype/target rows follow; every group has this same variance.

| Input dtype | Target variance | Represented variance | Float64 inverse cube | M15 stored var+eps | M15 inverse cube before storage | Actual M15 FP16 overflow |
|---|---:|---:|---:|---:|---:|---|
| float16 | 0.0001 | 0.00010004272917285562 | 866279.36756096117 | 0.00011003017425537109 | 866427.64110906946 | True |
| float16 | 1e-06 | 1.000808879325632e-06 | 27407099.128423598 | 1.1026859283447266e-05 | 27310034.658887811 | True |
| bfloat16 | 0.0001 | 0.00010019540786743164 | 864479.61135957914 | 0.00011014938354492188 | 865021.48335623718 | False |
| bfloat16 | 1e-06 | 9.9890166893601418e-07 | 27414228.027062975 | 1.1026859283447266e-05 | 27310034.658887811 | False |

BF16 itself has sufficient exponent range: the >65504 comparison for BF16-represented operands is a hypothetical FP16 range statement, not a claim that BF16 overflows. All eight new FP16 forward/backward/configuration IDs have an FP16-overflowing M15 backward intermediate. Arithmetic proves that intermediate range failure; an actual test kill remains G5. D1's FP32 storage keeps these powers representable. Prior art: IEEE 754 range/rounding, round one's AMENDMENT_01 and ops.cpp mechanism; literature unverified — lead to check.
