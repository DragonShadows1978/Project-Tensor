# A2 tolerance derivations (D8)

Evidence class: finite-census, first-order numerical error analysis, not a
CUDA run or a universal accuracy proof. Every A1 tolerance number is retained
exactly (`tolerance_bounds.json` compares the registration). `derive_tolerances.py`
reuses REV1 `gn_emulate.py::running_bound` without editing it and adds the four
new mixed-parameter value/gradient cases: 324 value IDs in total. Inputs are
represented source values, so their initial error is zero. No output of the
candidate enters the reference or the bound.

Prior art: Wilkinson running error analysis (1960s), Higham (1996/2002),
Accuracy and Stability of Numerical Algorithms, ch. 3, for error propagation
and gamma bounds; IEEE 754 rounding (1985/2019); REV1 graph transcription.
Literature unverified — lead to check named sources.

## Operation model and powf assumption

Let u=2^-24, gamma_k=ku/(1-ku), E(v) the absolute error bound, and |.| act
componentwise. For represented operands, E(input)=0. For a sum of m terms,
E(sum v_i) <= sum E(v_i)+gamma_(m-1) sum |v_i|. This covers any sequential
or shallower reduction tree; mean includes rounded reciprocal 1/m with
E(1/m)<=u/m and the final multiply. Broadcast/reshape/permutation are exact.
For a product, E(ab)<=|b|E(a)+|a|E(b)+u|ab|; for a sum/difference,
E(a+-b)<=E(a)+E(b)+u|a+-b|. Products of errors are discarded: these are
explicitly first-order bounds, not interval certificates. All finite cases
avoid overflow in FP32; underflow of final low precision destinations is
covered below. Singleton groups have exact mean, zero centering and dx=0;
the cancellation is used, rather than bounding two independent copies of x.

Assumption: CUDA Math API single-precision `powf` maximum error **4 ulp**,
including powers -1/2 and -3/2. **Unverified — lead to check:** CUDA 12.6 Math
API Standard Functions powf maximum ulp error (and compiler fast-math mode).
One ulp/|z| <= 2u, so 4 ulp gives relative error <=8u. This is an assumption
about both engine pow kernels, not an observed accuracy claim. For z>0,
E(z^p)<=|p|z^(p-1)E(z)+8u|z^p|. E(eps)<=u|eps| covers scalar storage.

## Forward and backward recurrences

For each group of M values:
mu=mean(x); d=x-mu; v=mean(d*d); a=v+eps; s=a^(-1/2);
h=d*s; t=h*w; y=t+b. Apply the sum/product rules at each node. In particular,
E(s)<=0.5*a^(-3/2)E(a)+8u*s. Affine multiplication and addition each have
their own rounding term; parameter upcasts are exact for represented FP16/BF16.

For the engine VJP graph (p is represented projection), q=p*w,
c=q*s, z=sum(q*d), k=-0.5*a^(-3/2), r=z*k/M,
f=2*r*d, j=mean(-f), l=mean(-c), dx=((f+j)+c)+l.
E(k) uses 1.5*a^(-5/2)E(a)+8u*a^(-3/2) before its -0.5 multiply.
Every reduction and every addition in the displayed accumulation gets the
operation rule above. dw=sum(p*h), db=sum(p) over batch/spatial axes, also with
gamma_(m-1). The executable recurrence records the worst element, operand,
ID, reference, error and exact budget, rather than introducing a fitted factor.

The FP32 compute-budget maximum is 0.907336823472800 of the backward budget, on
test_low_variance[backward-var1e-4-half-bfloat16] (dx). Remaining compute margin: 9.266317653%.
The largest forward compute ratio is 0.485682114204027. This reproduces REV1’s 0.907
worst-conditioned result and includes the new mixed configurations.

## Destination rounding and unchanged budgets

If FP32 compute has |error|<=A+R|ref|, one final destination rounding with
unit roundoff u_d and underflow half-step s_d gives
E_final <= (1+u_d)E_compute+u_d|ref|+s_d, hence the registered pair
A_d=(1+u_d)A+s_d, R_d=(1+u_d)R+u_d. FP16: u_d=2^-11, s_d=2^-25.
BF16: u_d=2^-8, s_d=2^-134. For an FP32 destination u_d=s_d=0 here because
its operation rounding is already in E_compute. Each parameter gradient uses
its own leaf dtype; it need not equal the input/output dtype.

| Destination | Phase | atol | rtol | Worst bound / budget | Margin | Worst case / operand |
|---|---|---|---|---|---|---|
| float32 | forward | 3e-05 | 3e-05 | 0.485682114204027 | 51.431788580% | `test_near_epsilon[forward-float32]` / y |
| float32 | backward | 0.0003 | 0.0003 | 0.803250678008349 | 19.674932199% | `test_near_epsilon[backward-float32]` / dx |
| float16 | forward | 3.0044450759887697e-05 | 0.0005182958984375 | 0.943650728613587 | 5.634927139% | `test_forward_values[float16-r5-n2-g2-default]` / y |
| float16 | backward | 0.00030017628669738765 | 0.000788427734375 | 0.937804853431982 | 6.219514657% | `test_low_variance[backward-var1e-4-half-float16]` / dx |
| bfloat16 | forward | 3.01171875e-05 | 0.0039363671875 | 0.992432055320783 | 0.756794468% | `test_forward_values[bfloat16-r5-n2-g2-default]` / y |
| bfloat16 | backward | 0.00030117187499999995 | 0.004207421875 | 0.981971213924931 | 1.802878608% | `test_low_variance[backward-var1e-4-half-bfloat16]` / dx |

Margins are first-order bound margins, not measured GPU errors. Even the
smallest final BF16 margin remains positive in this finite census. The bound
shows no unsound budget; none was widened. P87 HELD within these stated
assumptions. Network literature verification belongs to the lead.

The reference-only central difference checks retain their separate 2e-7
assertions: original h=2^-18 at eps=0.125 and low-variance h=2^-25. They validate
the analytic float64 oracle through independent finite computation (29 IDs,
G2 native receipt); they are not dtype error budgets or device error bounds.
The classical truncation term is h^2 sup|third derivative of F|/6 and evaluation
term is (E(F(x+h))+E(F(x-h)))/(2h). We do not promote the passing finite checks
to a uniform float64 derivative certificate. The former unexplained 7/70/20
factors are not used anywhere in the device-budget derivation.
