"""P1T-M2 G3 mutant builder: copies <worktree>/tc_signal (sources only) to /tmp/p1t_m2_mut_<id>/tc_signal and applies
one registered defect per mutant. Each replacement must match exactly once (assert), and a unified diff is written next
to the copy as the receipt. Prior art: mutation testing (DeMillo, Lipton & Sayward 1978); the four defects are the
order's registered list."""
import difflib, shutil, sys
from pathlib import Path

SRC = Path(sys.argv[1])            # <worktree>/tc_signal
ADAPTER = Path(sys.argv[2])        # <worktree>/artifacts/p1t_m2/tc_signal_torch_adapter.py

SCATTER = r'''
// MUTANT (d): ConvTranspose1d as an atomicAdd scatter (the order's registered nondeterministic defect).
__global__ void mutant_convt_init_kernel(float* __restrict__ y, const float* __restrict__ b, int64_t C_out,
                                         int64_t T_out, int64_t total) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    y[i] = b != nullptr ? b[(i / T_out) % C_out] : 0.0f;
}
__global__ void mutant_convt_scatter_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                            float* __restrict__ y, int64_t N, int64_t C_in, int64_t T,
                                            int64_t C_out, int64_t K, int64_t T_out, int64_t stride,
                                            int64_t pad_begin, int64_t dilation, int64_t groups) {
  const int64_t Cg = C_in / groups;
  const int64_t Og = C_out / groups;
  const int64_t total = N * C_in * T;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = i % T;
    const int64_t r = i / T;
    const int64_t c = r % C_in;
    const int64_t n = r / C_in;
    const int64_t g = c / Cg;
    const float xv = x[i];
    for (int64_t ol = 0; ol < Og; ++ol)
      for (int64_t k = 0; k < K; ++k) {
        const int64_t p = q * stride + k * dilation - pad_begin;
        if (p < 0 || p >= T_out) continue;
        atomicAdd(&y[(n * C_out + g * Og + ol) * T_out + p], w[(c * Og + ol) * K + k] * xv);
      }
  }
}
'''

MUTANTS = {
    'a': ('im2col ignores dilation', [
        ('src/conv.cu', 'const int64_t src = t * stride - pad_begin + k * dilation;',
         'const int64_t src = t * stride - pad_begin + k;  // MUTANT (a): dilation ignored'),
    ]),
    'b': ('pad_begin and pad_end swapped (both ops)', [
        ('src/bindings.cpp', '  g.pad_begin = pb;\n  g.pad_end = pe;\n  g.dilation = dilation;\n  g.groups = groups;\n  std::ostringstream os;\n  if (g.C_in % groups != 0) {',
         '  g.pad_begin = pe;  // MUTANT (b)\n  g.pad_end = pb;  // MUTANT (b)\n  g.dilation = dilation;\n  g.groups = groups;\n  std::ostringstream os;\n  if (g.C_in % groups != 0) {'),
        ('src/bindings.cpp', '  g.pad_begin = pb;\n  g.pad_end = pe;\n  g.output_padding = output_padding;',
         '  g.pad_begin = pe;  // MUTANT (b)\n  g.pad_end = pb;  // MUTANT (b)\n  g.output_padding = output_padding;'),
    ]),
    'c': ('TF32 math on the cuBLAS handle', [
        ('src/conv.cu', 'check_cublas(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(CUBLAS_DEFAULT_MATH)");',
         'check_cublas(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(TF32)");  // MUTANT (c)'),
    ]),
    'd': ('conv_transpose1d as an atomicAdd scatter', [
        ('src/conv.cu', '}  // namespace\n\nvoid fp32_divide_probe(', SCATTER + '\n}  // namespace\n\nvoid fp32_divide_probe('),
        ('src/conv.cu', '''  const int64_t total = g.N * g.C_out * g.T_out;
  conv_transpose1d_gather_kernel<<<grid_for(total, tuning::kConvTransposeBlock), tuning::kConvTransposeBlock, 0,
                                   kLegacyStream>>>(x, w, b, y, g.N, g.C_in, g.T, g.C_out, g.K, g.T_out, g.stride,
                                                    g.pad_begin, g.dilation, g.groups);
  check_launch("conv_transpose1d_gather_kernel");''',
         '''  const int64_t total = g.N * g.C_out * g.T_out;  // MUTANT (d): init + atomicAdd scatter instead of the gather
  mutant_convt_init_kernel<<<grid_for(total, tuning::kConvTransposeBlock), tuning::kConvTransposeBlock, 0,
                             kLegacyStream>>>(y, b, g.C_out, g.T_out, total);
  check_launch("mutant_convt_init_kernel");
  const int64_t total_in = g.N * g.C_in * g.T;
  mutant_convt_scatter_kernel<<<grid_for(total_in, tuning::kConvTransposeBlock), tuning::kConvTransposeBlock, 0,
                                kLegacyStream>>>(x, w, y, g.N, g.C_in, g.T, g.C_out, g.K, g.T_out, g.stride,
                                                 g.pad_begin, g.dilation, g.groups);
  check_launch("mutant_convt_scatter_kernel");'''),
    ]),
}

for mid, (desc, edits) in MUTANTS.items():
    root = Path(f'/tmp/p1t_m2_mut_{mid}')
    dst = root / 'tc_signal'
    if dst.exists():
        raise SystemExit(f'{dst} exists; use a fresh root')
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('build', '_tc_signal*.so', '__pycache__'))
    shutil.copy2(ADAPTER, root / 'tc_signal_torch_adapter.py')
    diffs = []
    for rel, old, new in edits:
        p = dst / rel
        s = p.read_text()
        assert s.count(old) == 1, f'mutant {mid}: pattern in {rel} matched {s.count(old)} times'
        t = s.replace(old, new)
        p.write_text(t)
        diffs += difflib.unified_diff(s.splitlines(True), t.splitlines(True), f'a/tc_signal/{rel}', f'b/tc_signal/{rel}')
    (root / f'mutant_{mid}.diff').write_text(''.join(diffs))
    (root / 'MUTANT.txt').write_text(f'mutant {mid}: {desc}\n')
    print(f'mutant {mid} ({desc}): {root}  diff lines {len(diffs)}')
