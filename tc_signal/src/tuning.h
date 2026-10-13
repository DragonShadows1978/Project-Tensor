// tc_signal kernel tunables. House tunable law: no launch-shape literals at the use sites; every knob lives here.
// None of these changes numerics: each output element's summation order is fixed by the kernel code, not by the
// launch shape (grid-stride loops; one thread owns one output element). Changing a knob changes speed only.
#pragma once

namespace tcs {
namespace tuning {

// Threads per block for the 1-D im2col kernel (one thread per col element).
constexpr int kIm2colBlock = 256;
// Threads per block for the bias-add kernel (one thread per output element).
constexpr int kBiasBlock = 256;
// Threads per block for the gather-form ConvTranspose1d kernel (one thread per output element).
constexpr int kConvTransposeBlock = 256;
// Threads per block for the fp32 division canary kernel (fast-math probe; tests only).
constexpr int kProbeBlock = 256;
// Upper bound on blocks per launch; the kernels are grid-stride loops, so larger problems loop instead.
constexpr long long kGridCap = 1LL << 20;

// DLPack version this consumer asks producers for (`__dlpack__(max_version=...)`); the vendored header is v1.3.
constexpr int kDlpackMaxMajor = 1;
constexpr int kDlpackMaxMinor = 3;
// DLPack stream argument for the consumer stream: 1 = the legacy default stream (Python array API convention).
constexpr int kDlpackLegacyStream = 1;

}  // namespace tuning
}  // namespace tcs
