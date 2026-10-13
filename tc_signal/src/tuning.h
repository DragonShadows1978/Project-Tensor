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

// P1T-FIX1 (F01) deferred release: bounds of the pending list of consumed DLPack managers that wait for the legacy
// stream to pass their call (dlpack_operand.h, RetainUntilComplete). When a new entry pushes the list past either
// bound, tc_signal synchronizes the legacy stream of every pending device and releases the whole list. These only
// trade memory held (by producers' blocks waiting for release) against how often a call blocks; numerics never change.
// Entries: one per tc_signal call that reached its launch.
constexpr long long kPendingMaxEntries = 1024;
// Bytes: the sum of the fp32 operand sizes held by the pending entries (inputs, outputs and scratch).
constexpr long long kPendingMaxBytes = 256LL << 20;
// P1T-FIX2 (N3): fixed capacities of the allocation-free bookkeeping in synchronize/the bound drain. Devices whose sync
// result is remembered per call (a device beyond this is still synchronized; its failure is kept as one conservative
// flag), and CUDA failures named in one synchronize() error message (the rest are counted). Neither changes numerics.
constexpr int kSyncMaxDevices = 64;
constexpr int kSyncReportMax = 8;

// P1T-FIX5 (V5-01) release lifecycle (dlpack_operand.cpp). None of these bounds a wait (no wait ever returns early) and
// none changes numerics:
//  * the cap on the exit hook's first-diagnostic delay (keeps the chrono conversion finite; the delay itself is the
//    Python knob tc_signal._RELEASE_EXIT_DIAG_S);
//  * the polling interval used only if the lifecycle mutex ever fails (never observed), so the wait still never returns
//    early.
// P1T-FIX6 (V6-01): FIX5's kSyncReleaseWaitFirstDiagS is gone with synchronize()'s wait for other threads; only the
// exit hook waits.
constexpr double kLifecycleMaxDiagDelayS = 86400.0;
constexpr int kLifecyclePollMs = 10;
// P1T-FIX7 (V7-01): the default first-diagnostic delay of the exit wait, in seconds. It is the initial value of the
// Python knob tc_signal._RELEASE_EXIT_DIAG_S (exported as _tc_signal._RELEASE_EXIT_DIAG_DEFAULT_S, so the two cannot
// drift apart), and the delay the exit entry uses when the knob holds anything it cannot read as a real number without
// calling a Python method (another type, an int beyond the double range, the knob deleted). Diagnostics only: it never
// bounds or gates the wait, and it changes no numerics.
constexpr double kExitDiagDefaultS = 5.0;
// P1T-FIX7, lead send-back L1: the polling interval of the diagnosed device wait at interpreter exit (inside the exit
// runner only; dlpack_operand.cpp wait_device_work_diagnosed). It adds at most one interval of latency to each device
// sync at exit; it never bounds the wait and changes no numerics.
constexpr int kExitDevicePollMs = 1;

}  // namespace tuning
}  // namespace tcs
