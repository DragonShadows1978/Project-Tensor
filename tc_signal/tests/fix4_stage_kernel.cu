// P1T-FIX4 test helper: stages a REAL synchronous launch-configuration error in the calling thread's last-error slot of
// the shared libcudart.so.12 (built by test_fix4_findings.py with nvcc --cudart shared, so it is the same runtime
// instance tc_signal and core use). A block of 4096 threads exceeds every device's 1024-thread limit, so the runtime
// refuses the launch at once with cudaErrorInvalidConfiguration and queues nothing. Test-only; it is not part of the
// package, and this kernel is never executed. Prior art: none beyond the CUDA launch-configuration limits (CUDA 12.6
// Programming Guide, "Thread Hierarchy": at most 1024 threads per block).
#include <cuda_runtime.h>

__global__ void fix4_never_runs() {}

extern "C" int fix4_stage_launch_config_error() {
  cudaGetLastError();
  fix4_never_runs<<<1, 4096>>>();
  return static_cast<int>(cudaPeekAtLastError());
}
