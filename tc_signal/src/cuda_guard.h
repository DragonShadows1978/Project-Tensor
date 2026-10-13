// P1T-FIX4: tc_signal's one device guard and its CUDA error-slot rules (replaces the four identical DeviceGuard copies
// of conv.cu / lstm.cu / norms.cu / small_ops.cu and dlpack_operand.cpp's QuietDeviceGuard).
//
// Why (round-4 review): each round found new members of two classes, implicit current-device assumptions (F05, V4-01)
// and unchecked or ignored CUDA returns and error state (N1, V4-03, V4-04). The FIX4 sweep found these members here:
//  * every guard IGNORED the result of the restoring cudaSetDevice (destructor), so a failed restore left the caller on
//    the operand's device silently;
//  * a raise after a failed call left that call's entry in the calling thread's last-error slot, so the next, unrelated
//    launch check (any tc_signal op, or core's, which shares this libcudart.so.12 instance) reported it again as its own.
// The rules, applied by everything below and by the callers:
//  1. Whoever reports a CUDA error consumes it: a raise after a failed CUDA call first takes the slot entry
//     (cudaGetLastError), so the error is reported exactly once. CUDA 12.6 Programming Guide 3.2.12: the runtime keeps
//     one error variable per host thread, "overwritten by the error code every time an error occurs", and
//     cudaGetLastError "returns this variable and resets it to cudaSuccess" (artifacts/p1t_fix4/cuda_doc_quotes.txt).
//  2. A restore of the caller's device is checked. On a success path it raises (DeviceGuard::restore()); where nothing
//     may raise (a destructor, or unwinding after another error) the failure is consumed, counted, and raised once by
//     the next tc_signal.synchronize() (note_unraisable_device_restore_failure, dlpack_operand.cpp).
// cuBLAS 12.6 does not import libcudart (nm -D libcublas.so.12: no undefined cuda* symbol), so a cuBLAS failure never
// touches this slot; it is reported by its cublasStatus_t only.
//
// Prior art: PyTorch c10::cuda::CUDAGuard / OptionalCUDAGuard (InlineDeviceGuard: switch in the constructor, restore in
// the destructor through CUDAGuardImpl::uncheckedSetDevice -> C10_CUDA_CHECK_WARN, which consumes the slot with
// cudaGetLastError and warns, never throws; read in the installed torch 2.11 headers, c10/cuda/impl/CUDAGuardImpl.h and
// c10/cuda/CUDAException.h); CuPy's cupy.cuda.Device context manager (__exit__ restores the previous device). Taken: the
// scoped switch-and-restore and consume-on-report. Ours: the raising restore() on success paths and routing the
// unraisable failures to the next synchronize().
#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace tcs {

// Defined in dlpack_operand.cpp: a caller-device restore that failed where nothing can raise. Counted
// (pending_stats()['device_restore_failures']) and raised once by the next synchronize(). Never allocates, never
// throws. GIL held (every tc_signal path runs under the GIL).
void note_unraisable_device_restore_failure(int device, cudaError_t err) noexcept;

// Rule 1: raise for a failed CUDA call of ours, consuming its slot entry first.
[[noreturn]] inline void raise_cuda_failure(const std::string& what, cudaError_t e) {
  cudaGetLastError();
  std::ostringstream os;
  os << what << " failed: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
  throw std::runtime_error(os.str());
}

// Makes `device` current for a launch path and restores the caller's device. The constructor raises RuntimeError (slot
// entry consumed) if the current device cannot be read or the switch fails; nothing is switched then. restore() is the
// success-path restore and raises if it fails. The destructor restores only if restore() was not reached (an exception
// is unwinding) and then routes a failure to note_unraisable_device_restore_failure (rule 2): no throw from a
// destructor.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) : device_(device) {
    cudaError_t e = cudaGetDevice(&prev_);
    if (e != cudaSuccess) raise_cuda_failure("tc_signal: cudaGetDevice", e);
    if (prev_ != device) {
      e = cudaSetDevice(device);
      if (e != cudaSuccess) raise_cuda_failure("tc_signal: cudaSetDevice(" + std::to_string(device) + ")", e);
      switched_ = true;
    }
  }
  void restore() {
    if (!switched_) return;
    switched_ = false;
    const cudaError_t e = cudaSetDevice(prev_);
    if (e != cudaSuccess)
      raise_cuda_failure("tc_signal: restoring the caller's CUDA device " + std::to_string(prev_) +
                             " after queueing work on device " + std::to_string(device_) +
                             " (cudaSetDevice; the calling thread's current device is still " +
                             std::to_string(device_) + ", and the call's operands stay retained until that work is done)",
                         e);
  }
  ~DeviceGuard() {
    if (!switched_) return;
    const cudaError_t e = cudaSetDevice(prev_);
    if (e != cudaSuccess) {
      cudaGetLastError();
      note_unraisable_device_restore_failure(prev_, e);
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int device_;
  int prev_ = -1;
  bool switched_ = false;
};

// The never-throwing variant for release paths (destructors, noexcept drains). A failed query or switch is reported
// through ok()/error() with its slot entry consumed; a failed restore follows rule 2.
class QuietDeviceGuard {
 public:
  explicit QuietDeviceGuard(int device) noexcept {
    cudaError_t e = cudaGetDevice(&prev_);
    if (e != cudaSuccess) {
      cudaGetLastError();
      err_ = e;
      return;
    }
    if (prev_ != device) {
      e = cudaSetDevice(device);
      if (e != cudaSuccess) {
        cudaGetLastError();
        err_ = e;
        return;
      }
      switched_ = true;
    }
  }
  ~QuietDeviceGuard() {
    if (!switched_) return;
    const cudaError_t e = cudaSetDevice(prev_);
    if (e != cudaSuccess) {
      cudaGetLastError();
      note_unraisable_device_restore_failure(prev_, e);
    }
  }
  QuietDeviceGuard(const QuietDeviceGuard&) = delete;
  QuietDeviceGuard& operator=(const QuietDeviceGuard&) = delete;
  bool ok() const { return err_ == cudaSuccess; }
  cudaError_t error() const { return err_; }

 private:
  int prev_ = 0;
  cudaError_t err_ = cudaSuccess;
  bool switched_ = false;
};

}  // namespace tcs
