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

#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tcs {

// Defined in dlpack_operand.cpp: a caller-device restore that failed where nothing can raise. Counted
// (pending_stats()['device_restore_failures']) and raised once by the next synchronize(). Never allocates, never
// throws. GIL held (every tc_signal path runs under the GIL).
void note_unraisable_device_restore_failure(int device, cudaError_t err) noexcept;

// P1T-FIX5 (round-5 review V5-03), defined in dlpack_operand.cpp: a cublasDestroy of a newly created handle that failed
// while an exception propagated (the cache insertion threw), so it could not raise. Counted, first status kept
// (pending_stats()['cublas_handle_destroy_failures' / 'cublas_handle_destroy_first_status']). Never throws, never
// allocates.
void note_cublas_handle_destroy_failure(int status) noexcept;

// P1T-FIX8 (sweep (k), consumed CUDA errors): the report exists BEFORE the slot entry it reports is consumed. Every
// raise below builds its message and its std::runtime_error first; a std::bad_alloc while building them propagates
// (MemoryError) with the entry still in the calling thread's slot, so the next synchronize() (or launch check) still
// raises it, as a pre-existing error, instead of the failure being consumed and lost. Only then is the entry consumed
// (rule 1) and the exception thrown (copying a std::runtime_error is noexcept). A launch check reads the slot with
// cudaPeekAtLastError for the same reason. Prior art: FIX5 V5-02's record-before-consume rule; the CUDA 12.6 error-slot
// contract of cudaGetLastError ("returns ... and resets") / cudaPeekAtLastError ("does not reset"). Ours: applying it
// to the raising paths.
[[noreturn]] inline void throw_after_consuming(std::ostringstream& os) {
  if (!os) throw std::bad_alloc();  // ostream swallows a bad_alloc into badbit: report the allocation failure itself
  std::runtime_error report(os.str());
  cudaGetLastError();  // consumed only now that the report exists
  throw report;
}

// Rule 1: raise for a failed CUDA call of ours, consuming its slot entry once the report exists.
[[noreturn]] inline void raise_cuda_failure(const std::string& what, cudaError_t e) {
  std::ostringstream os;
  os << what << " failed: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
  throw_after_consuming(os);
}

// P1T-FIX7, sweep (j) "error attribution": the messages of the launch paths name the call that actually returned the
// error, and the device. Three cases, each with its own wording:
//  * check_call_on: a CUDA runtime call returned an error (cudaMemcpyAsync, cudaMemsetAsync, ...): that call failed (its
//    own documented contract is to return the error), so the message names it and the device it ran on.
//  * check_launch_on: the check right after a kernel launch. A <<<...>>> launch returns nothing; cudaGetLastError then
//    reports the launch's own error (a refused configuration, a missing kernel image) OR an asynchronous error of
//    earlier work on the device (CUDA 12.6 Runtime API, cudaGetLastError: "Note that this function may also return
//    error codes from previous, asynchronous launches"). FIX6 said "<kernel> failed", which is false in the second
//    case; the message now says what was checked and both possibilities.
//  * check_after_cublas_on: the same check after a cuBLAS call that returned CUBLAS_STATUS_SUCCESS. cuBLAS 12.6 imports
//    no symbol of libcudart (nm -D libcublas.so.12 / libcublasLt.so.12: no undefined cuda* symbol, ledger), so it
//    never sets this runtime's error state: an error here is an asynchronous error of earlier work on the device.
//    FIX6 said "cublasSgemmStridedBatched failed", naming a call that had just succeeded.
// Every helper consumes the slot entry it reports (rule 1). Prior art: PyTorch's C10_CUDA_KERNEL_LAUNCH_CHECK message
// ("CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be
// incorrect", c10/cuda/CUDAException.h, torch 2.11 headers; taken: stating the asynchronous possibility instead of
// blaming the last launch); the CUDA 12.6 cudaGetLastError note quoted above. Ours: the cuBLAS case and naming the
// device.
[[noreturn]] inline void raise_call_failure_on(const char* prefix, const char* what, const char* api, int device,
                                               cudaError_t e) {
  std::ostringstream os;
  os << prefix << what << " failed: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e) << " (" << api
     << " on CUDA device " << device << " returned it)";
  throw_after_consuming(os);
}
inline void check_call_on(const char* prefix, const char* what, const char* api, int device, cudaError_t e) {
  if (e != cudaSuccess) raise_call_failure_on(prefix, what, api, device, e);
}
inline void check_launch_on(const char* prefix, const char* kernel, int device) {
  const cudaError_t e = cudaPeekAtLastError();  // P1T-FIX8: read, not consumed until the report exists
  if (e == cudaSuccess) return;
  std::ostringstream os;
  os << prefix << "CUDA error at the launch check of " << kernel << " on CUDA device " << device << ": "
     << cudaGetErrorName(e) << ": " << cudaGetErrorString(e)
     << " (cudaGetLastError right after the launch: the launch's own error, or an asynchronous error of earlier work "
        "on that device, which CUDA reports at a later call)";
  throw_after_consuming(os);
}
inline void check_after_cublas_on(const char* prefix, const char* fn, int device) {
  const cudaError_t e = cudaPeekAtLastError();  // P1T-FIX8: read, not consumed until the report exists
  if (e == cudaSuccess) return;
  std::ostringstream os;
  os << prefix << "CUDA error found by the check after " << fn << " on CUDA device " << device
     << ", which returned CUBLAS_STATUS_SUCCESS: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e)
     << " (cudaGetLastError; cuBLAS does not set the CUDA runtime's error state, so this is an asynchronous error of "
        "earlier work on that device)";
  throw_after_consuming(os);
}
[[noreturn]] inline void raise_cublas_failure_on(const char* prefix, const char* fn, int device, int status) {
  std::ostringstream os;
  os << prefix << fn << " failed: cuBLAS status " << status << " (on CUDA device " << device << ")";
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
      note_unraisable_device_restore_failure(prev_, e);  // P1T-FIX8 (sweep (k)): recorded before it is consumed
      cudaGetLastError();
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
// P1T-FIX7 (V7-02 sweep (j)): failed_at_switch() says WHICH of the guard's two calls failed (cudaSetDevice of the
// target device, or the cudaGetDevice before it), so a release path records the call that failed, not the call the
// guard prevented (FIX6 reported a failed switch as a failed stream sync / event record / event destroy).
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
        switch_failed_ = true;
        return;
      }
      switched_ = true;
    }
  }
  ~QuietDeviceGuard() {
    if (!switched_) return;
    const cudaError_t e = cudaSetDevice(prev_);
    if (e != cudaSuccess) {
      note_unraisable_device_restore_failure(prev_, e);  // P1T-FIX8 (sweep (k)): recorded before it is consumed
      cudaGetLastError();
    }
  }
  QuietDeviceGuard(const QuietDeviceGuard&) = delete;
  QuietDeviceGuard& operator=(const QuietDeviceGuard&) = delete;
  bool ok() const { return err_ == cudaSuccess; }
  cudaError_t error() const { return err_; }
  bool failed_at_switch() const { return switch_failed_; }  // P1T-FIX7: false = the cudaGetDevice before it failed

 private:
  int prev_ = 0;
  cudaError_t err_ = cudaSuccess;
  bool switched_ = false;
  bool switch_failed_ = false;
};

}  // namespace tcs
