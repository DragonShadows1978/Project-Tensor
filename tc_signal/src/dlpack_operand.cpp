// DLPack consumer implementation. Prior art and design notes: see dlpack_operand.h.
#include "dlpack_operand.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>  // P1T-FIX5 V5-01: exit-wait diagnostics written without the GIL
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>

#include "checked_math.h"
#include "cuda_guard.h"  // P1T-FIX4: DeviceGuard / QuietDeviceGuard and the error-slot rules
#include "tuning.h"

namespace tcs {

namespace {

constexpr const char* kCapsule = "dltensor";
constexpr const char* kCapsuleVersioned = "dltensor_versioned";
constexpr const char* kUsedCapsule = "used_dltensor";
constexpr const char* kUsedCapsuleVersioned = "used_dltensor_versioned";

std::string type_name(py::handle obj) { return Py_TYPE(obj.ptr())->tp_name; }

std::string capsule_name(py::handle cap) {
  const char* n = PyCapsule_GetName(cap.ptr());
  if (n == nullptr && PyErr_Occurred()) throw py::error_already_set();
  return n ? std::string(n) : std::string();
}

[[noreturn]] void refuse_value(const char* role, const std::string& msg) {
  throw py::value_error(std::string("tc_signal: operand '") + role + "': " + msg);
}

[[noreturn]] void refuse_type(const char* role, const std::string& msg) {
  throw py::type_error(std::string("tc_signal: operand '") + role + "': " + msg);
}

// Checks a capsule's name and returns its kind (true = versioned). Consumed and foreign capsules are refused.
bool capsule_kind(py::handle cap, const char* role) {
  const std::string n = capsule_name(cap);
  if (n == kCapsuleVersioned) return true;
  if (n == kCapsule) return false;
  if (n == kUsedCapsule || n == kUsedCapsuleVersioned)
    refuse_value(role, "the DLPack capsule was already consumed (name '" + n + "'); a capsule can be consumed once");
  refuse_type(role, "a capsule named '" + n + "' is not a DLPack capsule ('dltensor' or 'dltensor_versioned')");
}

void* capsule_pointer(py::handle cap, const char* name) {
  void* p = PyCapsule_GetPointer(cap.ptr(), name);
  if (p == nullptr) throw py::error_already_set();
  return p;
}

void check_version(const DLManagedTensorVersioned* m, const char* role) {
  if (m->version.major != DLPACK_MAJOR_VERSION) {
    std::ostringstream os;
    os << "DLPack major version " << m->version.major << " is not supported (this consumer reads major "
       << DLPACK_MAJOR_VERSION << ")";
    refuse_value(role, os.str());
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ release lifecycle (V5-01)
// P1T-FIX5, round-5 review finding V5-01 (MAJOR): interpreter exit overtook a producer deleter that was already running
// on another thread. A daemon thread's call refused a versioned read-only output; the Operand constructor ran the
// producer deleter while unwinding; the deleter released the GIL (time.sleep); the atexit hook saw an empty pending list
// and returned; CPython began finalizing; when the deleter re-took the GIL, CPython 3.12 exited the thread
// (PyEval_RestoreThread -> take_gil -> _PyThreadState_MustExit -> PyThread_exit_thread -> pthread_exit), and that forced
// unwind (abi::__forced_unwind) reached the catch-all of call_deleter_contained inside the noexcept ManagedOwner::reset:
// glibc aborts ("FATAL: exception not rethrown", rc -6). The deferred drain had the same gap (an entry is popped before
// its deleters run, so a running deleter was invisible to the pending list).
//
// The fix is an admission / closing / in-flight protocol for every add-on callback that may take the GIL, independent of
// the pending list. Two kinds of admitted section, each with an in-flight count and a per-thread depth:
//  * kRelease: one producer deleter call (ManagedOwner::reset, so every release path: the immediate release of a
//    refused call while unwinding, the next-call drain, synchronize(), the bound drain, the fallback sync of
//    ~RetainUntilComplete) and one drop of the last reference to an unconsumed DLPack capsule (CapsuleRef::drop: the
//    producer's capsule destructor calls its deleter);
//  * kCallback: the producer's Python callbacks in to_capsule (attribute lookups, __dlpack_device__, __dlpack__).
// Protocol:
//  * admit(k): if the interpreter is not accepting Python (not initialized, or finalizing: second line of defence, as
//    core's interpreter_accepts_gil_state), refuse; else increment in_flight[k], THEN read `closing`; if it is set, back
//    out (decrement, notify) and refuse; else ++t_depth[k] and run the section. leave(k): --t_depth[k], decrement,
//    notify. `closing` and the counts are sequentially consistent atomics: admit is "increment, then read closing", the
//    exit hook is "set closing, then read the counts", so at least one side sees the other (the store-buffering
//    handshake: either the releaser sees closing and backs out, or the hook sees the count and waits).
//  * A refused release is never run: the deleter is not called (the manager is leaked, counted), the capsule reference
//    is kept (counted); no producer code and no Python is entered. A refused producer callback raises RuntimeError
//    before calling it (nothing acquired).
//  * close_admission_and_wait (tc_signal's atexit hook, GIL held): set `closing` for good, then wait until no admitted
//    section of ANOTHER thread is in flight (in_flight[k] <= this thread's t_depth[k] for both kinds: self-exclusion, so
//    a hook running inside its own callback does not wait for itself). CPython 3.12.3 runs atexit callbacks before it
//    marks the runtime finalizing (Py_FinalizeEx: _PyAtExit_Call, then _PyInterpreterState_SetFinalizing /
//    _PyRuntimeState_SetFinalizing; verified by P1T-FIX2 in pylifecycle.c lines 1815 and 1832-1833), so once the hook
//    returns no admitted add-on callback is running, and every later one is refused: nothing in the add-on can take the
//    GIL after finalization begins.
//  * The wait has NO time bound (round-3 V3-01: a bounded return lets finalization overtake the callback). After a delay
//    (tc_signal._RELEASE_EXIT_DIAG_S, default 5 s) it writes one stderr line, then another each time the time waited
//    has doubled. Consequence: a producer deleter, capsule destructor or __dlpack__ that never returns makes interpreter
//    exit wait for it (the producer's contract violation; waiting is the honest outcome, an early return aborts).
//  * GIL: the waiting thread releases the GIL with PyEval_SaveThread (an admitted callback that needs the GIL can then
//    finish), waits on the condition variable with no Python object touched, drops the mutex, and only then re-takes
//    the GIL with PyEval_RestoreThread, called explicitly in a NON-noexcept frame (not from a destructor such as
//    py::gil_scoped_release's): if CPython exits a waiting daemon thread there during finalization, the forced unwind
//    travels through ordinary frames to pybind11's dispatcher, which rethrows abi::__forced_unwind (pybind11 v2.12
//    pybind11.h:1037), instead of meeting a noexcept boundary. No thread waits for the GIL while holding the mutex.
//  * P1T-FIX6, round-6 review finding V6-01 (MAJOR): FIX5 also made synchronize() (any thread) wait, without closing,
//    for releases already admitted on other threads. That wait deadlocked: thread A holds an application lock (a
//    producer's allocator lock) and calls synchronize(); thread B is releasing a refused operand whose finite producer
//    deleter needs that lock; A waits for B and B waits for A (releasing the GIL does not release A's lock; reviewer
//    probe sync_lock_inversion_cpu.py, rc 1 on acd42fa). The wait is REMOVED: synchronize() never waits for another
//    thread. A release that another thread popped off the pending list before this call is that thread's release:
//    it finishes there (it was handed to that thread's call, not to this one), and synchronize() reports and releases
//    only what is still on the list. The only waiter is the exit hook (close_admission_and_wait), which is where the
//    wait is needed: there, finalization must not start under a running callback (V5-01). The exit hook's wait is
//    documented (tc_signal/__init__.py, README): it waits for callbacks admitted before it closed admission, so the
//    exiting thread must not hold a lock such a callback needs (CPython's own exit has the same rule for the
//    non-daemon threads it joins, threading._shutdown).
//    Prior art: Coffman, Elphick & Shoshani (1971) "System Deadlocks" (circular wait is one of the four necessary
//    conditions; removing the wait from ordinary calls removes the cycle); lock-ordering discipline (never block on
//    another thread's callback while the caller may hold locks it needs); RCU (McKenney & Slingwine 1998:
//    synchronize_rcu waits only for readers that began before it) is why a separate "wait for releases" API would need a
//    pre-call generation boundary and a no-locks-held contract; none is offered (no caller needs it).
// The pending list itself is unchanged (still GIL-protected; nothing here holds a reference into it across a wait).
//
// Prior art: core's own ImportReleaseLifecycle (tensor_cuda/src/bindings.cpp, P1T-FIX2 N5 and P1T-FIX3 V3-01). Reused
// from it: the closing flag + in-flight count + condition variable, the unbounded wait with the GIL released and a
// doubling stderr diagnostic schedule, the thread-local depth for self-exclusion, "leak, never enter Python, after
// closing", and the atexit-before-finalizing argument. Different here: the add-on's releasers already hold the GIL (no
// PyGILState_Ensure), so admission is two seq_cst atomics instead of a mutex section; two section kinds (releases and
// producer callbacks); the GIL is re-taken explicitly rather than by a destructor; only the exit hook waits. Further
// prior art: CPython's threading._shutdown / wait_for_thread_shutdown (finalization waits, unbounded, for non-daemon
// threads already started); RCU grace periods (synchronize_rcu waits for every reader that entered before it, while new
// readers do not delay it; McKenney & Slingwine 1998) and the Linux percpu_ref "kill, then wait for zero"; the
// store-buffering (Dekker, Dijkstra 1965) handshake for flag + counter admission; glibc/libstdc++ forced unwinding
// (abi::__forced_unwind must be rethrown by a catch-all; a noexcept frame terminates). Ours: applying the protocol to the
// add-on's deleter, capsule-drop and producer-callback sites. Unverified here (lead to check): "CPython 3.12
// take_gil _PyThreadState_MustExit PyThread_exit_thread", "synchronize_rcu McKenney Slingwine 1998".
namespace {

enum LifecycleKind : int { kRelease = 0, kCallback = 1, kKinds = 2 };

struct Lifecycle {
  std::mutex mu;                           // only for the waiter's condition variable; admission never takes it
  std::condition_variable cv;
  std::atomic<bool> closing{false};
  std::atomic<long long> in_flight[kKinds];
  Lifecycle() {
    for (auto& c : in_flight) c.store(0);
  }
};

// Never destroyed (a callback may run during static destruction). Constructed at module import (init_pending_state).
Lifecycle& lifecycle() {
  static Lifecycle* s = new Lifecycle();
  return *s;
}

// Admitted sections on THIS thread's stack, per kind (self-exclusion; also synchronize()'s skip rule).
thread_local long long t_depth[kKinds] = {0, 0};

// Static storage: counting never allocates.
std::atomic<unsigned long long> g_admitted[kKinds];
std::atomic<unsigned long long> g_deleters_leaked_after_close{0};   // V5-01: "a late release leaks, and is counted"
std::atomic<unsigned long long> g_deleters_leaked_finalizing{0};
std::atomic<unsigned long long> g_capsules_kept_after_close{0};
std::atomic<unsigned long long> g_capsules_kept_finalizing{0};
std::atomic<unsigned long long> g_callbacks_refused{0};
std::atomic<unsigned long long> g_lifecycle_lock_failures{0};
std::atomic<unsigned long long> g_exit_hook_calls{0};
std::atomic<unsigned long long> g_exit_in_flight_at_close{0};
std::atomic<unsigned long long> g_exit_wait_us{0};
std::atomic<unsigned long long> g_exit_wait_diagnostics{0};
std::atomic<unsigned long long> g_exit_wait_overruns{0};
// P1T-FIX6 (V6-01): FIX5's g_sync_waits / g_sync_wait_us / g_sync_wait_diagnostics counted synchronize()'s wait for
// other threads; that wait is removed, and so are they (and their pending_stats keys).

bool interpreter_accepts_python() noexcept {
  if (!Py_IsInitialized()) return false;
#if PY_VERSION_HEX >= 0x030D0000
  if (Py_IsFinalizing()) return false;
#else
  if (_Py_IsFinalizing()) return false;
#endif
  return true;
}

enum class Admission { kAdmitted, kClosed, kFinalizing };

void leave_count(Lifecycle& lc, LifecycleKind k) noexcept {
  lc.in_flight[k].fetch_sub(1, std::memory_order_seq_cst);
  // Take and drop the mutex before notifying: a waiter checks its predicate while holding it, so this leave either
  // happened before that check (seen) or waits until the waiter blocks (then the notify wakes it). No lost wakeup.
  try {
    std::lock_guard<std::mutex> lk(lc.mu);
  } catch (const std::system_error&) {
    g_lifecycle_lock_failures.fetch_add(1, std::memory_order_relaxed);  // never observed; the waiter also polls
  }
  lc.cv.notify_all();
}

Admission admit(LifecycleKind k) noexcept {
  if (!interpreter_accepts_python()) return Admission::kFinalizing;
  Lifecycle& lc = lifecycle();
  lc.in_flight[k].fetch_add(1, std::memory_order_seq_cst);
  if (lc.closing.load(std::memory_order_seq_cst)) {
    leave_count(lc, k);  // back out: admission closed
    return Admission::kClosed;
  }
  ++t_depth[k];
  g_admitted[k].fetch_add(1, std::memory_order_relaxed);
  return Admission::kAdmitted;
}

void leave(LifecycleKind k) noexcept {
  --t_depth[k];
  leave_count(lifecycle(), k);
}

// RAII for the producer-callback section of to_capsule. Throws RuntimeError (before any callback) when refused.
class CallbackSection {
 public:
  explicit CallbackSection(const char* role) {
    const Admission a = admit(kCallback);
    if (a == Admission::kAdmitted) return;
    g_callbacks_refused.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error(std::string("tc_signal: operand '") + role + "': " +
                             (a == Admission::kClosed
                                  ? "interpreter exit has begun (tc_signal's exit hook closed admission)"
                                  : "the interpreter is finalizing") +
                             ", so the producer's __dlpack__ is not called; refused before acquiring anything");
  }
  ~CallbackSection() { leave(kCallback); }
  CallbackSection(const CallbackSection&) = delete;
  CallbackSection& operator=(const CallbackSection&) = delete;
};

struct WaitOutcome {
  long long running_at_start = 0;  // admitted sections of OTHER threads when the wait began
  long long own[kKinds] = {0, 0};  // this thread's own (excluded)
  long long waited_us = 0;
  unsigned long long diagnostics = 0;
  bool overran = false;            // outlasted the first-diagnostic delay
};

// Admitted sections (releases and producer callbacks) of OTHER threads: the calling thread's own are excluded.
long long others_running(const Lifecycle& lc, const long long own[kKinds]) noexcept {
  return lc.in_flight[kRelease].load(std::memory_order_seq_cst) - own[kRelease] +
         lc.in_flight[kCallback].load(std::memory_order_seq_cst) - own[kCallback];
}

// The part of the exit wait that runs with the GIL released: the mutex and condition variable only, no Python object.
void wait_without_gil(Lifecycle& lc, double first_diag_s, std::chrono::steady_clock::time_point t0,
                      WaitOutcome& w) noexcept {
  using clock = std::chrono::steady_clock;
  const auto first = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(first_diag_s));
  auto quiet = [&lc, &w] { return others_running(lc, w.own) <= 0; };
  try {
    std::unique_lock<std::mutex> lk(lc.mu);
    auto next = t0 + first;
    while (!lc.cv.wait_until(lk, next, quiet)) {
      // Still running at the diagnostic time: report, then keep waiting (never return early: V3-01, V5-01).
      w.overran = true;
      const long long running = others_running(lc, w.own);
      lk.unlock();  // a blocked stderr must not stop a callback from leaving
      const auto now = clock::now();
      std::fprintf(stderr,
                   "tc_signal: %lld DLPack producer callback(s) (deleter, capsule destructor or __dlpack__) admitted "
                   "on other threads before interpreter exit still running after %.1f s; interpreter exit waits for "
                   "them (no new one is admitted)\n",
                   running, std::chrono::duration<double>(now - t0).count());
      std::fflush(stderr);
      ++w.diagnostics;
      const auto waited = now - t0;  // next line when the time waited has doubled (at least one tick: no busy loop)
      const auto tick = clock::duration(1);
      next = now + (waited > tick ? waited : tick);
      lk.lock();
    }
  } catch (const std::system_error&) {
    // A mutex / condition-variable failure (never observed): keep waiting by polling, never return early.
    g_lifecycle_lock_failures.fetch_add(1, std::memory_order_relaxed);
    while (!quiet()) std::this_thread::sleep_for(std::chrono::milliseconds(tuning::kLifecyclePollMs));
  }
}

// The exit hook's wait (P1T-FIX6: its only caller; ordinary calls never wait for another thread, V6-01): until no
// admitted section of another thread is running. The GIL is released for the wait and re-taken explicitly here (see
// the design block above for why not by a destructor).
WaitOutcome wait_for_other_threads(double first_diag_s) {
  Lifecycle& lc = lifecycle();
  WaitOutcome w;
  w.own[kRelease] = t_depth[kRelease];
  w.own[kCallback] = t_depth[kCallback];
  w.running_at_start = others_running(lc, w.own);
  if (w.running_at_start <= 0) return w;  // nothing to wait for: the GIL is not touched
  if (!(first_diag_s >= 0.0)) first_diag_s = 0.0;   // NaN or negative: diagnose at once (the wait is the same)
  if (first_diag_s > tuning::kLifecycleMaxDiagDelayS) first_diag_s = tuning::kLifecycleMaxDiagDelayS;
  const auto t0 = std::chrono::steady_clock::now();
  const bool held = PyGILState_Check() != 0;
  PyThreadState* ts = held ? PyEval_SaveThread() : nullptr;  // GIL released: an admitted callback can finish
  wait_without_gil(lc, first_diag_s, t0, w);
  if (held) PyEval_RestoreThread(ts);  // the mutex is not held here; may exit a daemon thread during finalization
  w.waited_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  return w;
}

}  // namespace

void CapsuleRef::drop() noexcept {
  PyObject* p = p_;
  p_ = nullptr;
  if (p == nullptr) return;
  if (!interpreter_accepts_python()) {
    // Finalizing (for example a daemon thread unwinding after CPython exited it): never touch Python. Kept.
    g_capsules_kept_finalizing.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  // P1T-FIX6 (sweep (g), user Python run from a native frame): FIX5 admitted only the last reference of an UNCONSUMED
  // capsule. The last reference of a consumed ("used_*") capsule also runs the producer's PyCapsule destructor: per the
  // DLPack Python spec that destructor only returns for a consumed capsule, but it is still producer code (a ctypes or
  // Cython destructor runs Python, and may release and re-take the GIL), run from this native frame. Every
  // last-reference drop of a capsule is therefore admitted (waited for by the exit hook), or kept once admission has
  // closed (a consumed capsule kept at exit holds no producer memory: nothing is lost). A capsule with no destructor,
  // or a drop that is not the last reference, runs no producer code either way.
  if (PyCapsule_CheckExact(p) && Py_REFCNT(p) == 1) {
    // The last reference to a DLPack capsule: dropping it runs the producer's destructor (which, for an unconsumed
    // capsule, calls the producer's deleter). A release: admitted, or kept (never run) once admission has closed.
    const Admission a = admit(kRelease);
    if (a != Admission::kAdmitted) {
      (a == Admission::kClosed ? g_capsules_kept_after_close : g_capsules_kept_finalizing)
          .fetch_add(1, std::memory_order_relaxed);
      return;
    }
    Py_DECREF(p);
    leave(kRelease);
    return;
  }
  Py_DECREF(p);
}

// ------------------------------------------------------------------------------------------------ capsule acquisition
CapsuleRef to_capsule(py::handle obj, const char* role, bool writable) {
  if (PyCapsule_CheckExact(obj.ptr())) return CapsuleRef(py::reinterpret_borrow<py::object>(obj));
  // P1T-FIX5 (V5-01 sweep (c)): from here to the return, the producer's Python code may run (and release / re-take the
  // GIL): an admitted section, waited for by the exit hook; refused (RuntimeError, nothing called) after it closed.
  CallbackSection section(role);
  if (!py::hasattr(obj, "__dlpack__"))
    refuse_type(role, "expected a DLPack producer (an object with __dlpack__) or a DLPack capsule, got " +
                          type_name(obj));
  // Array API order: ask the producer's device first. A non-CUDA producer is refused here, before __dlpack__ is
  // called with a CUDA stream argument (CPU producers such as NumPy reject any stream other than None).
  if (py::hasattr(obj, "__dlpack_device__")) {
    int device_type = -1;
    try {
      py::tuple dev = obj.attr("__dlpack_device__")().cast<py::tuple>();
      device_type = dev[0].cast<int>();
    } catch (py::cast_error&) {
      refuse_type(role, "__dlpack_device__() did not return a (device_type, device_id) tuple of ints");
    }
    if (device_type != kDLCUDA) {
      std::ostringstream os;
      os << "device type " << device_type << " is not kDLCUDA (2); tc_signal takes CUDA device memory only";
      refuse_value(role, os.str());
    }
  }
  py::object fn = obj.attr("__dlpack__");
  py::object cap;
  const py::tuple max_version = py::make_tuple(tuning::kDlpackMaxMajor, tuning::kDlpackMaxMinor);
  try {
    // Consumer stream = the legacy default stream (DLPack int 1): the producer orders its pending work before it.
    if (writable)
      cap = fn(py::arg("stream") = tuning::kDlpackLegacyStream, py::arg("max_version") = max_version,
               py::arg("copy") = false);
    else
      cap = fn(py::arg("stream") = tuning::kDlpackLegacyStream, py::arg("max_version") = max_version);
  } catch (py::error_already_set& e) {
    // A producer that predates the 2023 keywords rejects them with TypeError: retry with the stream argument only
    // (the array API standard's documented fallback; NumPy and PyTorch consumers do the same).
    if (!e.matches(PyExc_TypeError)) throw;
    cap = fn(py::arg("stream") = tuning::kDlpackLegacyStream);
  }
  if (!PyCapsule_CheckExact(cap.ptr()))
    refuse_type(role, "__dlpack__ returned " + type_name(cap) + ", not a PyCapsule");
  return CapsuleRef(std::move(cap));
}

const DLTensor* peek(py::handle capsule, const char* role) {
  if (!PyCapsule_CheckExact(capsule.ptr())) refuse_type(role, "expected a DLPack capsule, got " + type_name(capsule));
  if (capsule_kind(capsule, role)) {
    auto* m = static_cast<DLManagedTensorVersioned*>(capsule_pointer(capsule, kCapsuleVersioned));
    check_version(m, role);
    return &m->dl_tensor;
  }
  auto* m = static_cast<DLManagedTensor*>(capsule_pointer(capsule, kCapsule));
  return &m->dl_tensor;
}

// ------------------------------------------------------------------------------------------------ validation
int64_t validate_tensor(const DLTensor* t, const char* role, int ndim) {
  if (t->device.device_type != kDLCUDA) {
    std::ostringstream os;
    os << "device type " << static_cast<int>(t->device.device_type)
       << " is not kDLCUDA (2); tc_signal takes CUDA device memory only";
    refuse_value(role, os.str());
  }
  if (t->dtype.code != kDLFloat || t->dtype.bits != 32 || t->dtype.lanes != 1) {
    std::ostringstream os;
    os << "dtype (code " << static_cast<int>(t->dtype.code) << ", bits " << static_cast<int>(t->dtype.bits)
       << ", lanes " << t->dtype.lanes
       << ") is not float32; tc_signal computes in fp32 only (expand fp16 weights to fp32 first)";
    refuse_type(role, os.str());
  }
  // F03 (P1T-FIX1): a negative rank or a NULL shape array (with rank > 0) is refused before the shape is read.
  if (t->ndim < 0) {
    std::ostringstream os;
    os << "negative number of dimensions " << t->ndim;
    refuse_value(role, os.str());
  }
  if (t->ndim > 0 && t->shape == nullptr) refuse_value(role, "NULL shape array for a tensor with dimensions");
  if (t->ndim != ndim) {
    std::ostringstream os;
    os << "expected " << ndim << " dimensions, got " << t->ndim;
    refuse_value(role, os.str());
  }
  // F03: checked element count and fp32 byte total, before the stride walk below multiplies extents and before any
  // later byte range is formed.
  // P1T-FIX2, review finding N2 (zero-extent rule, checked_math.h ck_extent_bound): EVERY extent is multiplied (a zero
  // or negative one counted as 1) BEFORE the empty-operand refusal, so a shape such as [0, 2^62, 4] is refused as an
  // overflow, never waved through because its count is 0. Prior art: NumPy's array constructor rule (see
  // checked_math.h); the checked builtins (GCC 5, 2015).
  int64_t numel = 1;
  for (int i = 0; i < t->ndim; ++i) {
    if (__builtin_mul_overflow(numel, t->shape[i] > 1 ? t->shape[i] : int64_t{1}, &numel))
      refuse_value(role, "element count of shape " + shape_str(shape_of(t)) +
                             " overflows int64 (every nonzero extent counts, also next to a zero extent)");
  }
  int64_t nbytes = 0;
  if (__builtin_mul_overflow(numel, static_cast<int64_t>(sizeof(float)), &nbytes))
    refuse_value(role, "byte size of shape " + shape_str(shape_of(t)) + " (fp32) overflows int64");
  for (int i = 0; i < t->ndim; ++i)
    if (t->shape[i] <= 0) refuse_value(role, "empty operand (shape " + shape_str(shape_of(t)) + ")");
  // Every extent is >= 1 from here: numel is the element count and nbytes its fp32 byte size.
  if (t->strides != nullptr) {
    // C-contiguous: every dimension of extent > 1 has the compact stride; size-1 dimensions may carry any stride
    // (they are never stepped), as the order and the DLPack/array API conventions allow.
    int64_t want = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
      if (t->shape[i] != 1 && t->strides[i] != want) {
        std::ostringstream os;
        os << "not C-contiguous (shape " << shape_str(shape_of(t)) << ", strides [";
        for (int j = 0; j < t->ndim; ++j) os << (j ? ", " : "") << t->strides[j];
        os << "]); pass a contiguous tensor";
        refuse_value(role, os.str());
      }
      want *= t->shape[i];
    }
  }
  if (t->data == nullptr) refuse_value(role, "NULL data pointer");
  // F03: data + byte_offset (element 0) and element 0 + nbytes (one past the end) must not wrap the address space.
  uintptr_t p = 0;
  uintptr_t end = 0;
  if (__builtin_add_overflow(reinterpret_cast<uintptr_t>(t->data), static_cast<uintptr_t>(t->byte_offset), &p) ||
      __builtin_add_overflow(p, static_cast<uintptr_t>(nbytes), &end))
    refuse_value(role, "data + byte_offset + byte size overflows the address space");
  if (p % alignof(float) != 0) refuse_value(role, "data + byte_offset is not 4-byte aligned");
  return numel;
}

void require_operand(py::handle obj, const char* role) {
  // F02 (P1T-FIX1): None is never a tensor. Refused with TypeError before any planning, allocation or consumption.
  if (obj.is_none())
    refuse_type(role, "is required: expected a DLPack producer (an object with __dlpack__) or a DLPack capsule, got "
                      "None (only optional operands such as a bias may be None)");
}

// ------------------------------------------------------------------------------------------------ ManagedOwner
ManagedOwner::ManagedOwner(ManagedOwner&& o) noexcept : legacy(o.legacy), versioned(o.versioned) {
  o.legacy = nullptr;
  o.versioned = nullptr;
}

ManagedOwner& ManagedOwner::operator=(ManagedOwner&& o) noexcept {
  if (this != &o) {
    reset();
    legacy = o.legacy;
    versioned = o.versioned;
    o.legacy = nullptr;
    o.versioned = nullptr;
  }
  return *this;
}

// P1T-FIX2, review finding N4: a producer deleter that throws a C++ exception must not terminate the process. Every
// deleter call in tc_signal goes through reset() (the immediate release of a refused call, the deferred drain, the
// bound drain, the fallback sync, synchronize() and the atexit hook all end here), and reset() is noexcept, so an
// escaping exception used to reach std::terminate (round-2 probe cpu_native throw_deleter, rc -6). Now each call is
// wrapped: the exception is caught (catch-all), counted (pending_stats()['deleter_exceptions']) and dropped; the owner
// pointer is cleared BEFORE the call, so a deleter is never called twice, and the caller (a loop over an entry's
// managers) keeps going with the remaining managers. The DLPack ABI declares the deleter as a C function, so a
// throwing deleter already breaks the producer's contract; containing it is the consumer's choice, not a recovery
// (the producer's memory may or may not have been freed: tc_signal does not touch it again either way).
// Prior art: C++11 noexcept semantics ([except.terminate]: an exception escaping a noexcept function calls
// std::terminate) and the "destructors/cleanup must not throw" guideline (C++ Core Guidelines C.36/E.16, Sutter &
// Alexandrescu 2004, item 51). Taken: catch at the noexcept boundary. Ours: counting and continuing per manager.
// Known gap (not closed): a deleter that calls pthread_exit or is cancelled unwinds with abi::__forced_unwind, which
// may not be swallowed (glibc aborts); no C++ handler can contain that.
// P1T-FIX5 (V5-01): the forced unwind that round 5 produced was not the producer's: CPython exited the thread when the
// deleter re-took the GIL during finalization. Every deleter call is now an admitted release (release lifecycle above):
// the exit hook does not return while one is running, and once it has closed admission no deleter is called at all, so
// CPython never finalizes under a running deleter and no forced unwind of that origin can reach this catch-all. What
// this catch-all does with a forced unwind is unchanged: it swallows it, and glibc then aborts ("FATAL: exception not
// rethrown"); rethrowing would instead meet the noexcept of reset() (std::terminate). Either way that remains the
// producer's own pthread_exit / cancellation, a DLPack contract violation (the deleter is a C function that returns).
namespace {
std::atomic<unsigned long long> g_deleter_exceptions{0};  // static storage: counting never allocates

template <typename M>
void call_deleter_contained(M* m) noexcept {
  if (m == nullptr || m->deleter == nullptr) return;
  // P1T-FIX5 (V5-01): admitted release, or deliberately leaked (deleter never called, counted) after admission closed.
  const Admission a = admit(kRelease);
  if (a != Admission::kAdmitted) {
    (a == Admission::kClosed ? g_deleters_leaked_after_close : g_deleters_leaked_finalizing)
        .fetch_add(1, std::memory_order_relaxed);
    return;
  }
  try {
    m->deleter(m);
  } catch (...) {
    g_deleter_exceptions.fetch_add(1, std::memory_order_relaxed);
  }
  leave(kRelease);  // reached on both outcomes: an ordinary C++ exception is contained, and the section still leaves
}
}  // namespace

void ManagedOwner::reset() noexcept {
  // DLPack consumer rule: when done, call the producer's deleter (it may be NULL). Called with the GIL held (every
  // tc_signal entry point runs under the GIL), so a deleter that touches Python objects is safe. P1T-FIX5: each call
  // is an admitted release (call_deleter_contained).
  if (legacy != nullptr) {
    DLManagedTensor* m = legacy;
    legacy = nullptr;  // cleared first: never called twice, even if the deleter throws or re-enters
    call_deleter_contained(m);
  }
  if (versioned != nullptr) {
    DLManagedTensorVersioned* m = versioned;
    versioned = nullptr;
    call_deleter_contained(m);
  }
}

unsigned long long deleter_exceptions() noexcept { return g_deleter_exceptions.load(std::memory_order_relaxed); }

// ------------------------------------------------------------------------------------------------ Operand
Operand::Operand(py::handle capsule, const char* role, int ndim, bool writable) : role_(role) {
  if (!PyCapsule_CheckExact(capsule.ptr())) refuse_type(role, "expected a DLPack capsule, got " + type_name(capsule));
  const bool versioned = capsule_kind(capsule, role);
  // Consume: take the pointer, rename the capsule to "used_*" (so its destructor no longer calls the deleter), and
  // own the managed tensor from here on (owner_ calls the deleter on every path, including a throw below).
  if (versioned) {
    auto* m = static_cast<DLManagedTensorVersioned*>(capsule_pointer(capsule, kCapsuleVersioned));
    if (PyCapsule_SetName(capsule.ptr(), kUsedCapsuleVersioned) != 0) throw py::error_already_set();
    owner_.versioned = m;
    check_version(m, role);
    if (writable && (m->flags & DLPACK_FLAG_BITMASK_READ_ONLY))
      refuse_value(role, "the producer marked this tensor read-only; tc_signal writes this operand");
    if (writable && (m->flags & DLPACK_FLAG_BITMASK_IS_COPIED))
      refuse_value(role, "the producer exported a copy; writes to it would not reach the caller's tensor");
    t_ = &m->dl_tensor;
  } else {
    auto* m = static_cast<DLManagedTensor*>(capsule_pointer(capsule, kCapsule));
    if (PyCapsule_SetName(capsule.ptr(), kUsedCapsule) != 0) throw py::error_already_set();
    owner_.legacy = m;
    t_ = &m->dl_tensor;
  }
  numel_ = validate_tensor(t_, role, ndim);
  data_ = reinterpret_cast<float*>(static_cast<char*>(t_->data) + t_->byte_offset);  // byte_offset honoured
}

Operand::~Operand() = default;

Operand::Operand(Operand&& o) noexcept
    : owner_(std::move(o.owner_)), t_(o.t_), data_(o.data_), numel_(o.numel_), role_(o.role_) {
  o.t_ = nullptr;
  o.data_ = nullptr;
  o.numel_ = 0;
}

Operand& Operand::operator=(Operand&& o) noexcept {
  if (this != &o) {
    owner_ = std::move(o.owner_);
    t_ = o.t_;
    data_ = o.data_;
    numel_ = o.numel_;
    role_ = o.role_;
    o.t_ = nullptr;
    o.data_ = nullptr;
    o.numel_ = 0;
  }
  return *this;
}

const DLTensor* Operand::meta() const {
  // F02 second line of defence: an absent (None) operand has no metadata; never dereference NULL.
  if (t_ == nullptr)
    refuse_type(role_[0] ? role_ : "?", "is absent (None); its shape/device cannot be read");
  return t_;
}

int64_t Operand::numel() const {
  meta();
  return numel_;  // checked at construction (validate_tensor)
}

void Operand::release() {
  owner_.reset();
  t_ = nullptr;
  data_ = nullptr;
  numel_ = 0;
}

ManagedOwner Operand::take_owner() {
  ManagedOwner o(std::move(owner_));
  t_ = nullptr;
  data_ = nullptr;
  numel_ = 0;
  return o;
}

// ------------------------------------------------------------------------------------------------ helpers
std::vector<int64_t> shape_of(const DLTensor* t) { return std::vector<int64_t>(t->shape, t->shape + t->ndim); }

std::string shape_str(const std::vector<int64_t>& s) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < s.size(); ++i) os << (i ? ", " : "") << s[i];
  os << "]";
  return os.str();
}

void expect_shape(const Operand& op, const std::vector<int64_t>& want) {
  std::vector<int64_t> got(op.ndim());
  for (int i = 0; i < op.ndim(); ++i) got[i] = op.dim(i);
  if (got != want) refuse_value(op.role(), "shape " + shape_str(got) + " != expected " + shape_str(want));
}

int common_device(const std::vector<const Operand*>& ops) {
  int dev = -1;
  const char* first = "";
  for (const Operand* op : ops) {
    if (op == nullptr || !op->present()) continue;
    if (dev < 0) {
      dev = op->device();
      first = op->role();
    } else if (op->device() != dev) {
      std::ostringstream os;
      os << "on CUDA device " << op->device() << " but operand '" << first << "' is on CUDA device " << dev
         << "; every operand must be on the same device";
      refuse_value(op->role(), os.str());
    }
  }
  return dev;
}

void expect_disjoint(const Operand& written, const Operand& other) {
  if (!written.present() || !other.present()) return;
  if (written.byte_begin() < other.byte_end() && other.byte_begin() < written.byte_end())
    refuse_value(written.role(), std::string("its memory overlaps operand '") + other.role() +
                                     "'; the written operand must not alias any other operand");
}

// ------------------------------------------------------------------------------------------------ deferred release (F01)
// Design and prior art: dlpack_operand.h, RetainUntilComplete. Every function here runs on the calling host thread with
// the GIL held (every _tc_signal entry point is a pybind11 function that keeps the GIL), so the pending list needs no
// lock. Deleters may run Python (a ctypes producer) and may even re-enter tc_signal; therefore an entry is always taken
// OFF the list before its deleters run, and nothing holds a reference into the list across a deleter call.
//
// P1T-FIX2 (round-2 review N1, N3) changed the release paths as follows:
//  * N3, no allocation failure terminates. The noexcept functions here (drain_completed, the RetainUntilComplete
//    destructor, sync_and_release_all_impl, recycle_event) no longer allocate, except the pending-list push in the
//    destructor, which is caught and falls back to a synchronous release (counted). sync_and_release_all_impl works on
//    the live list in place (phase 1 synchronizes the devices into a fixed-capacity set, phase 2 pops completed
//    entries), instead of a take-all std::deque swap and a std::set / std::vector of devices, which allocated inside a
//    noexcept function (libstdc++'s deque allocates even when default-constructed: the round-2 probe oom_sync, rc -6).
//    The recycle slot of every completion event is reserved when the event is created (constructor path, before any
//    launch), so recycling never allocates. The owner vector is reserved in the constructor, as before.
//  * N1, no CUDA error is reported as success: a failed current-device query (cudaGetDevice) is now an error of
//    synchronize() (it used to be skipped, so an empty list "synchronized" successfully on a machine with no usable
//    device: probe empty_sync); a stream sync that fails is an error; and a release-path failure that happens where
//    nothing can raise (the fallback sync and the bound drain inside the destructor) is recorded and raised by the next
//    synchronize() (once), and logged by the atexit hook.
// Prior art for the in-place two-phase drain: PyTorch's CUDA caching allocator processes its event list in place from
// the host (process_events: query each recorded event, stop at the first one not ready). Taken: query-and-pop in FIFO
// order. Ours: the sync-then-pop split with the per-device failure set, and leak-instead-of-release for entries whose
// device could not be synchronized. Fixed-capacity, allocation-free bookkeeping on failure paths is ordinary
// "no allocation on the error path" practice (e.g. the C++ Core Guidelines E.16 discussion); no specific prior art
// claimed for the set itself.
namespace test_hooks {
namespace {
std::atomic<unsigned> g_armed{0};
}  // namespace
void arm(unsigned bits) noexcept { g_armed.fetch_or(bits, std::memory_order_relaxed); }
void clear() noexcept { g_armed.store(0, std::memory_order_relaxed); }
unsigned armed() noexcept { return g_armed.load(std::memory_order_relaxed); }
bool take(unsigned bit) noexcept {
  if ((g_armed.load(std::memory_order_relaxed) & bit) == 0) return false;  // the production path: one relaxed load
  return (g_armed.fetch_and(~bit, std::memory_order_relaxed) & bit) != 0;
}
}  // namespace test_hooks

namespace {

// TEST HOOK: the CUDA error an armed kCuda* site reports instead of calling CUDA (tests only; never in production).
constexpr cudaError_t kTestInjectedCudaError = cudaErrorLaunchFailure;

// What one sync_and_release_all_impl call found (fixed capacity: the first kSyncReportMax failures are named, the
// rest only counted). P1T-FIX6 (V6-03): also the persisted record of synchronize() drains whose diagnostic was not
// raised yet (PendingState::drain_unreported), so `merge` and the counts are allocation-free and saturating.
struct SyncReport {
  bool query_failed = false;               // cudaGetDevice failed (N1: never a success)
  cudaError_t query_error = cudaSuccess;
  int n = 0;
  int dev[tuning::kSyncReportMax];
  cudaError_t err[tuning::kSyncReportMax];
  unsigned long long unnamed = 0;          // failures beyond the named ones
  unsigned long long leaked_managers = 0;  // managers leaked by this call (no completion proof)

  void add(int d, cudaError_t e) noexcept {
    if (n < tuning::kSyncReportMax) {
      dev[n] = d;
      err[n] = e;
      ++n;
    } else if (unnamed < ~0ULL) {
      ++unnamed;
    }
  }
  // P1T-FIX6 (V6-03): fold another report into this one (first query error kept; named failures appended up to the
  // capacity, the rest counted). Never allocates, never throws, saturating counts.
  void merge(const SyncReport& o) noexcept {
    if (o.query_failed && !query_failed) {
      query_failed = true;
      query_error = o.query_error;
    }
    for (int i = 0; i < o.n; ++i) add(o.dev[i], o.err[i]);
    unnamed = (o.unnamed > ~0ULL - unnamed) ? ~0ULL : unnamed + o.unnamed;
    leaked_managers = (o.leaked_managers > ~0ULL - leaked_managers) ? ~0ULL : leaked_managers + o.leaked_managers;
  }
  bool failed() const noexcept { return query_failed || n > 0 || unnamed > 0 || leaked_managers > 0; }
  int first_device() const noexcept { return n > 0 ? dev[0] : -1; }
  cudaError_t first_error() const noexcept { return query_failed ? query_error : (n > 0 ? err[0] : cudaErrorUnknown); }
};

struct PendingEntry {
  cudaEvent_t event = nullptr;   // recorded on `device`'s legacy stream right after the call's last launch
  int device = -1;
  int64_t bytes = 0;             // sum of the fp32 operand sizes this entry keeps alive
  std::vector<ManagedOwner> owners;
};

// Completion events of one device: `free` holds completed events for reuse; its capacity is kept >= `created` (the
// number of live events of this device), so returning an event to it never allocates (N3).
struct DeviceEvents {
  std::vector<cudaEvent_t> free;
  size_t created = 0;
};

struct PendingState {
  std::deque<PendingEntry> q;
  int64_t bytes = 0;            // V3-02: saturating sum of the queued entries' bytes (exact while !bytes_saturated)
  bool bytes_saturated = false;            // V3-02: a push overflowed; `bytes` is pinned at INT64_MAX until empty
  unsigned long long bytes_saturations = 0;  // V3-02: how many times the total saturated
  long long managers = 0;
  long long high_water_entries = 0;
  long long high_water_managers = 0;
  int64_t high_water_bytes = 0;
  unsigned long long retained_entries = 0;   // entries ever pushed
  unsigned long long released_entries = 0;
  unsigned long long released_managers = 0;
  unsigned long long bound_drains = 0;       // a push exceeded a bound -> synchronize + release everything
  unsigned long long fallback_syncs = 0;     // the event could not be recorded/queued -> the legacy stream was synced
  unsigned long long leaked_managers = 0;    // no completion proof (broken context): deliberately not released
  unsigned long long push_alloc_failures = 0;     // N3: the pending-list push failed to allocate (-> fallback sync)
  unsigned long long event_recycle_failures = 0;  // N3: an event could not be pooled (destroyed instead)
  unsigned long long release_errors = 0;          // N1: CUDA failures on the release paths (fallback sync, bound drain)
  // N1: release-path failures not yet raised by synchronize() (the first one is named; all are counted).
  unsigned long long unreported_count = 0;
  unsigned long long unreported_leaked = 0;
  int unreported_device = -1;
  cudaError_t unreported_error = cudaSuccess;
  // P1T-FIX4 (sweep): CUDA failures on paths that cannot raise and leak no operand (a restore of the caller's device,
  // an event destroy). Counted; the first is named and all are raised once by the next synchronize().
  unsigned long long device_restore_failures = 0;
  unsigned long long event_destroy_failures = 0;
  unsigned long long unraisable_unreported = 0;
  const char* unraisable_what = "";
  int unraisable_device = -1;
  cudaError_t unraisable_error = cudaSuccess;
  // P1T-FIX4 V4-04: the calling thread's pre-existing error taken by synchronize() on entry, held until its message
  // exists (so a MemoryError while building the message does not lose it: the next synchronize() raises it).
  unsigned long long prior_unreported = 0;
  cudaError_t prior_error = cudaSuccess;
  // P1T-FIX5: CUDA failures that used to be consumed from the error slot without being recorded. Each is now counted
  // here AND recorded through note_unraisable_failure before the slot entry is consumed, so the next synchronize()
  // raises it exactly once: V5-02, a non-NotReady cudaEventQuery failure in the next-call drain (FIX4 rows 47/50);
  // sweep (d), a failed completion-event record in ~RetainUntilComplete (the call then used the synchronous fallback).
  unsigned long long event_query_failures = 0;
  unsigned long long event_record_failures = 0;
  // P1T-FIX6, round-6 review finding V6-03 (MINOR): synchronize()'s OWN drain failures (the current-device query, a
  // device's stream sync, a completion-event query, managers leaked for lack of a completion proof) lived only in a local
  // SyncReport; when building the RuntimeError's message threw bad_alloc, the caller saw MemoryError and the CUDA
  // failure was gone (a one-shot failure: the next synchronize() succeeded). They are now persisted here, with no
  // allocation, BEFORE any message is built, and cleared only once the RuntimeError carrying them exists (see
  // synchronize_and_release_all). `drain_unreported_calls` counts the synchronize() calls whose drain failed and has
  // not been raised yet (more than 1 only after a message allocation failed).
  SyncReport drain_unreported;
  unsigned long long drain_unreported_calls = 0;
  std::unordered_map<int, DeviceEvents> event_pool;   // per device
};

// Never destroyed: no static destructor may call a producer deleter at process exit (Python may be gone by then).
// Constructed at module import (init_pending_state), so the first call from a noexcept release path never allocates.
PendingState& pending() {
  static PendingState* state = new PendingState();
  return *state;
}

// QuietDeviceGuard (makes `device` current for its scope, never throws) moved to cuda_guard.h in P1T-FIX4, where its
// restore is now checked (62c8d9f ignored the restoring cudaSetDevice's result).

// Synchronizes the legacy stream of `device` (switching to it and back). Never throws, never allocates.
cudaError_t sync_device_legacy(int device) noexcept {
  QuietDeviceGuard g(device);
  if (!g.ok()) return g.error();
  const cudaError_t r =
      test_hooks::take(test_hooks::kCudaSync) ? kTestInjectedCudaError : cudaStreamSynchronize(cudaStreamLegacy);
  if (r != cudaSuccess) cudaGetLastError();
  return r;
}

// N1/N3: the devices one sync_and_release_all_impl call synchronized, with each result. Fixed capacity, no allocation.
// A device beyond the capacity is still synchronized; its failure is remembered only as `spill_failed`, which makes
// every unrecorded device count as failed (conservative: its not-yet-complete entries are leaked, never released).
struct DeviceSyncSet {
  int dev[tuning::kSyncMaxDevices];
  cudaError_t err[tuning::kSyncMaxDevices];
  int n = 0;
  bool spill_failed = false;
  cudaError_t spill_error = cudaSuccess;

  void sync(int d) noexcept {
    for (int i = 0; i < n; ++i)
      if (dev[i] == d) return;
    const cudaError_t r = sync_device_legacy(d);
    if (n < tuning::kSyncMaxDevices) {
      dev[n] = d;
      err[n] = r;
      ++n;
    } else if (r != cudaSuccess) {
      spill_failed = true;
      spill_error = r;
    }
  }
  bool failed(int d) const noexcept {
    for (int i = 0; i < n; ++i)
      if (dev[i] == d) return err[i] != cudaSuccess;
    return spill_failed;
  }
};


// N1: a CUDA failure on a release path that cannot raise (inside the noexcept destructor). Raised by the next
// synchronize(), logged at exit. Never allocates.
void note_release_failure(PendingState& st, int device, cudaError_t err, unsigned long long leaked) noexcept {
  ++st.release_errors;
  if (st.unreported_count == 0) {
    st.unreported_device = device;
    st.unreported_error = err;
  }
  ++st.unreported_count;
  st.unreported_leaked += leaked;
}

// P1T-FIX4 (sweep): a failure that cannot raise where it happens and leaks no operand (see PendingState). Never
// allocates.
void note_unraisable_failure(PendingState& st, const char* what, int device, cudaError_t err) noexcept {
  if (st.unraisable_unreported == 0) {
    st.unraisable_what = what;
    st.unraisable_device = device;
    st.unraisable_error = err;
  }
  ++st.unraisable_unreported;
}

// P1T-FIX4 (sweep): an event is destroyed on its own device (62c8d9f destroyed it on whatever device was current and
// ignored the result). Prior art: PyTorch's CUDAEvent destructor, "event destruction done on creating device to avoid
// creating a CUDA context on other devices" (c10/cuda/CUDAEvent.h, torch 2.11 headers: CUDAGuard, then
// C10_CUDA_CHECK_WARN(cudaEventDestroy)). Taken: destroy under the event's device, checked, never thrown. Ours: the
// count and the deferred raise by the next synchronize().
void destroy_event_on_device(PendingState& st, int device, cudaEvent_t ev) noexcept {
  QuietDeviceGuard g(device);
  const cudaError_t r = g.ok() ? cudaEventDestroy(ev) : g.error();
  if (r != cudaSuccess) {
    cudaGetLastError();  // the guard consumed its own failure already; this consumes cudaEventDestroy's
    ++st.event_destroy_failures;
    note_unraisable_failure(st, "cudaEventDestroy", device, r);
  }
}

void recycle_event(PendingState& st, int device, cudaEvent_t ev) noexcept {
  if (ev == nullptr) return;
  DeviceEvents* de = nullptr;
  try {
    auto it = st.event_pool.find(device);  // find, never operator[]: no insertion on the release path
    if (it != st.event_pool.end()) de = &it->second;
    // N3: the slot was reserved when the event was created; refuse (never allocate) if it somehow is not there.
    if (test_hooks::take(test_hooks::kAllocEventRecycle) || de == nullptr || de->free.size() >= de->free.capacity())
      throw std::bad_alloc();
    de->free.push_back(ev);  // within capacity: no allocation
  } catch (...) {
    ++st.event_recycle_failures;
    if (de != nullptr && de->created > 0) --de->created;
    destroy_event_on_device(st, device, ev);  // P1T-FIX4: on its own device, checked
  }
}

// P1T-FIX3, round-3 review finding V3-02: the pending-list byte total is SATURATING. note_push / note_pop run inside the
// noexcept RetainUntilComplete destructor and the drain loops, so they may neither throw nor drain; an unchecked
// signed addition here was undefined behaviour for near-limit totals (review probe pending_bytes queue: 4 +
// 9223372036854775804), and a wrapped negative total could have suppressed the bound drain.
//  * Push: checked addition; on overflow the total is pinned at INT64_MAX and marked saturated (counted). Every entry's
//    own bytes is a checked int64 sum >= 0 (aggregate_operand_bytes, before the launch), so only the queue total can
//    overflow.
//  * Pop: exact subtraction while not saturated (the total is then the exact sum of the queued entries, so it never
//    goes below 0); while saturated the total stays pinned at INT64_MAX until the list is empty, where the true total is
//    0 again. A pinned total over-reports, never under-reports, so the bound comparison (pending_over_bound) is exact
//    below INT64_MAX and conservative above it: a push that would overflow always drains, as the exact arithmetic would
//    have (any total above INT64_MAX is far above kPendingMaxBytes).
// Chosen over drain-before-add because these functions are pure accounting: draining runs producer deleters (Python,
// possibly re-entering tc_signal) and cannot run in the middle of a push; and a drain cannot guarantee headroom anyway
// (entries pushed by a re-entrant call during the drain stay queued), so a non-throwing, non-overflowing rule is needed
// in any case. Saturation is that rule; the drain itself is left to the bound comparison right after the push.
// Prior art: saturating arithmetic (DSP/SIMD saturated add, e.g. Intel MMX/SSE PADDS, 1997-; ARM QADD) and the checked
// builtins it is built from (GCC 5 __builtin_add_overflow, 2015; C23 ckd_add); SEI CERT INT32-C. Taken: checked
// add with a saturated result. Ours: the sticky saturated state that is cleared only when the list empties, so pops
// never make a saturated total under-report. Unverified here (lead to check): "PADDSW saturating add MMX 1997".
void note_push(PendingState& st, const PendingEntry& e) noexcept {
  int64_t sum = 0;
  if (!st.bytes_saturated && __builtin_add_overflow(st.bytes, e.bytes, &sum)) {
    st.bytes_saturated = true;
    ++st.bytes_saturations;
  }
  st.bytes = st.bytes_saturated ? std::numeric_limits<int64_t>::max() : sum;
  st.managers += static_cast<long long>(e.owners.size());
  ++st.retained_entries;
  st.high_water_entries = std::max<long long>(st.high_water_entries, static_cast<long long>(st.q.size()));
  st.high_water_managers = std::max(st.high_water_managers, st.managers);
  st.high_water_bytes = std::max(st.high_water_bytes, st.bytes);
}

// Called right after the entry was taken off the list.
void note_pop(PendingState& st, const PendingEntry& e) noexcept {
  if (st.bytes_saturated) {
    if (st.q.empty()) {  // the true total is 0 again
      st.bytes = 0;
      st.bytes_saturated = false;
    }
  } else {
    int64_t rest = 0;
    // Exact while not saturated (e.bytes <= st.bytes); the guard only keeps a corrupted total from wrapping.
    st.bytes = (__builtin_sub_overflow(st.bytes, e.bytes, &rest) || rest < 0) ? 0 : rest;
  }
  st.managers -= static_cast<long long>(e.owners.size());
}

// V3-02: the bound comparison after a push. Exact below INT64_MAX (a total of exactly kPendingMaxBytes is within the
// bound, one byte more is over it, as before); a saturated total is over it.
bool pending_over_bound(const PendingState& st) noexcept {
  return static_cast<long long>(st.q.size()) > tuning::kPendingMaxEntries || st.bytes_saturated ||
         st.bytes > tuning::kPendingMaxBytes;
}

// Runs the entry's deleters (they may run Python and re-enter tc_signal; the entry is already off the list). A throwing
// deleter is contained in ManagedOwner::reset (N4) and the loop goes on with the remaining managers.
void release_entry(PendingState& st, PendingEntry& e) noexcept {
  recycle_event(st, e.device, e.event);
  e.event = nullptr;
  ++st.released_entries;
  st.released_managers += e.owners.size();
  for (ManagedOwner& o : e.owners) o.reset();
}

// Returns the number of managers leaked.
unsigned long long leak_entry(PendingState& st, PendingEntry& e) noexcept {
  unsigned long long n = 0;
  for (ManagedOwner& o : e.owners) {
    if (!o.empty()) ++n;
    o.leak();
  }
  st.leaked_managers += n;
  // The event is not recycled either: it may belong to a broken context.
  e.event = nullptr;
  return n;
}

// Phase 1: synchronize the legacy stream of the current device and of every device with pending entries (no deleter
// runs, so iterating the live list is safe). Phase 2: pop entries in FIFO order and release each one whose event has
// completed; leak (counted) an entry with no completion proof whose device could not be synchronized or whose event
// query failed; stop at the first entry that is merely not ready on a device that DID synchronize: it was pushed after
// phase 1 by a re-entrant call (a deleter that calls tc_signal) and is left pending. Every entry is popped before its
// deleters run, so a re-entrant drain or bound drain pops each entry at most once. No allocation, never throws.
SyncReport sync_and_release_all_impl() noexcept {
  PendingState& st = pending();
  SyncReport rep;
  DeviceSyncSet set;
  int cur = -1;
  // TEST HOOK kCudaCurrentDevice (P1T-FIX6 V6-03 regression): report a one-shot failure instead of calling CUDA.
  const cudaError_t qe =
      test_hooks::take(test_hooks::kCudaCurrentDevice) ? kTestInjectedCudaError : cudaGetDevice(&cur);
  if (qe != cudaSuccess) {
    cudaGetLastError();
    rep.query_failed = true;  // N1: was silently skipped (synchronize() then "succeeded" with no device)
    rep.query_error = qe;
  } else {
    set.sync(cur);
  }
  for (const PendingEntry& e : st.q) set.sync(e.device);
  for (int i = 0; i < set.n; ++i)
    if (set.err[i] != cudaSuccess) rep.add(set.dev[i], set.err[i]);
  if (set.spill_failed) rep.add(-1, set.spill_error);

  while (!st.q.empty()) {
    // TEST HOOK kCudaEventQuery (P1T-FIX6 V6-03 regression): a one-shot query failure instead of calling CUDA.
    const cudaError_t r = test_hooks::take(test_hooks::kCudaEventQuery) ? kTestInjectedCudaError
                                                                          : cudaEventQuery(st.q.front().event);
    const int dev = st.q.front().device;
    const bool complete = (r == cudaSuccess);
    if (!complete) {
      if (r == cudaErrorNotReady) {
        if (cudaPeekAtLastError() == cudaErrorNotReady) cudaGetLastError();
        if (!set.failed(dev)) break;  // pushed after phase 1 (re-entrant call): not ours to force
      } else {
        cudaGetLastError();           // the event itself cannot be queried: no completion proof (N1: an error)
        rep.add(dev, r);
      }
    }
    PendingEntry e = std::move(st.q.front());
    st.q.pop_front();
    note_pop(st, e);
    if (complete)
      release_entry(st, e);
    else
      rep.leaked_managers += leak_entry(st, e);
  }
  return rep;
}

cudaEvent_t acquire_event(int device) {
  PendingState& st = pending();
  DeviceEvents& de = st.event_pool[device];  // may allocate: constructor path, nothing has been launched yet
  if (!de.free.empty()) {
    cudaEvent_t ev = de.free.back();
    de.free.pop_back();
    return ev;
  }
  // N3: reserve the recycle slot of the event about to be created HERE, where an allocation failure only refuses the
  // call (MemoryError, operands released at once), so returning the event to the pool later never allocates.
  if (test_hooks::take(test_hooks::kAllocEventPoolReserve)) throw std::bad_alloc();
  if (de.free.capacity() < de.created + 1) de.free.reserve(std::max<size_t>(2 * de.free.capacity(), de.created + 1));
  QuietDeviceGuard g(device);
  cudaEvent_t ev = nullptr;
  cudaError_t r = g.ok() ? cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) : g.error();
  if (r != cudaSuccess) {
    cudaGetLastError();
    std::ostringstream os;
    os << "tc_signal: cannot create the completion event on CUDA device " << device << ": " << cudaGetErrorName(r)
       << ": " << cudaGetErrorString(r);
    throw std::runtime_error(os.str());
  }
  ++de.created;
  return ev;
}

}  // namespace

void init_pending_state() {
  (void)pending();
  (void)lifecycle();  // P1T-FIX5 (V5-01): the release lifecycle too, so no release path is the first to allocate it
}

// P1T-FIX5 (round-5 review V5-03): a newly created cuBLAS handle that could not be cached is destroyed by its owner while
// an exception propagates; a failed destroy is counted here (never thrown: the owner runs during unwinding). Static
// storage: never allocates. Read through pending_stats()['cublas_handle_destroy_failures'] (and the first status).
namespace {
std::atomic<unsigned long long> g_cublas_destroy_failures{0};
std::atomic<int> g_cublas_destroy_first_status{0};
}  // namespace

void note_cublas_handle_destroy_failure(int status) noexcept {
  int expected = 0;
  g_cublas_destroy_first_status.compare_exchange_strong(expected, status, std::memory_order_relaxed);
  g_cublas_destroy_failures.fetch_add(1, std::memory_order_relaxed);
}

py::dict close_admission_and_wait(double first_diag_s) {
  Lifecycle& lc = lifecycle();
  lc.closing.store(true, std::memory_order_seq_cst);  // admission closed for good: nothing new is admitted
  const long long pending_at_close = static_cast<long long>(pending().q.size());  // never released from here on
  const WaitOutcome w = wait_for_other_threads(first_diag_s);
  g_exit_hook_calls.fetch_add(1, std::memory_order_relaxed);
  if (w.running_at_start > 0)
    g_exit_in_flight_at_close.fetch_add(static_cast<unsigned long long>(w.running_at_start), std::memory_order_relaxed);
  g_exit_wait_us.fetch_add(static_cast<unsigned long long>(w.waited_us), std::memory_order_relaxed);
  g_exit_wait_diagnostics.fetch_add(w.diagnostics, std::memory_order_relaxed);
  if (w.overran) g_exit_wait_overruns.fetch_add(1, std::memory_order_relaxed);
  py::dict d;
  d["in_flight_at_close"] = w.running_at_start > 0 ? w.running_at_start : 0;
  d["own_thread_in_flight"] = w.own[kRelease] + w.own[kCallback];
  d["pending_entries_at_close"] = pending_at_close;
  d["waited_us"] = w.waited_us;
  d["diagnostics"] = w.diagnostics;
  d["overran"] = w.overran;
  return d;
}

// P1T-FIX4 (cuda_guard.h rule 2): a restore of the caller's device that failed where nothing can raise.
void note_unraisable_device_restore_failure(int device, cudaError_t err) noexcept {
  PendingState& st = pending();
  ++st.device_restore_failures;
  note_unraisable_failure(st, "restoring the caller's CUDA device (cudaSetDevice)", device, err);
}

int64_t aggregate_operand_bytes(const std::vector<Operand*>& ops) {
  // V3-02 (see dlpack_operand.h): checked sum of the present operands' fp32 bytes; ValueError on overflow (ck_add).
  // Prior art: the checked builtins (GCC 5 __builtin_add_overflow, 2015; C23 ckd_add), as in checked_math.h. Ours:
  // checking the per-call aggregate before the first launch, where a refusal can still raise.
  int64_t total = 0;
  for (const Operand* op : ops) {
    if (op == nullptr || !op->present()) continue;
    const int64_t b = op->numel() * static_cast<int64_t>(sizeof(float));  // checked at construction (F03)
    total = ck_add(total, b, "the aggregate fp32 byte count of this call's operands (their sum)");
  }
  return total;
}

RetainUntilComplete::RetainUntilComplete(int device, std::initializer_list<Operand*> ops)
    : device_(device), ops_(ops) {
  // V3-02: refuse a call whose operands' aggregate byte count overflows int64, FIRST: before any allocation, event or
  // launch (ValueError; the operands are then released at once by their own destructors, nothing was queued).
  bytes_ = aggregate_operand_bytes(ops_);
  // N3: every allocation the release needs happens here (may throw: then nothing has been launched and the operands
  // are released at once by their own destructors); the destructor below allocates only for the pending-list push.
  if (test_hooks::take(test_hooks::kAllocOwnerReserve)) throw std::bad_alloc();
  owners_.reserve(ops_.size());
  event_ = acquire_event(device_);  // if it throws, nothing else is held and nothing has been launched
  // P1T-FIX4 (sweep, CUDA error state): every launch of every entry point follows this constructor, and each launch is
  // checked by reading the calling thread's last-error slot. 62c8d9f never looked at the slot BEFORE the first launch,
  // so an error already pending there (a refused launch elsewhere in this thread, a failure some other code did not
  // consume) was reported after this call's launches as this call's own launch failure, with the work queued anyway.
  // Now it is taken here, right before the first launch, and raised as what it is: RuntimeError naming the pending
  // error, nothing launched, and the operands released at once by their own destructors (nothing was queued on them;
  // the completion event goes back to its pool). From here on a launch check can only report this call's own error.
  // Placed after acquire_event so a machine with no usable device keeps its earlier refusal ("cannot create the
  // completion event", cudaErrorNoDevice, which the runtime keeps in the slot permanently: receipt
  // artifacts/p1t_fix4/cuda_semantics.txt).
  // Prior art: CUDA 12.6 Programming Guide 3.2.12: "one has to make sure that the runtime error variable is set to
  // cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel
  // launch" (artifacts/p1t_fix4/cuda_doc_quotes.txt). Taken: that rule. Ours: raising the pending error as pre-existing
  // instead of discarding it, at the one constructor every launch path shares.
  const cudaError_t pending_err = cudaGetLastError();
  if (pending_err != cudaSuccess) {
    recycle_event(pending(), device_, event_);  // never allocates (its recycle slot was reserved by acquire_event)
    event_ = nullptr;
    std::ostringstream os;
    os << "tc_signal: a CUDA error was already pending in this thread before this call's first launch ("
       << cudaGetErrorName(pending_err) << ": " << cudaGetErrorString(pending_err)
       << "), from earlier work outside this call; taken from the error slot and reported here, nothing was launched";
    throw std::runtime_error(os.str());
  }
}

RetainUntilComplete::~RetainUntilComplete() {
  PendingState& st = pending();
  PendingEntry e;
  e.device = device_;
  e.bytes = bytes_;  // V3-02: summed with checked addition in the constructor; no addition here
  for (Operand* op : ops_) {
    if (op == nullptr || !op->present()) continue;
    owners_.push_back(op->take_owner());  // reserved: no allocation
  }
  e.owners = std::move(owners_);
  cudaError_t r = cudaSuccess;
  {
    QuietDeviceGuard g(device_);
    r = !g.ok() ? g.error()
                : (test_hooks::take(test_hooks::kCudaEventRecord) ? kTestInjectedCudaError
                                                                  : cudaEventRecord(event_, cudaStreamLegacy));
  }
  if (r == cudaSuccess) {
    e.event = event_;
    // N3: the push may allocate (a new deque node); this destructor is noexcept, so a failure must be caught here
    // and turned into the synchronous fallback below (round-2 review: bad_alloc on the release path terminated).
    try {
      if (test_hooks::take(test_hooks::kAllocPendingPush)) throw std::bad_alloc();
      st.q.push_back(std::move(e));
      note_push(st, st.q.back());
    } catch (...) {
      ++st.push_alloc_failures;
      r = cudaErrorMemoryAllocation;  // e was not moved (push_back's strong guarantee): use the fallback below
      e.event = nullptr;
    }
  } else {
    // P1T-FIX5 (sweep (d), consumed-but-unrecorded errors): the failed record (or the guard's failed switch, whose slot
    // entry the guard consumed already) is RECORDED first, then this call's slot entry is consumed. 1475fae consumed it
    // without a record, so when the fallback sync below succeeded the CUDA failure was never reported. It is a failure
    // where nothing can raise (a destructor) and no operand is leaked by it (the fallback releases or leaks on its own
    // proof), so it is an unraisable failure: raised once by the next synchronize().
    ++st.event_record_failures;
    note_unraisable_failure(st, "recording a call's completion event (cudaEventRecord on the operands' device; the "
                                "call's operands were synchronized instead)",
                            device_, r);
    cudaGetLastError();
  }
  if (r != cudaSuccess) {
    // Fallback: no queued event, so synchronize this device's legacy stream; release on success, leak on failure.
    ++st.fallback_syncs;
    const cudaError_t sr = sync_device_legacy(device_);
    if (sr == cudaSuccess) {
      recycle_event(st, device_, event_);
      ++st.released_entries;
      st.released_managers += e.owners.size();
      for (ManagedOwner& o : e.owners) o.reset();  // N4: a throwing deleter is contained per manager
    } else {
      const unsigned long long leaked = leak_entry(st, e);
      note_release_failure(st, device_, sr, leaked);  // N1: raised by the next synchronize()
    }
    return;
  }
  if (pending_over_bound(st)) {  // V3-02: overflow-safe (saturating total), exact at the boundary
    ++st.bound_drains;
    const SyncReport rep = sync_and_release_all_impl();
    if (rep.failed()) note_release_failure(st, rep.first_device(), rep.first_error(), rep.leaked_managers);  // N1
  }
}

void drain_completed() noexcept {
  PendingState& st = pending();
  while (!st.q.empty()) {
    const cudaError_t r = test_hooks::take(test_hooks::kCudaEventQuery) ? kTestInjectedCudaError
                                                                          : cudaEventQuery(st.q.front().event);
    if (r != cudaSuccess) {
      // cudaErrorNotReady: still pending (FIFO: later entries wait too). CUDA 12.6 does not leave NotReady in the
      // last-error slot (ledger receipt; Programming Guide 3.2.12: NotReady "is not considered an error"); clear it
      // defensively if a runtime ever does. Any other error (a broken context) leaves the entry for
      // synchronize_and_release_all, which re-queries it and reports and leaks it if the device cannot synchronize.
      // P1T-FIX4 (sweep): that other error's slot entry is consumed here (62c8d9f left it, so the next, unrelated
      // launch check reported it as its own); the entry itself still carries the failure to synchronize().
      // P1T-FIX5, round-5 review finding V5-02 (MINOR): FIX4's last clause was wrong for a ONE-SHOT failure. The entry
      // stays queued, but a queued entry does not remember the error: synchronize() re-queries it, and if the query now
      // succeeds the entry is released and synchronize() returns success, so the failure was consumed and never
      // reported (review probe query_error_cpu.py / query_error_single_gpu.py; FIX4 sweep rows 47 and 50). Now a
      // non-NotReady failure is RECORDED first (counted in event_query_failures and noted as a failure where nothing
      // can raise: the drain is noexcept and runs at the start of an unrelated call), and only then is the slot entry
      // consumed; the next synchronize() raises it exactly once (FIX4's deferred-error routing, cleared once its message
      // exists). Completion and release stay as before: the entry is left for synchronize(), which re-queries it and
      // releases on a successful query, or leaks it with no completion proof. NotReady is not recorded: CUDA 12.6
      // documents it as "not considered an error" (Programming Guide 3.2.12) and it is not an error of the query.
      // Prior art: NVIDIA CUDA 12.6 cudaEventQuery / cudaGetLastError contracts (a returned error vs the per-thread slot);
      // the "record before you clear" rule is ordinary error accounting, no specific prior art known to me.
      if (r == cudaErrorNotReady) {
        if (cudaPeekAtLastError() == cudaErrorNotReady) cudaGetLastError();
      } else {
        ++st.event_query_failures;
        note_unraisable_failure(st, "cudaEventQuery of a pending entry's completion event in the next-call drain (the "
                                    "entry stayed pending; synchronize() re-queried it)",
                                st.q.front().device, r);
        cudaGetLastError();
      }
      return;
    }
    PendingEntry e = std::move(st.q.front());
    st.q.pop_front();
    note_pop(st, e);
    release_entry(st, e);
  }
}

// P1T-FIX4, round-4 review finding V4-04 (MINOR). 62c8d9f never looked at the calling thread's CUDA last-error slot:
// after a synchronous launch refusal (cudaErrorInvalidConfiguration, a NULL kernel, ...) that queued no work, every
// stream sync succeeded and synchronize() returned success with the error still pending. Now:
//  * FIRST, before any query, sync or release can replace it, the existing error is TAKEN (cudaGetLastError) and held;
//  * the drain runs exactly as before (sync every relevant device's legacy stream, release what completed, leak what
//    has no completion proof); a pending launch refusal does not stop the drain;
//  * then RuntimeError is raised naming the pre-existing error (and any drain failure), even if every sync succeeded.
// Clear (take), not peek (decision justified against the CUDA semantics, receipts artifacts/p1t_fix4/):
//  * non-sticky errors (a refused launch, a failed API call) stay in the slot until someone takes them, survive
//    successful calls, and are OVERWRITTEN by the next failing call (measured 98 -> 101). This function reports the
//    error, so it consumes it: peeking would leave it for the next unrelated launch check of this shared
//    libcudart.so.12 instance (any tc_signal op, or core's cuda_check_last) to report a second time as its own; and a
//    failing sync here would overwrite it anyway, so a peek could not preserve it reliably.
//  * sticky errors (a device-side fault) live in the CONTEXT: taking the slot entry resets only the slot (measured:
//    710, then 0), every later call that touches the context still returns the error, so nothing is hidden; the drain's
//    syncs fail and the pending operands are leaked and reported, as before.
//  * the runtime's own no-device state (cudaErrorNoDevice) is not reset by cudaGetLastError at all (measured).
// Prior art: NVIDIA CUDA 12.6 cudaGetLastError / cudaPeekAtLastError ("returns this variable and resets it" versus
// "does not reset"; Programming Guide 3.2.12) and cudaStreamSynchronize; PyTorch's convention of consuming the slot when
// it reports (C10_CUDA_KERNEL_LAUNCH_CHECK = C10_CUDA_CHECK(cudaGetLastError()); C10_CUDA_CHECK_WARN, torch 2.11
// headers). Taken: report-and-consume. Ours: capturing at synchronize() entry and still draining before raising.
// P1T-FIX6, round-6 review finding V6-03 (MINOR): this call's own drain failures (rep) are persisted into
// PendingState::drain_unreported BEFORE anything is allocated for the message, with allocation-free bookkeeping (a
// fixed-capacity SyncReport merge), exactly like the earlier-failure records (N1, FIX4, FIX5) and the pre-existing error
// (V4-04). Every record is cleared only once the diagnostic has been raised, i.e. once the Python RuntimeError carrying
// the full message exists and is held by a captured py::error_already_set: a bad_alloc anywhere before that point (the
// message string, the Python str, pybind11's error capture) propagates as MemoryError and leaves every record in place,
// so the next synchronize() raises it, exactly once. (acd42fa cleared the records after building the message but
// before the exception object existed, and never persisted `rep` at all: reviewer probes release_trace_cpu.py
// diagnostic_oom_query / _sync / _device, rc 1.)
// Prior art: SQLite's OOM testing discipline (fail each allocation in turn; an error must survive the failure of its own
// reporting; "sqlite3 malloc failure testing", unverified here: lead to check "SQLite testing OOM anomaly"); ordinary
// "commit after the side effect" ordering. Taken: persist before allocating, clear after the raise. Ours: capturing
// the Python exception (PyErr_SetString, then py::error_already_set) before clearing, so pybind11's translation cannot
// lose a cleared record.
namespace {
void clear_reported_records(PendingState& st) noexcept {
  st.unreported_count = 0;
  st.unreported_leaked = 0;
  st.unreported_device = -1;
  st.unreported_error = cudaSuccess;
  st.unraisable_unreported = 0;
  st.unraisable_what = "";
  st.unraisable_device = -1;
  st.unraisable_error = cudaSuccess;
  st.prior_unreported = 0;
  st.prior_error = cudaSuccess;
  st.drain_unreported = SyncReport{};
  st.drain_unreported_calls = 0;
}
}  // namespace

void synchronize_and_release_all() {
  // P1T-FIX6 (V6-01): no wait for other threads here (FIX5's wait deadlocked against caller-held locks; see the release
  // lifecycle block). This call releases what is on the pending list, on this thread.
  PendingState& st = pending();
  const cudaError_t prior = cudaGetLastError();
  if (prior != cudaSuccess) {
    if (st.prior_unreported == 0) st.prior_error = prior;
    ++st.prior_unreported;
  }
  const SyncReport rep = sync_and_release_all_impl();
  if (rep.failed()) {  // V6-03: persisted before any allocation
    st.drain_unreported.merge(rep);
    ++st.drain_unreported_calls;
  }
  if (st.drain_unreported_calls == 0 && st.unreported_count == 0 && st.unraisable_unreported == 0 &&
      st.prior_unreported == 0)
    return;
  // N1: any CUDA failure is an error of synchronize(), never a success. N3: building the message may allocate; a
  // bad_alloc here propagates as MemoryError (an error, not a success, and not a terminate), and every record stays.
  if (test_hooks::take(test_hooks::kAllocErrorMessage)) throw std::bad_alloc();
  const SyncReport& d = st.drain_unreported;
  std::ostringstream os;
  os << "tc_signal.synchronize: CUDA error, so the release is not reported as a success:";
  const char* sep = " ";
  if (st.prior_unreported > 0) {
    os << sep << "a CUDA error was already pending in this thread when synchronize() was called ("
       << cudaGetErrorName(st.prior_error) << ": " << cudaGetErrorString(st.prior_error)
       << "), from earlier work (for example a refused kernel launch), not from the release; taken from the error slot "
          "and reported here";
    sep = "; ";
  }
  if (d.query_failed) {
    os << sep << "the current-device query cudaGetDevice failed (" << cudaGetErrorName(d.query_error) << ": "
       << cudaGetErrorString(d.query_error) << ")";
    sep = "; ";
  }
  for (int i = 0; i < d.n; ++i) {
    os << sep << "the legacy stream of device " << d.dev[i] << " could not be synchronized ("
       << cudaGetErrorName(d.err[i]) << ": " << cudaGetErrorString(d.err[i]) << ")";
    sep = "; ";
  }
  if (d.unnamed > 0) os << sep << d.unnamed << " more device failure(s)";
  if (d.leaked_managers > 0)
    os << "; " << d.leaked_managers
       << " consumed operand manager(s) still pending there were deliberately leaked, not released";
  if (st.drain_unreported_calls > 1)
    os << sep << "(these drain failures span " << st.drain_unreported_calls
       << " synchronize() calls: an earlier call could not allocate its error message and raised MemoryError)";
  if (st.unreported_count > 0) {
    os << sep << st.unreported_count << " earlier release-path failure(s) since the last synchronize, the first on device "
       << st.unreported_device << " (" << cudaGetErrorName(st.unreported_error) << ": "
       << cudaGetErrorString(st.unreported_error) << "), deliberately leaked " << st.unreported_leaked
       << " operand manager(s)";
    sep = "; ";
  }
  if (st.unraisable_unreported > 0) {
    os << sep << st.unraisable_unreported
       << " CUDA failure(s) since the last synchronize on paths that cannot raise and leak no operand, the first: "
       << st.unraisable_what << " on device " << st.unraisable_device << " (" << cudaGetErrorName(st.unraisable_error)
       << ": " << cudaGetErrorString(st.unraisable_error) << ")";
  }
  // std::ostream swallows a bad_alloc from its buffer and only sets badbit (a truncated message would follow): report
  // the allocation failure itself instead.
  if (!os) throw std::bad_alloc();
  const std::string msg = os.str();
  // V6-03: raise first, clear second. The Python RuntimeError is created here (GIL held: every caller is a binding or
  // the exit hook); if its str cannot be allocated CPython sets MemoryError instead, and the records stay.
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  if (!PyErr_ExceptionMatches(PyExc_RuntimeError)) {
    PyErr_Clear();
    throw std::bad_alloc();  // -> MemoryError; every record kept for the next synchronize()
  }
  try {
    py::error_already_set raised;  // captures the RuntimeError (may throw bad_alloc: then nothing is cleared)
    clear_reported_records(st);    // noexcept: the diagnostic exists, so its records are consumed exactly once
    throw raised;                  // copy is noexcept (shared state); pybind11 restores it as the RuntimeError
  } catch (const std::bad_alloc&) {
    PyErr_Clear();                 // the capture failed before or after fetching: no stale indicator is left behind
    throw;
  }
}

py::dict pending_stats(bool reset_high_water) {
  PendingState& st = pending();
  py::dict d;
  d["entries"] = static_cast<long long>(st.q.size());
  d["managers"] = st.managers;
  d["bytes"] = st.bytes;
  d["high_water_entries"] = st.high_water_entries;
  d["high_water_managers"] = st.high_water_managers;
  d["high_water_bytes"] = st.high_water_bytes;
  d["retained_entries"] = st.retained_entries;
  d["released_entries"] = st.released_entries;
  d["released_managers"] = st.released_managers;
  d["bound_drains"] = st.bound_drains;
  d["fallback_syncs"] = st.fallback_syncs;
  d["leaked_managers"] = st.leaked_managers;
  d["max_entries"] = tuning::kPendingMaxEntries;
  d["max_bytes"] = tuning::kPendingMaxBytes;
  // P1T-FIX2 counters.
  d["deleter_exceptions"] = deleter_exceptions();           // N4
  d["push_alloc_failures"] = st.push_alloc_failures;        // N3
  d["event_recycle_failures"] = st.event_recycle_failures;  // N3
  d["bytes_saturated"] = st.bytes_saturated;                // P1T-FIX3 V3-02
  d["bytes_saturations"] = st.bytes_saturations;            // P1T-FIX3 V3-02
  d["release_errors"] = st.release_errors;                  // N1
  d["unreported_release_errors"] = st.unreported_count;     // N1
  d["device_restore_failures"] = st.device_restore_failures;  // P1T-FIX4 (sweep): caller-device restores that failed
  d["event_destroy_failures"] = st.event_destroy_failures;    // P1T-FIX4 (sweep)
  d["unreported_unraisable_errors"] = st.unraisable_unreported;  // P1T-FIX4: raised by the next synchronize()
  // P1T-FIX5 counters. V5-01 release lifecycle: admission state, sections in flight (all threads), admitted totals,
  // releases refused after admission closed (deleters not called, capsule references kept: "a late release leaks
  // without entering Python, and is counted") or while finalizing, producer callbacks refused, the exit hook's wait and
  // mutex failures (P1T-FIX6: synchronize() no longer waits for other threads, its counters are gone). V5-02 / sweep
  // (d): recorded query and record failures.
  // V5-03: cuBLAS handles whose destroy failed while an exception propagated.
  {
    const Lifecycle& lc = lifecycle();
    d["release_admission_closed"] = lc.closing.load();
    d["releases_in_flight"] = lc.in_flight[kRelease].load();
    d["callbacks_in_flight"] = lc.in_flight[kCallback].load();
  }
  d["admitted_releases"] = g_admitted[kRelease].load();
  d["admitted_callbacks"] = g_admitted[kCallback].load();
  d["deleters_leaked_after_close"] = g_deleters_leaked_after_close.load();
  d["deleters_leaked_finalizing"] = g_deleters_leaked_finalizing.load();
  d["capsules_kept_after_close"] = g_capsules_kept_after_close.load();
  d["capsules_kept_finalizing"] = g_capsules_kept_finalizing.load();
  d["callbacks_refused_after_close"] = g_callbacks_refused.load();
  d["exit_hook_calls"] = g_exit_hook_calls.load();
  d["exit_in_flight_at_close"] = g_exit_in_flight_at_close.load();
  d["exit_wait_us"] = g_exit_wait_us.load();
  d["exit_wait_diagnostics"] = g_exit_wait_diagnostics.load();
  d["exit_wait_overruns"] = g_exit_wait_overruns.load();
  // P1T-FIX6 (V6-03): synchronize() drains whose failure is persisted but not raised yet (0 or 1 normally; more only
  // after a message allocation failed).
  d["unreported_drain_failures"] = st.drain_unreported_calls;
  d["lifecycle_lock_failures"] = g_lifecycle_lock_failures.load();
  d["event_query_failures"] = st.event_query_failures;    // V5-02
  d["event_record_failures"] = st.event_record_failures;  // sweep (d)
  d["cublas_handle_destroy_failures"] = g_cublas_destroy_failures.load();             // V5-03
  d["cublas_handle_destroy_first_status"] = g_cublas_destroy_first_status.load();     // V5-03
  if (reset_high_water) {
    st.high_water_entries = static_cast<long long>(st.q.size());
    st.high_water_managers = st.managers;
    st.high_water_bytes = st.bytes;
  }
  return d;
}

}  // namespace tcs
