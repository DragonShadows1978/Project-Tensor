// DLPack consumer side of tc_signal: turn any DLPack CUDA producer (or a bare capsule) into a validated fp32 operand.
//
// Prior art: DLPack (dmlc/dlpack, 2017-; header v1.3 vendored in include/dlpack/dlpack.h) for the DLManagedTensor /
// DLManagedTensorVersioned ABI and the capsule rules ("dltensor" / "dltensor_versioned", renamed to "used_*" on
// consumption, consumer calls the deleter); the Python array API standard's data-interchange protocol
// (`__dlpack__(stream=, max_version=, copy=)`, Consortium for Python Data API Standards, 2021-2023) for the stream
// argument and the max_version/TypeError fallback, which NumPy's and PyTorch's `from_dlpack` also implement. Ours: the
// two-phase use (peek shapes for sizing without consuming, then consume and validate everything right before the launch),
// and the validation set (device, dtype, compact strides with size-1 dimensions free, byte_offset, alignment, output
// overlap, read-only/copied refusal for written operands).
#pragma once

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

struct CUevent_st;  // cudaEvent_t is `CUevent_st*` (CUDA driver_types.h); kept opaque in this header

namespace tcs {
namespace py = pybind11;

// P1T-FIX5 (round-5 review V5-01, sweep (c), the capsule-destructor path). One reference to a capsule that tc_signal
// holds in C++ (to_capsule's result). Dropping the LAST reference to a capsule that is still unconsumed ("dltensor" /
// "dltensor_versioned") runs the producer's capsule destructor, which by the DLPack protocol calls the producer's
// deleter: that drop is a release, so it goes through the release admission of dlpack_operand.cpp (admitted: dropped,
// and waited for at interpreter exit; admission closed or the interpreter finalizing: the reference is deliberately
// kept, counted, and no producer code runs). Every other drop is a plain Py_DECREF. Before FIX5 these references were
// py::object members (Peeked / Peek4) whose destructor ran the producer's destructor unadmitted, for example when a
// plan refused an operand after acquiring its capsule. Prior art: pybind11's py::object (reference ownership, Jakob
// 2015) and RAII; no prior art known to me for admitting the last-reference drop.
// P1T-FIX6 (sweep (g)): the last reference of a CONSUMED ("used_*") capsule is admitted too: its drop still runs the
// producer's capsule destructor, which is producer code (Python, for a ctypes/Cython producer) even though it only
// returns for a consumed capsule. Every last-reference drop is now admitted or kept; non-last drops are plain decrefs.
class CapsuleRef {
 public:
  CapsuleRef() = default;
  explicit CapsuleRef(py::object o) noexcept : p_(o.release().ptr()) {}
  CapsuleRef(CapsuleRef&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  CapsuleRef& operator=(CapsuleRef&& o) noexcept {
    if (this != &o) {
      drop();
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  CapsuleRef(const CapsuleRef&) = delete;
  CapsuleRef& operator=(const CapsuleRef&) = delete;
  ~CapsuleRef() { drop(); }
  py::handle handle() const noexcept { return py::handle(p_); }
  // A new reference to the object (for a returned tuple); this CapsuleRef keeps its own.
  py::object object() const { return p_ ? py::reinterpret_borrow<py::object>(p_) : py::none(); }

 private:
  void drop() noexcept;  // dlpack_operand.cpp
  PyObject* p_ = nullptr;
};

// A DLPack capsule for `obj`: `obj` itself when it is a capsule, else `obj.__dlpack__(stream=1, max_version=(1, 3)
// [, copy=False])`, falling back to `obj.__dlpack__(stream=1)` when the producer rejects the newer keywords with a
// TypeError (pre-2023 producers). `writable` adds copy=False: a written operand must alias the producer's memory.
// P1T-FIX5 (V5-01 sweep (c)): the producer's attribute lookups, __dlpack_device__ and __dlpack__ are Python callbacks
// that may release and re-take the GIL, so they run inside an admitted section that tc_signal's exit hook waits for;
// once the exit hook has closed admission they are not called at all (RuntimeError, nothing acquired). A bare capsule
// involves no callback and is returned at once.
CapsuleRef to_capsule(py::handle obj, const char* role, bool writable);

// The DLTensor inside an unconsumed capsule (the capsule keeps ownership). Raises ValueError for a consumed capsule,
// TypeError for a capsule of another kind.
const DLTensor* peek(py::handle capsule, const char* role);

// Device / dtype / layout checks shared by peek-time and consume-time validation (raises TypeError/ValueError). Also
// the F03 (P1T-FIX1) checked arithmetic: rank >= 0, a non-NULL shape array, the element count, the fp32 byte total and
// data + byte_offset + bytes must not overflow (ValueError). Returns the element count.
int64_t validate_tensor(const DLTensor* t, const char* role, int ndim);

// F02 (P1T-FIX1): a REQUIRED operand passed as None raises TypeError, before any planning, allocation or consumption.
// Optional operands (bias, scale, initial state) are the only ones a caller may pass as None.
void require_operand(py::handle obj, const char* role);

// Owner of one consumed managed tensor (legacy or versioned). Calls the producer's deleter exactly once, in reset() or
// in the destructor. It is a separate member so that a throw from Operand's constructor (after the capsule has been
// consumed) still runs it: C++ destroys fully constructed members when a constructor throws.
struct ManagedOwner {
  DLManagedTensor* legacy = nullptr;
  DLManagedTensorVersioned* versioned = nullptr;
  ManagedOwner() = default;
  ~ManagedOwner() { reset(); }
  ManagedOwner(ManagedOwner&& o) noexcept;
  ManagedOwner& operator=(ManagedOwner&& o) noexcept;
  ManagedOwner(const ManagedOwner&) = delete;
  ManagedOwner& operator=(const ManagedOwner&) = delete;
  // Calls the producer's deleter at most once. P1T-FIX2 (N4): a deleter that throws is contained (caught, counted in
  // deleter_exceptions()), never std::terminate; the pointer is cleared before the call.
  // P1T-FIX5 (V5-01): the call is an admitted release (dlpack_operand.cpp, release lifecycle): tc_signal's exit hook
  // waits for it; once the hook has closed admission (or the interpreter is finalizing) the deleter is NOT called, the
  // manager is deliberately leaked and counted (pending_stats()['deleters_leaked_after_close' / '..._finalizing']).
  void reset() noexcept;
  // F01: deliberately drop ownership WITHOUT calling the deleter (used only when tc_signal cannot prove the GPU is done
  // with the memory, e.g. a broken CUDA context: a leak is safe, a release could be a use-after-free).
  void leak() noexcept {
    legacy = nullptr;
    versioned = nullptr;
  }
  bool empty() const { return legacy == nullptr && versioned == nullptr; }
};

// One consumed DLPack operand. Owns the managed tensor (through ManagedOwner) and calls the producer's deleter exactly
// once: in release() or in the destructor, so every exception path after consumption still returns the tensor to its
// producer.
//
// Absent operands (default-constructed, for an optional None) report present() == false. Reading an absent operand's
// metadata (dim, ndim, device, numel) raises TypeError instead of dereferencing NULL (F02's second line of defence).
class Operand {
 public:
  Operand() = default;
  Operand(py::handle capsule, const char* role, int ndim, bool writable);
  ~Operand();
  Operand(Operand&& o) noexcept;
  Operand& operator=(Operand&& o) noexcept;
  Operand(const Operand&) = delete;
  Operand& operator=(const Operand&) = delete;

  bool present() const { return t_ != nullptr; }
  float* data() const { return data_; }
  int64_t dim(int i) const { return meta()->shape[i]; }
  int ndim() const { return meta()->ndim; }
  int device() const { return meta()->device.device_id; }
  int64_t numel() const;
  // Validated at construction: begin + numel * 4 does not overflow (F03).
  uintptr_t byte_begin() const { return reinterpret_cast<uintptr_t>(data_); }
  uintptr_t byte_end() const { return byte_begin() + static_cast<uintptr_t>(numel_) * sizeof(float); }
  const char* role() const { return role_; }
  void release();
  // F01: hands the consumed manager to the deferred-release list (RetainUntilComplete); the operand becomes absent.
  ManagedOwner take_owner();

 private:
  const DLTensor* meta() const;  // t_, or TypeError when absent
  ManagedOwner owner_;
  const DLTensor* t_ = nullptr;
  float* data_ = nullptr;
  int64_t numel_ = 0;
  const char* role_ = "";
};

// ------------------------------------------------------------------------------------------------ deferred release (F01)
// P1T-FIX1, review finding F01 (BLOCKING): a consumed DLPack manager may be the LAST owner of its memory (a bare capsule,
// or a producer's temporary copy). Calling its deleter right after the launches let a producer on another stream reuse
// the block before the legacy stream had read or written it (use-after-free, reproduced by addon_lifetime.py). Now
// every consumed manager of a call that reached the launch stays alive until the legacy stream has passed that call:
//   * RetainUntilComplete (constructed right before the launch, destroyed on scope exit, success OR exception after a
//     partial launch) records one CUDA event on the operand device's legacy stream and moves every operand's manager
//     into a module-level FIFO pending list together with that event;
//   * completed entries (cudaEventQuery == cudaSuccess) are released in FIFO order, on the calling host thread with
//     the GIL held, at the start of the next tc_signal call (drain_completed), in tc_signal.synchronize(), and from an
//     atexit hook (both: synchronize_and_release_all);
//   * the list is bounded (tuning.h: kPendingMaxEntries, kPendingMaxBytes); when a push exceeds a bound, the legacy
//     stream of every pending device is synchronized and the whole list is released;
//   * deleters are NEVER called from a CUDA host callback (a deleter may call CUDA, e.g. torch's caching allocator);
//   * if an event cannot be recorded, the legacy stream is synchronized instead; if that also fails (a broken context),
//     the managers are deliberately leaked: tc_signal cannot prove the GPU is done with them, and a leak is safe where a
//     release could be a use-after-free.
// A call refused BEFORE the launch (validation, shape, overlap) still releases its managers immediately: nothing was
// queued on them.
//
// Prior art: PyTorch's `Tensor.record_stream` and its CUDA caching allocator, which defers a block's reuse until CUDA
// events recorded on every stream that used it have completed, polling them with cudaEventQuery from the allocating
// host thread (c10/cuda/CUDACachingAllocator.cpp, "insert_events" / "process_events", 2017-; also its EventPool of
// reusable events); CuPy's memory pool, which keeps freed blocks stream-ordered so a block is reused only in its own
// stream's order (cupy/cuda/memory.pyx, 2017-); the DLPack consumer contract (dmlc/dlpack: the consumer calls the
// deleter when it no longer needs the tensor); the CUDA rule that host callbacks must not call CUDA (CUDA Runtime API,
// cudaLaunchHostFunc). Taken: event-fenced deferred release polled from the host, event reuse. Ours: applying it to
// consumed DLPack managers on the consumer side, the GIL-held FIFO drain at every tc_signal entry point, the
// count/bytes bound with a synchronize-and-drain fallback, and the leak-on-unfenceable rule.
//
// Construct it AFTER every validation and right BEFORE the first launch: the constructor may still throw (event
// acquisition), and then nothing has been queued. Its destructor never throws.
// P1T-FIX4: the constructor's last step takes any CUDA error already pending in the calling thread and raises it as
// pre-existing (RuntimeError, nothing launched), so every launch check after it reports only this call's own error.
//
// P1T-FIX3, round-3 review finding V3-02 (aggregate byte accounting). Every operand's own byte count is checked at
// validation (F03), but their SUM was formed later, unchecked, inside this noexcept destructor (signed overflow for
// two disjoint 2^61-1 element fp32 descriptors), and added unchecked to the pending-list total. Now:
//  * the constructor's FIRST step (before any allocation, event or launch, in every entry point, since each one
//    constructs a RetainUntilComplete right before its first launch) sums the present operands' fp32 byte counts with
//    checked addition and refuses the call with ValueError on overflow (aggregate_operand_bytes); the destructor reuses
//    that checked sum and performs no addition of its own;
//  * the pending-list total is SATURATING (note_push / note_pop in dlpack_operand.cpp), never throwing, and the bound
//    comparison stays exact below INT64_MAX and conservative (over the bound) once saturated.
class RetainUntilComplete {
 public:
  RetainUntilComplete(int device, std::initializer_list<Operand*> ops);
  ~RetainUntilComplete();
  RetainUntilComplete(const RetainUntilComplete&) = delete;
  RetainUntilComplete& operator=(const RetainUntilComplete&) = delete;

 private:
  int device_;
  std::vector<Operand*> ops_;
  int64_t bytes_ = 0;                 // V3-02: the present operands' fp32 bytes, summed with checked addition
  std::vector<ManagedOwner> owners_;  // reserved in the constructor, so the destructor never allocates for it
  CUevent_st* event_ = nullptr;       // a cudaEvent_t from the per-device pool, acquired in the constructor
};

// V3-02: the fp32 byte total of the present operands, summed with checked addition; ValueError ("... overflows 64-bit
// arithmetic; refused") when it exceeds INT64_MAX. Each operand's own count was checked at validation (F03). Called by
// the RetainUntilComplete constructor before anything is acquired or launched.
int64_t aggregate_operand_bytes(const std::vector<Operand*>& ops);

// Releases every completed pending entry (FIFO, stops at the first incomplete one). Called first thing by every
// _tc_signal entry point that takes operands. GIL held.
void drain_completed() noexcept;
// Synchronizes the legacy stream of the current device and of every device with pending entries, then releases all
// entries. P1T-FIX2 (N1): raises RuntimeError naming the CUDA error if the current-device query (cudaGetDevice) or a
// synchronization failed (that device's entries are leaked, never released), and also for any release-path failure
// recorded since the last call (a fallback sync or bound drain that failed inside a destructor, which cannot raise);
// it never reports a CUDA failure as success. N3: allocation-free up to the error message; a bad_alloc while building
// the message propagates as MemoryError.
// P1T-FIX4 (V4-04): it first TAKES the calling thread's pre-existing CUDA error (cudaGetLastError), drains exactly as
// above, and then raises RuntimeError naming that error even when every synchronization succeeded; it also raises
// (once) the failures recorded on paths that cannot raise and leak no operand (a failed caller-device restore, a failed
// event destroy; cuda_guard.h).
// P1T-FIX5: (V5-02) the recorded failures include a non-NotReady event-query failure of the next-call drain and
// (sweep d) a failed completion-event record, each reported exactly once.
// P1T-FIX6: (V6-01) it never waits for another thread (FIX5's wait for releases running on other threads deadlocked
// against caller-held locks and is removed; a release another thread already popped finishes on that thread); (V6-03)
// its own drain's failures are persisted before any message allocation and every record is cleared only once the
// Python RuntimeError exists, so a MemoryError while reporting loses nothing (the next call raises it once). Raises
// RuntimeError as a captured py::error_already_set (a std::exception). GIL held.
void synchronize_and_release_all();
// P1T-FIX5 (V5-01): the exit side of the release lifecycle, called by tc_signal's atexit hook with the GIL held. Closes
// admission for good (from here on every producer deleter call and capsule drop is deliberately leaked without entering
// Python, and every producer __dlpack__ callback is refused), then waits, GIL released and with NO time bound, until no
// admitted section (release or producer callback) of ANOTHER thread is running; sections on the calling thread's own
// stack are excluded (self-exclusion: the hook may run inside its own callback). After `first_diag_s` seconds it writes
// a stderr line, then another each time the time waited has doubled. Returns {in_flight_at_close, own_thread_in_flight,
// pending_entries_at_close, waited_us, diagnostics, overran}.
// P1T-FIX6 (V6-01): the ONLY tc_signal function that waits for another thread. Its contract (documented in
// tc_signal/__init__.py and the README): the calling (exiting) thread must not hold a lock that an admitted callback
// needs, the same rule CPython applies when it joins non-daemon threads at exit.
py::dict close_admission_and_wait(double first_diag_s);
// Counters of the pending list (tests, G4 high-water mark), plus the P1T-FIX2 counters (deleter_exceptions,
// push_alloc_failures, event_recycle_failures, release_errors, unreported_release_errors) and the P1T-FIX3 V3-02 ones
// (bytes_saturated: the byte total is pinned at INT64_MAX until the list empties; bytes_saturations: times that
// happened), and the P1T-FIX4 ones (device_restore_failures, event_destroy_failures, unreported_unraisable_errors).
py::dict pending_stats(bool reset_high_water);
// P1T-FIX2 (N3): constructs the pending-list state (it allocates) at module import, so no noexcept release path is the
// first to touch it.
void init_pending_state();
// P1T-FIX2 (N4): producer deleters that threw a C++ exception (contained in ManagedOwner::reset), process-wide.
unsigned long long deleter_exceptions() noexcept;

// TEST HOOKS (P1T-FIX2, N1/N3 regression tests only; not part of the API). Arming a bit makes the NEXT pass through
// that release-path site fail once: kAlloc* sites throw std::bad_alloc exactly where the real allocation would
// (simulating an allocation failure there), kCuda* sites report cudaErrorLaunchFailure instead of calling CUDA. The
// production cost is one relaxed atomic load per site. Prior art: fault injection at named sites (e.g. the Linux
// kernel's fault-injection framework, failslab / fail_page_alloc; SQLite's OOM test harness, which fails the Nth
// malloc). Taken: arm-once named failure points. Ours: the site list.
namespace test_hooks {
enum : unsigned {
  kAllocEventPoolReserve = 1u << 0,  // acquire_event: reserving the event's recycle slot (constructor path)
  kAllocEventRecycle = 1u << 1,      // recycle_event: returning an event to the pool (release path)
  kAllocPendingPush = 1u << 2,       // ~RetainUntilComplete: pushing the entry onto the pending list (release path)
  kAllocOwnerReserve = 1u << 3,      // RetainUntilComplete: reserving the owner vector (constructor path)
  kAllocErrorMessage = 1u << 4,      // synchronize_and_release_all: building the error message
  kCudaEventRecord = 1u << 5,        // ~RetainUntilComplete: cudaEventRecord (-> fallback sync)
  kCudaSync = 1u << 6,               // any legacy-stream sync on a release path (fallback, synchronize, bound drain)
  // P1T-FIX6 (V6-03 regression tests): one-shot failures of synchronize()'s own drain.
  kCudaCurrentDevice = 1u << 7,      // sync_and_release_all_impl: cudaGetDevice
  kCudaEventQuery = 1u << 8,         // a pending entry's cudaEventQuery (synchronize's drain or the next-call drain)
};
void arm(unsigned bits) noexcept;
void clear() noexcept;
unsigned armed() noexcept;
bool take(unsigned bit) noexcept;  // true once per arming: the site must fail now
}  // namespace test_hooks

std::vector<int64_t> shape_of(const DLTensor* t);
std::string shape_str(const std::vector<int64_t>& s);

// Exact-shape check (ValueError).
void expect_shape(const Operand& op, const std::vector<int64_t>& want);
// Same CUDA ordinal for every present operand (ValueError); returns the ordinal.
int common_device(const std::vector<const Operand*>& ops);
// Byte ranges of `written` and `other` must not overlap (ValueError).
void expect_disjoint(const Operand& written, const Operand& other);

}  // namespace tcs
