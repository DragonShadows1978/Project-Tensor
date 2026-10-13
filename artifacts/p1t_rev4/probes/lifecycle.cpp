// Prior art: CPython 3.12 lifecycle; C++17 noexcept and RAII. Taken: unchanged implementation slices; ours: CPU driver.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace py=pybind11;
std::atomic<unsigned long long> g_import_released{0};
std::atomic<unsigned long long> g_import_leaked_shutdown{0};
std::atomic<unsigned long long> g_import_leaked_closing{0};      // N5: release after the exit hook closed
std::atomic<unsigned long long> g_import_deleter_exceptions{0};  // N4
std::atomic<unsigned long long> g_import_lock_failures{0};       // N5: mutex failure (never observed) -> leak
std::atomic<unsigned long long> g_shutdown_calls{0};
std::atomic<unsigned long long> g_shutdown_in_flight_at_close{0};
std::atomic<unsigned long long> g_shutdown_timeouts{0};
std::atomic<unsigned long long> g_shutdown_still_in_flight{0};
std::atomic<unsigned long long> g_shutdown_wait_us{0};
std::atomic<unsigned long long> g_shutdown_diagnostics{0};       // V3-01: stderr lines written while waiting

// V3-01: admitted releases on the CURRENT thread's stack (incremented on admission, decremented after leaving). The
// exit hook does not wait for these: they cannot finish while the hook's thread waits (self-exclusion, see above).
thread_local long long t_admitted_depth = 0;

struct ImportReleaseLifecycle {
  std::mutex mu;
  std::condition_variable cv;
  bool closing = false;
  long long in_flight = 0;
};

// Never destroyed (a foreign thread may release after static destruction began). Constructed at module import
// (PYBIND11_MODULE below), so no release path is the first caller (its allocation could fail there).
ImportReleaseLifecycle& release_lifecycle() {
  static ImportReleaseLifecycle* s = new ImportReleaseLifecycle();
  return *s;
}

bool interpreter_accepts_gil_state() {
  if (!Py_IsInitialized()) return false;
#if PY_VERSION_HEX >= 0x030D0000
  if (Py_IsFinalizing()) return false;
#else
  if (_Py_IsFinalizing()) return false;
#endif
  return true;
}

template <typename M>
void import_release_impl(M* m) noexcept {
  if (!m || !m->deleter) return;
  ImportReleaseLifecycle& lc = release_lifecycle();
  bool admitted = false;
  try {
    std::lock_guard<std::mutex> lk(lc.mu);
    // Both checks leak without touching the GIL; the order only decides which counter records it (a release during
    // finalization keeps FIX1's leaked_interpreter_shutdown attribution).
    if (!interpreter_accepts_gil_state()) {
      g_import_leaked_shutdown.fetch_add(1);         // F04: second line of defence
    } else if (lc.closing) {
      g_import_leaked_closing.fetch_add(1);          // N5: after the exit hook closed admission: deliberate leak
    } else {
      ++lc.in_flight;
      admitted = true;
    }
  } catch (...) {
    g_import_lock_failures.fetch_add(1);
  }
  if (!admitted) return;
  ++t_admitted_depth;                                // V3-01: this thread now has an admitted release on its stack
  PyGILState_STATE g = PyGILState_Ensure();
  try {
    m->deleter(m);
    g_import_released.fetch_add(1);
  } catch (...) {
    g_import_deleter_exceptions.fetch_add(1);        // N4: contained (GIL still held while the exception dies)
  }
  PyGILState_Release(g);
  // Leave only after PyGILState_Release: the exit hook must not let finalization start while this thread still
  // touches the interpreter. notify_all on every leave (not only at zero): the hook may be waiting for a nonzero
  // count, its own thread's admitted releases (V3-01 self-exclusion).
  try {
    std::lock_guard<std::mutex> lk(lc.mu);
    --lc.in_flight;
    lc.cv.notify_all();
  } catch (...) {
    g_import_lock_failures.fetch_add(1);
  }
  --t_admitted_depth;
}

void import_release_legacy(void* ctx) noexcept {
  import_release_impl(static_cast<DLManagedTensor*>(ctx));
}
void import_release_versioned(void* ctx) noexcept {
  import_release_impl(static_cast<DLManagedTensorVersioned*>(ctx));
}

// N5 + V3-01: the atexit side (tensor_cuda/__init__.py _dlpack_shutdown_hook). Called with the GIL held by the main
// thread. Closes admission, then waits until every admitted release of OTHER threads has left (in_flight minus this
// thread's own depth == 0). There is NO bound: it never returns while one is still running (V3-01). `first_diag_s`
// (the knob tensor_cuda._DLPACK_SHUTDOWN_WAIT_S) is the time waited before the first stderr diagnostic; each later one
// is written when the time waited has doubled since the previous one (geometric schedule: k lines take 2^(k-1) times
// the first delay, so the line rate is bounded and falls as the wait goes on). NaN or negative: diagnose at once.
// Returns {in_flight_at_close, still_in_flight, timed_out, waited_us, diagnostics, own_thread_in_flight}; `timed_out`
// now means "the wait outlasted the first-diagnostic delay" (it no longer means the hook gave up), `still_in_flight`
// (other threads' admitted releases still running at return) is always 0 now, and `own_thread_in_flight` is the
// calling thread's own admitted releases that it did not wait for (0 at a real interpreter exit).
py::dict dlpack_begin_shutdown(double first_diag_s) {
  ImportReleaseLifecycle& lc = release_lifecycle();
  if (!(first_diag_s >= 0.0)) first_diag_s = 0.0;           // NaN or negative: diagnose at once (the wait is the same)
  if (first_diag_s > 86400.0) first_diag_s = 86400.0;       // keep the chrono conversion finite
  const auto first = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(first_diag_s));
  const long long own = t_admitted_depth;                   // this thread's own admitted releases: never waited for
  long long at_close = 0;
  long long remaining = 0;
  bool overran = false;
  unsigned long long diagnostics = 0;
  const auto t0 = std::chrono::steady_clock::now();
  {
    py::gil_scoped_release nogil;                           // admitted releases need the GIL to finish
    std::unique_lock<std::mutex> lk(lc.mu);
    lc.closing = true;                                      // admission closed for good: nothing new can start
    at_close = lc.in_flight;
    auto next = t0 + first;
    while (!lc.cv.wait_until(lk, next, [&lc, own] { return lc.in_flight <= own; })) {
      // Still running at the diagnostic time: report, then keep waiting (never return early: V3-01).
      overran = true;
      const long long running = lc.in_flight - own;
      lk.unlock();                                          // a blocked stderr must not stop a release from leaving
      const auto now = std::chrono::steady_clock::now();
      std::fprintf(stderr,
                   "tensor_cuda: %lld DLPack release(s) admitted before exit still running after %.1f s; interpreter "
                   "exit waits for them (no new release is admitted)\n",
                   running, std::chrono::duration<double>(now - t0).count());
      std::fflush(stderr);
      ++diagnostics;
      // Next line when the time waited has doubled (at least one clock tick later: no busy loop at a zero delay).
      const auto waited = now - t0;
      const auto tick = std::chrono::steady_clock::duration(1);
      next = now + (waited > tick ? waited : tick);
      lk.lock();
    }
    remaining = lc.in_flight - own;
  }  // lk is released before nogil re-takes the GIL
  const long long waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  g_shutdown_calls.fetch_add(1);
  g_shutdown_in_flight_at_close.fetch_add(static_cast<unsigned long long>(at_close));
  if (overran) g_shutdown_timeouts.fetch_add(1);
  g_shutdown_still_in_flight.store(static_cast<unsigned long long>(remaining));
  g_shutdown_wait_us.fetch_add(static_cast<unsigned long long>(waited_us));
  g_shutdown_diagnostics.fetch_add(diagnostics);
  py::dict d;
  d["in_flight_at_close"] = at_close;
  d["still_in_flight"] = remaining;
  d["timed_out"] = overran;
  d["waited_us"] = waited_us;
  d["diagnostics"] = diagnostics;
  d["own_thread_in_flight"] = own;
  return d;
}
static DLManagedTensor manager{};
static PyObject* callback=nullptr;
static void run_callback(DLManagedTensor*) { PyObject* r=PyObject_CallNoArgs(callback); if(!r) throw py::error_already_set(); Py_DECREF(r); }
PYBIND11_MODULE(_p1t_rev4_lifecycle,m) {
 (void)release_lifecycle();
 m.def("start",[](py::object f){callback=f.ptr();Py_INCREF(callback);manager.deleter=run_callback;std::thread([]{import_release_impl(&manager);}).detach();});
 m.def("close",&dlpack_begin_shutdown);
 m.def("stats",[]{py::dict d;auto& lc=release_lifecycle();std::lock_guard<std::mutex> l(lc.mu);d["in_flight"]=lc.in_flight;d["closing"]=lc.closing;return d;});
}
