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
  PyGILState_STATE g = PyGILState_Ensure();
  try {
    m->deleter(m);
    g_import_released.fetch_add(1);
  } catch (...) {
    g_import_deleter_exceptions.fetch_add(1);        // N4: contained (GIL still held while the exception dies)
  }
  PyGILState_Release(g);
  // Leave only after PyGILState_Release: the exit hook must not let finalization start while this thread still
  // touches the interpreter.
  try {
    std::lock_guard<std::mutex> lk(lc.mu);
    if (--lc.in_flight == 0) lc.cv.notify_all();
  } catch (...) {
    g_import_lock_failures.fetch_add(1);
  }
}

void import_release_legacy(void* ctx) noexcept {
  import_release_impl(static_cast<DLManagedTensor*>(ctx));
}
void import_release_versioned(void* ctx) noexcept {
  import_release_impl(static_cast<DLManagedTensorVersioned*>(ctx));
}

// N5: the atexit side (tensor_cuda/__init__.py _dlpack_shutdown_hook). Called with the GIL held by the main thread.
py::dict dlpack_begin_shutdown(double timeout_s) {
  ImportReleaseLifecycle& lc = release_lifecycle();
  if (!(timeout_s >= 0.0)) timeout_s = 0.0;                 // NaN or negative: do not wait
  if (timeout_s > 86400.0) timeout_s = 86400.0;             // keep the chrono conversion finite
  const auto bound = std::chrono::microseconds(static_cast<long long>(timeout_s * 1e6));
  long long at_close = 0;
  long long remaining = 0;
  bool timed_out = false;
  const auto t0 = std::chrono::steady_clock::now();
  {
    py::gil_scoped_release nogil;                           // in-flight releases need the GIL to finish
    std::unique_lock<std::mutex> lk(lc.mu);
    lc.closing = true;
    at_close = lc.in_flight;
    if (at_close > 0) timed_out = !lc.cv.wait_for(lk, bound, [&lc] { return lc.in_flight == 0; });
    remaining = lc.in_flight;
  }  // lk is released before nogil re-takes the GIL
  const long long waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  g_shutdown_calls.fetch_add(1);
  g_shutdown_in_flight_at_close.fetch_add(static_cast<unsigned long long>(at_close));
  if (timed_out) g_shutdown_timeouts.fetch_add(1);
  g_shutdown_still_in_flight.store(static_cast<unsigned long long>(remaining));
  g_shutdown_wait_us.fetch_add(static_cast<unsigned long long>(waited_us));
  py::dict d;
  d["in_flight_at_close"] = at_close;
  d["still_in_flight"] = remaining;
  d["timed_out"] = timed_out;
  d["waited_us"] = waited_us;
  return d;
}

static DLManagedTensor manager{};
static PyObject* callback=nullptr;
static void run_callback(DLManagedTensor*) { PyObject* r=PyObject_CallNoArgs(callback); if(!r) throw py::error_already_set(); Py_DECREF(r); }
PYBIND11_MODULE(_p1t_rev3_lifecycle,m) {
 (void)release_lifecycle();
 m.def("start",[](py::object f){callback=f.ptr();Py_INCREF(callback);manager.deleter=run_callback;std::thread([]{import_release_impl(&manager);}).detach();});
 m.def("close",&dlpack_begin_shutdown);
 m.def("stats",[]{py::dict d;auto& lc=release_lifecycle();std::lock_guard<std::mutex> l(lc.mu);d["in_flight"]=lc.in_flight;d["closing"]=lc.closing;return d;});
}
