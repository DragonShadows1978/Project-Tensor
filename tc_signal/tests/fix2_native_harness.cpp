// P1T-FIX2 native harness for tc_signal's release paths (N3) and checked shape arithmetic (N2). Compiled and run by
// tests/test_fix2_findings.py (it is not part of the package). It compiles the REAL tc_signal/src/dlpack_operand.cpp
// into this executable, as the round-2 reviewer's probe artifacts/p1t_rev2/probes/cpu_native.cpp does, with UBSan
// (-fsanitize=undefined -fno-sanitize-recover=undefined) and a replaced global operator new.
//
// Modes:
//   oom_sweep   N3, real allocation failure, swept. For each release path (the RetainUntilComplete destructor, i.e.
//               the pending-list push and the bound drain; drain_completed; synchronize_and_release_all) and each
//               pre-filled list length (0..23 entries, which crosses std::deque node boundaries, and 1024, which makes
//               the next push a bound drain), it first COUNTS the global operator-new calls the path makes, then
//               re-runs it once per k = 1..count with the k-th allocation failing (std::bad_alloc), in fresh state.
//               Invariants per run: the process survives (no std::terminate), no deleter is called twice, and after an
//               unarmed cleanup every consumed manager was released exactly once or deliberately leaked (counted).
//               No kernel is launched: the operands are real cudaMalloc memory, the events and syncs are real.
//   ck_census   N2. tc_signal's checked_math.h (ck_prod, ck_shape_numel) and validate_tensor on zero-extent shapes
//               next to 2^62 and 2^63-1 extents (zero first / middle / last): each must be REFUSED with ValueError and
//               raise no UBSan diagnostic; control shapes must be accepted with the right count.
// Output: one JSON object per line; the last line is {"summary": ...}; exit 0 iff every invariant held.
//
// Prior art: SQLite's out-of-memory test harness (its test allocator fails the Nth malloc, swept over N) and the
// Linux kernel's failslab fault injection; the round-2 reviewer's cpu_native.cpp (operator new replacement that fails
// the next allocation). Taken: fail-the-Nth-allocation sweeps through a replaced allocator. Ours: the release-path
// list, the pre-fill lengths and the release-or-leak-exactly-once invariant. UBSan: GCC/Clang (2013-).
#include <pybind11/embed.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {
std::atomic<bool> g_counting{false};
std::atomic<long> g_countdown{0};  // > 0: the countdown-th allocation from arming fails
std::atomic<long> g_allocs{0};     // allocations seen while counting
std::atomic<bool> g_hit{false};    // the armed allocation was reached (and failed)
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (g_countdown.load(std::memory_order_relaxed) > 0 && g_countdown.fetch_sub(1) == 1) {
      g_hit.store(true);
      throw std::bad_alloc();
    }
  }
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// The unchanged implementation under test (resolved through -I<tree>/tc_signal/src).
#include "dlpack_operand.cpp"

namespace {
namespace py = pybind11;

void arm_count() {
  g_allocs = 0;
  g_countdown = 0;
  g_hit = false;
  g_counting = true;
}
void arm_fail(long k) {
  g_allocs = 0;
  g_hit = false;
  g_countdown = k;
  g_counting = true;
}
void disarm() {
  g_counting = false;
  g_countdown = 0;
}

// ------------------------------------------------------------------------------------------------ managers
constexpr int kMaxManagers = 2 * 1100;
DLManagedTensorVersioned g_mgr[kMaxManagers];
int64_t g_shape[kMaxManagers];
int g_calls[kMaxManagers];
bool g_consumed[kMaxManagers];
int g_next = 0;
float* g_dev = nullptr;

extern "C" void count_deleter(DLManagedTensorVersioned* m) {
  g_calls[reinterpret_cast<intptr_t>(m->manager_ctx)] += 1;
}

void reset_managers() {
  for (int i = 0; i < kMaxManagers; ++i) {
    g_calls[i] = 0;
    g_consumed[i] = false;
  }
  g_next = 0;
}

// A legal fp32 CUDA operand of shape [1] over the device buffer; its deleter counts calls.
py::object make_capsule(int& idx) {
  idx = g_next++;
  DLManagedTensorVersioned& m = g_mgr[idx];
  std::memset(&m, 0, sizeof m);
  m.version.major = 1;
  m.version.minor = 3;
  m.manager_ctx = reinterpret_cast<void*>(static_cast<intptr_t>(idx));
  m.deleter = count_deleter;
  g_shape[idx] = 1;
  m.dl_tensor.data = g_dev + idx;
  m.dl_tensor.device.device_type = kDLCUDA;
  m.dl_tensor.device.device_id = 0;
  m.dl_tensor.ndim = 1;
  m.dl_tensor.dtype.code = kDLFloat;
  m.dl_tensor.dtype.bits = 32;
  m.dl_tensor.dtype.lanes = 1;
  m.dl_tensor.shape = &g_shape[idx];
  m.dl_tensor.strides = nullptr;
  PyObject* c = PyCapsule_New(&m, "dltensor_versioned", nullptr);
  if (c == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(c);
}

// One call's worth of consumed operands (x read, out written) handed to the pending list, unarmed.
void push_entry() {
  int ia = 0, ib = 0;
  py::object ca = make_capsule(ia), cb = make_capsule(ib);
  {
    tcs::Operand a(ca, "x", 1, false);
    tcs::Operand b(cb, "out", 1, true);
    g_consumed[ia] = g_consumed[ib] = true;
    tcs::RetainUntilComplete keep(0, {&a, &b});
  }
}

unsigned long long leaked_now() { return tcs::pending_stats(false)["leaked_managers"].cast<unsigned long long>(); }

struct RunResult {
  bool ok = true;
  std::string why;
  long allocs = 0;
  bool hit = false;
  std::string exc;
};

// Runs `path` once with pre-fill `prefill` and fail-at `k` (0 = count only).
RunResult run_once(const std::string& path, int prefill, long k) {
  RunResult r;
  reset_managers();
  try {
    tcs::synchronize_and_release_all();
  } catch (...) {
  }
  reset_managers();
  const unsigned long long leaked0 = leaked_now();
  for (int i = 0; i < prefill; ++i) push_entry();
  cudaStreamSynchronize(cudaStreamLegacy);  // every pre-filled event is complete (no kernel was launched)
  int ia = 0, ib = 0;
  py::object ca, cb;
  if (path == "dtor") {
    ca = make_capsule(ia);
    cb = make_capsule(ib);
  }
  try {
    if (path == "dtor") {
      tcs::Operand a(ca, "x", 1, false);
      tcs::Operand b(cb, "out", 1, true);
      g_consumed[ia] = g_consumed[ib] = true;
      tcs::RetainUntilComplete keep(0, {&a, &b});
      if (k > 0) arm_fail(k); else arm_count();
    } else if (path == "drain") {
      if (k > 0) arm_fail(k); else arm_count();
      tcs::drain_completed();
    } else {
      if (k > 0) arm_fail(k); else arm_count();
      tcs::synchronize_and_release_all();
    }
    disarm();
  } catch (const std::bad_alloc&) {
    disarm();
    r.exc = "bad_alloc";
  } catch (const std::exception& e) {
    disarm();
    r.exc = std::string("exception: ") + e.what();
  }
  r.allocs = g_allocs.load();
  r.hit = g_hit.load();
  for (int i = 0; i < g_next; ++i)
    if (g_calls[i] > 1) {
      r.ok = false;
      r.why = "deleter called twice";
    }
  try {
    tcs::synchronize_and_release_all();  // unarmed cleanup
  } catch (const std::exception& e) {
    r.ok = false;
    r.why = std::string("cleanup synchronize raised: ") + e.what();
  }
  const unsigned long long leaked = leaked_now() - leaked0;
  unsigned long long consumed = 0, called = 0;
  for (int i = 0; i < g_next; ++i) {
    if (g_consumed[i]) ++consumed;
    if (g_calls[i] > 1) {
      r.ok = false;
      r.why = "deleter called twice";
    }
    if (!g_consumed[i] && g_calls[i] != 0) {
      r.ok = false;
      r.why = "deleter of an unconsumed manager called";
    }
    called += static_cast<unsigned long long>(g_calls[i]);
  }
  if (called + leaked != consumed) {
    r.ok = false;
    r.why = "released + leaked (" + std::to_string(called) + " + " + std::to_string(leaked) + ") != consumed (" +
            std::to_string(consumed) + ")";
  }
  return r;
}

int oom_sweep() {
  if (cudaMalloc(&g_dev, sizeof(float) * kMaxManagers) != cudaSuccess) {
    std::printf("{\"error\": \"cudaMalloc failed\"}\n");
    return 2;
  }
  tcs::pending_stats(false);  // construct the pending state before any arming (as the reviewer's probe)
  int runs = 0, failures = 0;
  long hits_dtor = 0, hits_other = 0, allocs_drain_sync = 0;
  std::vector<int> prefills;
  for (int p = 0; p <= 23; ++p) prefills.push_back(p);  // crosses std::deque node boundaries (10 entries per node)
  prefills.push_back(static_cast<int>(tcs::tuning::kPendingMaxEntries));  // the next push is a bound drain
  for (const char* path : {"dtor", "drain", "sync"}) {
    const bool is_dtor = std::strcmp(path, "dtor") == 0;
    for (int p : prefills) {
      const RunResult c = run_once(path, p, 0);
      ++runs;
      if (!c.ok) ++failures;
      if (!is_dtor) allocs_drain_sync += c.allocs;
      std::printf("{\"path\": \"%s\", \"prefill\": %d, \"k\": 0, \"allocs\": %ld, \"ok\": %s, \"why\": \"%s\"}\n", path,
                  p, c.allocs, c.ok ? "true" : "false", c.why.c_str());
      // Where the path allocates depends on the deque's internal node offset, which differs from run to run, so k
      // sweeps at least 1..2 whatever the count run saw; a run whose k-th allocation did not happen is still checked.
      const long kmax = c.allocs > 2 ? c.allocs : 2;
      for (long k = 1; k <= kmax; ++k) {
        const RunResult r = run_once(path, p, k);
        ++runs;
        if (!r.ok) ++failures;
        if (r.hit) (is_dtor ? hits_dtor : hits_other) += 1;
        std::printf(
            "{\"path\": \"%s\", \"prefill\": %d, \"k\": %ld, \"hit\": %s, \"exception\": \"%s\", \"ok\": %s, "
            "\"why\": \"%s\"}\n",
            path, p, k, r.hit ? "true" : "false", r.exc.c_str(), r.ok ? "true" : "false", r.why.c_str());
      }
      std::fflush(stdout);
    }
  }
  py::dict st = tcs::pending_stats(false);
  // Pass: no invariant failed, the destructor's allocation (the pending-list push) was actually failed at least once,
  // and drain_completed / synchronize_and_release_all made no allocation at all on the success path.
  const bool pass = failures == 0 && hits_dtor > 0 && allocs_drain_sync == 0;
  std::printf("{\"summary\": {\"mode\": \"oom_sweep\", \"runs\": %d, \"failures\": %d, \"injected_in_dtor\": %ld, "
              "\"injected_elsewhere\": %ld, \"allocs_in_drain_and_sync\": %ld, \"push_alloc_failures\": %llu, "
              "\"fallback_syncs\": %llu, \"leaked_managers\": %llu, \"pass\": %s}}\n",
              runs, failures, hits_dtor, hits_other, allocs_drain_sync,
              st["push_alloc_failures"].cast<unsigned long long>(), st["fallback_syncs"].cast<unsigned long long>(),
              st["leaked_managers"].cast<unsigned long long>(), pass ? "true" : "false");
  return pass ? 0 : 1;
}

// ------------------------------------------------------------------------------------------------ ck_census (N2)
int ck_census() {
  const int64_t P62 = int64_t{1} << 62, P61 = int64_t{1} << 61, MAX = INT64_MAX;
  const std::vector<std::vector<int64_t>> refuse = {
      {0, P62, 4}, {P62, 0, 4}, {P62, 4, 0},          // element bound 2^64
      {0, MAX, 2}, {MAX, 2, 0}, {2, 0, MAX},          // element bound 2^64 - 2
      {0, MAX}, {MAX, 0}, {0, MAX, 1}, {1, 0, MAX},   // element bound fits, fp32 bytes overflow
      {0, P62}, {P62, 0}, {0, P61}, {P61, 0, 1},      // bytes 2^64 / 2^63
  };
  const std::vector<std::pair<std::vector<int64_t>, int64_t>> accept = {
      {{0, int64_t{1} << 20, 4}, 0}, {{2, 3, 4}, 24}, {{0, int64_t{1} << 60}, 0}, {{}, 1}};
  int failures = 0, cases = 0;
  auto refused = [](auto&& fn, std::string& msg) {
    try {
      fn();
    } catch (const py::value_error& e) {
      msg = e.what();
      return true;
    } catch (const py::error_already_set& e) {
      msg = e.what();
      return true;
    }
    return false;
  };
  for (const auto& s : refuse) {
    std::string m1, m2, m3;
    const bool r1 = refused([&] { tcs::ck_shape_numel(s, "census ck_shape_numel"); }, m1);
    const bool r2 = refused([&] { tcs::ck_prod(s, 0, s.size(), "census ck_prod"); }, m2);
    // validate_tensor on a descriptor (never dereferenced beyond its shape): must refuse, as an OVERFLOW.
    DLTensor t{};
    std::vector<int64_t> shape = s;
    t.data = reinterpret_cast<void*>(0x1000);
    t.device.device_type = kDLCUDA;
    t.device.device_id = 0;
    t.ndim = static_cast<int>(shape.size());
    t.dtype.code = kDLFloat;
    t.dtype.bits = 32;
    t.dtype.lanes = 1;
    t.shape = shape.data();
    const bool r3 = refused([&] { tcs::validate_tensor(&t, "census", t.ndim); }, m3);
    const bool r3_overflow = r3 && m3.find("overflows") != std::string::npos;
    // ck_prod checks the element bound only (byte-only cases are refused by ck_shape_numel / validate_tensor).
    bool elem_overflow = false;
    {
      int64_t p = 1;
      for (int64_t e : s) elem_overflow |= __builtin_mul_overflow(p, e > 1 ? e : 1, &p);
    }
    const bool ok = r1 && (r2 == elem_overflow) && r3_overflow;
    ++cases;
    if (!ok) ++failures;
    std::printf("{\"shape\": \"%s\", \"ck_shape_numel_refused\": %s, \"ck_prod_refused\": %s, "
                "\"ck_prod_expected_refused\": %s, \"validate_refused_as_overflow\": %s, \"validate_msg\": \"%s\", "
                "\"ok\": %s}\n",
                tcs::shape_str(s).c_str(), r1 ? "true" : "false", r2 ? "true" : "false",
                elem_overflow ? "true" : "false", r3_overflow ? "true" : "false", m3.c_str(), ok ? "true" : "false");
  }
  for (const auto& a : accept) {
    std::string m;
    int64_t n = -1;
    const bool r = refused([&] { n = tcs::ck_shape_numel(a.first, "census control"); }, m);
    const bool ok = !r && n == a.second;
    ++cases;
    if (!ok) ++failures;
    std::printf("{\"control\": \"%s\", \"count\": %lld, \"expected\": %lld, \"ok\": %s}\n",
                tcs::shape_str(a.first).c_str(), static_cast<long long>(n), static_cast<long long>(a.second),
                ok ? "true" : "false");
  }
  std::printf("{\"summary\": {\"mode\": \"ck_census\", \"cases\": %d, \"failures\": %d}}\n", cases, failures);
  return failures == 0 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  py::scoped_interpreter python{};
  if (argc != 2) return 2;
  int rc = 2;
  if (std::strcmp(argv[1], "oom_sweep") == 0) rc = oom_sweep();
  if (std::strcmp(argv[1], "ck_census") == 0) rc = ck_census();
  std::fflush(stdout);
  return rc;
}
