// P1T-FIX3 native harness for round-3 review finding V3-02 (aggregate deferred-release byte accounting). Compiled and
// run by tests/test_fix3_findings.py (it is not part of the package). It compiles the REAL tc_signal/src/dlpack_operand.cpp
// into this executable, as the round-3 reviewer's probe artifacts/p1t_rev3/probes/pending_bytes.cpp does, with UBSan
// (-fsanitize=undefined -fno-sanitize-recover=undefined): any signed overflow aborts the run with a diagnostic.
//
// Modes:
//   accounting  No CUDA call. The saturating pending-list total (note_push / note_pop) and the bound comparison
//               (pending_over_bound) on a local PendingState whose queue holds real PendingEntry objects:
//                 * the reviewer's queue case (4 + 9223372036854775804) saturates at INT64_MAX, is over the bound,
//                   stays pinned while entries remain, and returns to an exact 0 when the list empties;
//                 * an exact sum of exactly INT64_MAX is not saturated; one more byte saturates;
//                 * the bound at the boundary: a total of exactly kPendingMaxBytes is within it, 4 more bytes are
//                   over it (fp32 granularity), and popping back returns within it; kPendingMaxEntries entries are
//                   within the entry bound, one more is over it;
//                 * aggregate_operand_bytes on real Operands (synthetic descriptors, never dereferenced): the reviewer's
//                   two disjoint (2^61-1,) fp32 operands are REFUSED (ValueError, "overflows"), as are three (2^60,)
//                   ones; two (2^60-1,) ones sum exactly to 2^63-8.
//   dtor        Real CUDA events and syncs, NO kernel launch (the operands' fake addresses are never dereferenced):
//               the real RetainUntilComplete constructor and destructor near INT64_MAX:
//                 * D1: a queued 8-byte entry plus a (2^60-1, 2^60-1) call (2^63-8 bytes, accepted): the push would
//                   overflow -> saturated -> bound drain -> every manager released exactly once, the total back to an
//                   exact 0 and unsaturated;
//                 * D2: the reviewer's (2^61-1, 2^61-1) pair is refused by the CONSTRUCTOR (ValueError) before anything
//                   is queued; both managers are released at once by the operands' own destructors;
//                 * D3: entries totalling exactly kPendingMaxBytes stay queued (no drain); the next 4-byte call drains.
// Output: one JSON object per line; the last line is {"summary": ...}; exit 0 iff every check held.
//
// Prior art: the round-3 reviewer's pending_bytes.cpp (actual Operand validation and note_push under UBSan; its two
// boundary cases are reused here); saturating arithmetic (DSP/SIMD saturated add) and the checked builtins (GCC 5, 2015;
// C23 ckd_add); UBSan (GCC/Clang, 2013-). Ours: the pop/empty sequence, the boundary cases and the real-destructor runs.
#include <pybind11/embed.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// The implementation under test (resolved through -I<tree>/tc_signal/src).
#include "dlpack_operand.cpp"

namespace {
namespace py = pybind11;

int g_cases = 0;
int g_failures = 0;

void check(const char* name, bool ok, const std::string& detail = "") {
  ++g_cases;
  if (!ok) ++g_failures;
  std::printf("{\"case\": \"%s\", \"ok\": %s, \"detail\": \"%s\"}\n", name, ok ? "true" : "false", detail.c_str());
  std::fflush(stdout);
}

constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

// ------------------------------------------------------------------------------------------------ accounting
void push(tcs::PendingState& st, int64_t bytes) {
  tcs::PendingEntry e;
  e.bytes = bytes;
  st.q.push_back(std::move(e));
  tcs::note_push(st, st.q.back());  // as ~RetainUntilComplete does
}

void pop(tcs::PendingState& st) {
  tcs::PendingEntry e = std::move(st.q.front());
  st.q.pop_front();
  tcs::note_pop(st, e);  // as drain_completed / sync_and_release_all_impl do
}

std::string state(const tcs::PendingState& st) {
  return "bytes=" + std::to_string(st.bytes) + " saturated=" + (st.bytes_saturated ? "1" : "0") +
         " entries=" + std::to_string(st.q.size()) + " over=" + (tcs::pending_over_bound(st) ? "1" : "0");
}

// Synthetic legacy managers (no deleter: nothing to release) over fake, never-dereferenced addresses.
struct Desc {
  DLManagedTensor m{};
  int64_t shape = 0;
};

py::object capsule(Desc& d, int64_t n, uintptr_t addr) {
  d.shape = n;
  d.m.dl_tensor.data = reinterpret_cast<void*>(addr);
  d.m.dl_tensor.device = {kDLCUDA, 0};
  d.m.dl_tensor.ndim = 1;
  d.m.dl_tensor.dtype = {kDLFloat, 32, 1};
  d.m.dl_tensor.shape = &d.shape;
  d.m.dl_tensor.strides = nullptr;
  d.m.dl_tensor.byte_offset = 0;
  PyObject* c = PyCapsule_New(&d.m, "dltensor", nullptr);
  if (c == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(c);
}

int accounting() {
  {  // the reviewer's queue case, inside a real queue
    tcs::PendingState st;
    push(st, 4);
    push(st, kMax - 3);  // 4 + 9223372036854775804 overflows int64
    check("saturates_on_overflowing_push", st.bytes == kMax && st.bytes_saturated && st.bytes_saturations == 1,
          state(st));
    check("saturated_total_is_over_the_bound", tcs::pending_over_bound(st), state(st));
    push(st, kMax);  // further pushes while saturated stay pinned, never overflow
    check("pushes_while_saturated_stay_pinned", st.bytes == kMax && st.bytes_saturations == 1, state(st));
    pop(st);
    check("pop_keeps_saturated_total_pinned_while_entries_remain", st.bytes == kMax && st.bytes_saturated,
          state(st));
    pop(st);
    pop(st);
    check("empty_list_resets_to_exact_zero", st.bytes == 0 && !st.bytes_saturated && !tcs::pending_over_bound(st),
          state(st));
    push(st, 12);
    check("exact_again_after_reset", st.bytes == 12 && !st.bytes_saturated, state(st));
  }
  {  // exact up to INT64_MAX itself; one more byte saturates
    tcs::PendingState st;
    push(st, kMax - 10);
    push(st, 10);
    check("exact_sum_of_int64_max_is_not_saturated", st.bytes == kMax && !st.bytes_saturated, state(st));
    pop(st);
    check("exact_pop_from_int64_max", st.bytes == 10 && !st.bytes_saturated, state(st));
    push(st, kMax - 10);
    push(st, 1);
    check("one_byte_past_int64_max_saturates", st.bytes == kMax && st.bytes_saturated, state(st));
  }
  {  // the byte bound at the boundary (fp32 granularity: 4 bytes)
    tcs::PendingState st;
    push(st, tcs::tuning::kPendingMaxBytes - 4);
    push(st, 4);
    check("total_equal_to_the_byte_bound_is_within_it", st.bytes == tcs::tuning::kPendingMaxBytes &&
                                                            !tcs::pending_over_bound(st), state(st));
    push(st, 4);
    check("four_bytes_over_the_byte_bound_is_over_it", tcs::pending_over_bound(st), state(st));
    tcs::PendingEntry e = std::move(st.q.back());
    st.q.pop_back();
    tcs::note_pop(st, e);
    check("back_within_the_byte_bound_after_pop", !tcs::pending_over_bound(st), state(st));
  }
  {  // the entry bound at the boundary
    tcs::PendingState st;
    for (long long i = 0; i < tcs::tuning::kPendingMaxEntries; ++i) push(st, 0);
    check("entries_equal_to_the_entry_bound_are_within_it", !tcs::pending_over_bound(st), state(st));
    push(st, 0);
    check("one_entry_over_the_entry_bound_is_over_it", tcs::pending_over_bound(st), state(st));
  }
  {  // the per-call aggregate (pre-launch check) on real Operands
    const int64_t n61 = (int64_t{1} << 61) - 1, n60 = int64_t{1} << 60;
    Desc dx, dy, dz;
    tcs::Operand X(capsule(dx, n61, 4), "x", 1, false), Y(capsule(dy, n61, uintptr_t{1} << 63), "out", 1, true);
    tcs::expect_disjoint(Y, X);  // individually valid AND disjoint, as the reviewer staged them
    std::string msg;
    bool refused = false;
    try {
      tcs::aggregate_operand_bytes({&X, &Y});
    } catch (const py::value_error& e) {
      refused = true;
      msg = e.what();
    }
    check("aggregate_of_two_2p61m1_operands_refused_as_overflow",
          refused && msg.find("aggregate") != std::string::npos && msg.find("overflows") != std::string::npos, msg);
    Desc a, b, c;
    tcs::Operand A(capsule(a, n60, 0x1000), "x", 1, false), B(capsule(b, n60, 0x1000 + (uintptr_t{1} << 62)), "y", 1,
                                                                false),
        C(capsule(c, n60, 0x1000 + (uintptr_t{1} << 63)), "out", 1, true);
    refused = false;
    try {
      tcs::aggregate_operand_bytes({&A, &B, &C});
    } catch (const py::value_error&) {
      refused = true;
    }
    check("aggregate_of_three_2p60_operands_refused", refused);
    Desc p, q;
    tcs::Operand P(capsule(p, n60 - 1, 0x1000), "x", 1, false), Q(capsule(q, n60 - 1, 0x1000 + (uintptr_t{1} << 62)),
                                                                  "out", 1, true);
    int64_t sum = -1;
    try {
      sum = tcs::aggregate_operand_bytes({&P, &Q, nullptr});
    } catch (...) {
    }
    check("aggregate_that_fits_is_exact", sum == kMax - 7, std::to_string(sum));  // 2^63 - 8
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------ dtor (real CUDA)
int g_calls[64];
extern "C" void count_deleter(DLManagedTensor* m) { g_calls[reinterpret_cast<intptr_t>(m->manager_ctx)] += 1; }

struct Managed {
  DLManagedTensor m{};
  int64_t shape = 0;
};
Managed g_mgr[64];
int g_next = 0;

py::object counted_capsule(int64_t n, uintptr_t addr, int& idx) {
  idx = g_next++;
  Managed& d = g_mgr[idx];
  d = Managed{};
  d.shape = n;
  d.m.manager_ctx = reinterpret_cast<void*>(static_cast<intptr_t>(idx));
  d.m.deleter = count_deleter;
  d.m.dl_tensor.data = reinterpret_cast<void*>(addr);
  d.m.dl_tensor.device = {kDLCUDA, 0};
  d.m.dl_tensor.ndim = 1;
  d.m.dl_tensor.dtype = {kDLFloat, 32, 1};
  d.m.dl_tensor.shape = &d.shape;
  PyObject* c = PyCapsule_New(&d.m, "dltensor", nullptr);
  if (c == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(c);
}

long long stat(const char* k) { return tcs::pending_stats(false)[k].cast<long long>(); }
bool flag(const char* k) { return tcs::pending_stats(false)[k].cast<bool>(); }

// One "call": consume operands of the given element counts at adjacent, disjoint fake addresses (from 4 up, as the
// reviewer staged them, so two (2^61-1,) operands end at 2^64-4 without wrapping) and run the real retainer
// (constructor + destructor), launching nothing. Returns "" or the constructor's ValueError text.
std::string call(const std::vector<int64_t>& numels, std::vector<int>& idx) {
  std::vector<py::object> caps;
  uintptr_t addr = 4;
  for (int64_t n : numels) {
    int i = 0;
    caps.push_back(counted_capsule(n, addr, i));
    idx.push_back(i);
    addr += static_cast<uintptr_t>(n) * sizeof(float);
  }
  std::vector<tcs::Operand> ops;
  ops.reserve(caps.size());
  for (size_t k = 0; k < caps.size(); ++k) ops.emplace_back(caps[k], k + 1 == caps.size() ? "out" : "x", 1, k + 1 == caps.size());
  try {
    if (ops.size() == 1) {
      tcs::RetainUntilComplete keep(0, {&ops[0]});
    } else {
      tcs::RetainUntilComplete keep(0, {&ops[0], &ops[1]});
    }
  } catch (const py::value_error& e) {
    return e.what();
  }
  return "";
}

bool all_once(const std::vector<int>& idx) {
  for (int i : idx)
    if (g_calls[i] != 1) return false;
  return true;
}
bool all_zero(const std::vector<int>& idx) {
  for (int i : idx)
    if (g_calls[i] != 0) return false;
  return true;
}

int dtor() {
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    std::printf("{\"error\": \"no CUDA device\"}\n");
    return 2;
  }
  tcs::init_pending_state();
  tcs::synchronize_and_release_all();
  const int64_t n60m1 = (int64_t{1} << 60) - 1, n61m1 = (int64_t{1} << 61) - 1;
  {  // D1
    std::vector<int> small, huge;
    const long long drains0 = stat("bound_drains"), sat0 = stat("bytes_saturations");
    const std::string e1 = call({1, 1}, small);
    const bool queued = stat("entries") == 1 && stat("bytes") == 8 && all_zero(small);
    const std::string e2 = call({n60m1, n60m1}, huge);  // 8 + (2^63 - 8) would overflow the total
    check("D1_small_entry_queued", e1.empty() && queued, e1);
    check("D1_huge_call_accepted_by_the_constructor", e2.empty(), e2);
    check("D1_saturated_push_drained_the_list", stat("bound_drains") == drains0 + 1 &&
                                                    stat("bytes_saturations") == sat0 + 1 && stat("entries") == 0,
          "drains=" + std::to_string(stat("bound_drains") - drains0));
    check("D1_total_back_to_exact_zero_unsaturated", stat("bytes") == 0 && !flag("bytes_saturated"));
    check("D1_every_manager_released_exactly_once", all_once(small) && all_once(huge));
  }
  {  // D2
    std::vector<int> pair;
    const long long retained0 = stat("retained_entries");
    const std::string e = call({n61m1, n61m1}, pair);
    check("D2_constructor_refuses_the_reviewer_pair",
          e.find("aggregate") != std::string::npos && e.find("overflows") != std::string::npos, e);
    check("D2_nothing_queued", stat("retained_entries") == retained0 && stat("entries") == 0);
    check("D2_both_managers_released_at_once", all_once(pair));
  }
  {  // D3
    std::vector<int> a, b, c;
    tcs::synchronize_and_release_all();
    const long long drains0 = stat("bound_drains");
    const int64_t first = tcs::tuning::kPendingMaxBytes / 4 - 2;  // + 1 element = kPendingMaxBytes - 4 bytes
    call({first, 1}, a);
    call({1}, b);  // total == kPendingMaxBytes exactly
    check("D3_total_equal_to_the_bound_stays_queued", stat("bytes") == tcs::tuning::kPendingMaxBytes &&
                                                          stat("bound_drains") == drains0 && stat("entries") == 2,
          "bytes=" + std::to_string(stat("bytes")));
    call({1}, c);  // 4 bytes over
    check("D3_four_bytes_over_the_bound_drains", stat("bound_drains") == drains0 + 1 && stat("entries") == 0 &&
                                                     stat("bytes") == 0);
    check("D3_every_manager_released_exactly_once", all_once(a) && all_once(b) && all_once(c));
  }
  tcs::synchronize_and_release_all();
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  py::scoped_interpreter python{};
  if (argc != 2) return 2;
  int rc = 2;
  if (std::strcmp(argv[1], "accounting") == 0) rc = accounting();
  if (std::strcmp(argv[1], "dtor") == 0) rc = dtor();
  std::printf("{\"summary\": {\"mode\": \"%s\", \"cases\": %d, \"failures\": %d}}\n", argv[1], g_cases, g_failures);
  std::fflush(stdout);
  return rc != 0 ? rc : (g_failures == 0 && g_cases > 0 ? 0 : 1);
}
