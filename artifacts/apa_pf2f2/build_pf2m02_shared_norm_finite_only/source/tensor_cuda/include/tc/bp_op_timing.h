#pragma once
// BP-CENSUS-1, opt-in/default-OFF observation only.
// Prior art: NVIDIA CUDA events / controlled operator profiling (CUDA system,
// 2007 onward; exact documentation year unverified — lead to check
// "CUDA events elapsed time profiler overhead"). Nested exclusive accounting
// follows call-tree profilers (gprof, Graham/Kessler/McKusick, 1982;
// unverified — lead to check). Ours: GRAPA boundaries and boundary memory tags.
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace tc { namespace bp {
struct Sample {
  size_t used = 0, total = 0, pool_used = 0, pool_reserved = 0;
};
struct Row {
  std::string name;
  float inclusive_ms = 0, exclusive_ms = 0, child_ms = 0;
  Sample before, after;
  cudaEvent_t start = nullptr, stop = nullptr;
};
inline constexpr const char* source_pin = "a9effc827e8fe76bea1af05c636546517ee63cf7d1fa0bb2739a91a6acd5ed38";
inline thread_local bool enabled = false;
inline thread_local std::vector<Row> rows;
inline thread_local std::vector<size_t> stack;
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
inline Sample sample() {
  Sample s; size_t free = 0;
  check(cudaMemGetInfo(&free, &s.total)); s.used = s.total - free;
  int dev = 0; cudaMemPool_t pool;
  check(cudaGetDevice(&dev)); check(cudaDeviceGetDefaultMemPool(&pool, dev));
  check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &s.pool_used));
  check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &s.pool_reserved));
  return s;
}
inline cudaEvent_t wall_start = nullptr, wall_stop = nullptr;
inline void wall_begin() {
  check(cudaDeviceSynchronize());
  check(cudaEventCreate(&wall_start)); check(cudaEventCreate(&wall_stop));
  check(cudaEventRecord(wall_start, 0));
}
inline float wall_end() {
  check(cudaEventRecord(wall_stop, 0)); check(cudaEventSynchronize(wall_stop));
  float ms = 0; check(cudaEventElapsedTime(&ms, wall_start, wall_stop));
  check(cudaEventDestroy(wall_start)); check(cudaEventDestroy(wall_stop));
  wall_start = wall_stop = nullptr; return ms;
}
inline void configure(bool on) {
  if (!stack.empty()) throw std::runtime_error("BP timer has unclosed scopes");
  const char* value = std::getenv("TC_OP_TIMING");
  if (on && (!value || std::strcmp(value, "1")))
    throw std::runtime_error("BP timing requires TC_OP_TIMING=1");
  enabled = on; rows.clear();
}
inline size_t begin(const std::string& name) {
  if (!enabled) return static_cast<size_t>(-1);
  check(cudaDeviceSynchronize());
  Row row; row.name = name; row.before = sample();
  check(cudaEventCreate(&row.start)); check(cudaEventCreate(&row.stop));
  check(cudaEventRecord(row.start, 0));
  size_t index = rows.size(); rows.push_back(row); stack.push_back(index);
  return index;
}
inline void end(size_t index) {
  if (index == static_cast<size_t>(-1)) return;
  if (stack.empty() || stack.back() != index)
    throw std::runtime_error("BP timer stack mismatch");
  Row& row = rows.at(index);
  check(cudaEventRecord(row.stop, 0)); check(cudaEventSynchronize(row.stop));
  check(cudaEventElapsedTime(&row.inclusive_ms, row.start, row.stop));
  row.after = sample();
  row.exclusive_ms = row.inclusive_ms - row.child_ms;
  check(cudaEventDestroy(row.start)); check(cudaEventDestroy(row.stop));
  row.start = row.stop = nullptr;
  stack.pop_back();
  if (!stack.empty()) rows.at(stack.back()).child_ms += row.inclusive_ms;
}
struct Scope {
  size_t index;
  explicit Scope(const std::string& name): index(begin(name)) {}
  ~Scope() noexcept(false) { end(index); }
};
} }
