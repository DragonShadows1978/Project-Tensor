// PT-RING-1. Prior art: NVIDIA CUDA streams/events/pinned memory (12.6, 2024),
// PyTorch record_stream and pin_memory (2016 onward). Taken: mechanisms and
// event-bound storage lifetime. Ours: legacy-stream integration; no new algorithm.
#pragma once
#include "tc/core.h"
#include <cuda_runtime_api.h>

namespace tc::ring {
struct PinnedBuffer {
  const Shape shape;
  const DType dtype;
  const size_t nbytes;
  void* ptr = nullptr;
  PinnedBuffer(Shape shape, DType dtype);
  ~PinnedBuffer();
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};
struct Event;
struct Stream {
  cudaStream_t handle = nullptr;
  bool non_blocking;
  bool owned;
  explicit Stream(bool non_blocking = true, bool legacy = false);
  ~Stream();
  Stream(const Stream&) = delete;
  void wait(const std::shared_ptr<Event>& event);
  void synchronize();
};
struct Event {
  cudaEvent_t handle = nullptr;
  bool timing;
  bool recorded = false;
  explicit Event(bool enable_timing = false);
  ~Event();
  Event(const Event&) = delete;
  void record(const std::shared_ptr<Stream>& stream);
  bool query();
  void synchronize();
  float elapsed_time(const std::shared_ptr<Event>& end);
};
std::shared_ptr<Stream> legacy_stream();
size_t pinned_bytes();
size_t pinned_memory_limit();
void set_pinned_memory_limit(size_t limit);
size_t collect_async_copies(bool wait = false);
std::pair<size_t, size_t> mem_get_info();
NDArray empty_like(const NDArray& source);
std::shared_ptr<Event> copy_to_host_async(std::vector<NDArray> sources,
    std::vector<std::shared_ptr<PinnedBuffer>> buffers,
    std::shared_ptr<Stream> stream, std::shared_ptr<Event> after);
std::shared_ptr<Event> copy_many(std::vector<NDArray> destinations,
    std::vector<NDArray> sources, const std::string& method);
// Private, bounded GPU gate hooks, never called by training operations.
void test_delay(unsigned milliseconds, const std::shared_ptr<Stream>& stream);
void test_compute(const NDArray& out, unsigned iterations);
}  // namespace tc::ring
