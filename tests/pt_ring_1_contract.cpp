// PT-RING-1 author CPU bridge: actual shared copy_contract.h, no CUDA calls.
// Prior art: C ABI/ctypes (Python, 2006) and exhaustive interval oracle testing;
// ours: adversarial shape/range contract cases for these copy entry points.
#include "tc/copy_contract.h"
#include <string>
static thread_local std::string last_error;
extern "C" const char* ring_error() { return last_error.c_str(); }
extern "C" int ring_bytes(const int64_t* dims, size_t rank, int dtype, size_t* result) {
  try { *result = tc::ring::checked_bytes(tc::Shape(dims, dims+rank), static_cast<tc::DType>(dtype)); return 0; }
  catch (const std::exception& e) { last_error = e.what(); return 1; }
}
extern "C" int ring_alias(const uintptr_t* dst, const uintptr_t* src, const size_t* bytes, size_t n) {
  try {
    std::vector<tc::ring::Range> writes, reads;
    for (size_t i=0; i<n; ++i) {
      writes.push_back(tc::ring::range(reinterpret_cast<void*>(dst[i]), bytes[i], i));
      reads.push_back(tc::ring::range(reinterpret_cast<void*>(src[i]), bytes[i], i));
    }
    tc::ring::validate_aliases(writes, reads); return 0;
  } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
