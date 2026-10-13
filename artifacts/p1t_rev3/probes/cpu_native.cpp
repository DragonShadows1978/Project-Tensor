// CPU-only actual-source probe. Prior art: C++17 noexcept exception rules / allocation failure;
// GCC 5 (2015) UBSan signed overflow; CPython 3.12 embedding. Taken: diagnostics, ours: cases.
#include <pybind11/embed.h>
#include <atomic>
#include <cstdlib>
#include <new>
#include <iostream>
#include <cstring>
static std::atomic<bool> fail_next{false};
void* operator new(std::size_t n) {
  if (fail_next.exchange(false)) throw std::bad_alloc();
  if (void* p = std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
// Compile the unchanged reviewed implementation directly; no CUDA API replacements.
#include "dlpack_operand.cpp"
int main(int argc, char** argv) {
  pybind11::scoped_interpreter python{};
  if (argc != 2) return 2;
  if (std::strcmp(argv[1], "oom_sync") == 0) {
    tcs::pending_stats(false); // construct the real pending state before arming allocation failure
    std::cerr << "ARMED: next host allocation fails; no CUDA call reached yet\n";
    fail_next = true;
    try { tcs::synchronize_and_release_all(); }
    catch (const std::bad_alloc&) { std::cout << "PASS: allocation failure propagated safely\n"; return 0; }
    catch (const std::exception& e) { std::cout << "PASS: safe error: " << e.what() << "\n"; return 0; }
    std::cout << "PASS: nonallocating completion path\n";
    return 0;
  }
  if (std::strcmp(argv[1], "throw_deleter") == 0) {
    DLManagedTensor m{};
    m.deleter = [](DLManagedTensor*) { throw std::runtime_error("producer deleter threw"); };
    tcs::ManagedOwner o; o.legacy = &m;
    std::cerr << "ARMED: real ManagedOwner reset calls throwing native producer\n";
    o.reset();
    std::cout << "PASS: producer exception contained\n";
    return 0;
  }
  return 2;
}
