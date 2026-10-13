// CPU call tracing only, not a CUDA runtime reproduction.
// Prior art: NVIDIA CUDA 12.6 synchronous-launch last-error slot vs stream completion.
// Taken: that API distinction; ours: execute unchanged consumer code with explicit successful sync returns.
#include <pybind11/embed.h>
#include <cuda_runtime.h>
#include <iostream>
static cudaError_t slot = cudaErrorInvalidConfiguration;
extern "C" cudaError_t cudaGetDevice(int* d) { *d=0; return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t cudaPeekAtLastError() { return slot; }
extern "C" cudaError_t cudaGetLastError() { auto e=slot;slot=cudaSuccess;return e; }
#include "dlpack_operand.cpp"
int main() {
 pybind11::scoped_interpreter py{};
 tcs::init_pending_state();
 try { tcs::synchronize_and_release_all(); }
 catch(const std::exception& e) { std::cout<<"ERROR "<<e.what()<<std::endl;return 1; }
 std::cout<<"synchronize returned success; previous_error_slot="<<int(cudaPeekAtLastError())<<std::endl;
 return slot==cudaErrorInvalidConfiguration?0:1;
}
