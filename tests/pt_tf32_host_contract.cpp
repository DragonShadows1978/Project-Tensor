// PT-TF32-1 host-only guard test. Prior art: BP-KERNEL-4 contract checks
// (Project-Tensor 2026), taken; ours: CPU NDArrays without host-transfer APIs.
// NDArray::from_host calls cuda_check_last even for CPU arrays in this engine,
// so construct raw CPU buffers to exercise rejection without CUDA discovery.
#include "tc/tf32.h"
#include <iostream>
#include <limits>
#include <stdexcept>

template<class F> void rejects(F fn, const std::string& expected) {
  try { fn(); }
  catch(const std::runtime_error& e) {
    if(std::string(e.what()).find(expected)!=std::string::npos) return;
    throw;
  }
  throw std::runtime_error("expected rejection: " + expected);
}
int main() {
  using namespace tc;
  Device cpu{DeviceType::CPU,0};
  NDArray q({1,1,2,3},DType::Float32,cpu);
  NDArray wide({1,1,2,129},DType::Float32,cpu);
  NDArray flat({2,3},DType::Float32,cpu);
  rejects([&]{apa_selective_fwd_tf32(q,q,q,q,1,0,true,"h_tf32");},"same CUDA FP32");
  rejects([&]{apa_selective_fwd_tf32(q,q,q,q,1,0,true,"a",true);},"same CUDA FP32");
  rejects([&]{apa_selective_fwd_tf32(wide,wide,wide,wide,1,0,true,"h_tf32");},"geometry");
  rejects([&]{apa_selective_fwd_tf32(flat,q,q,q,1,0,true,"h_tf32");},"rank");
  rejects([&]{apa_selective_fwd_tf32(q,q,q,q,std::numeric_limits<float>::quiet_NaN(),0,true,"h_tf32");},"geometry");
  rejects([&]{apa_selective_bwd_tf32(q,q,q,q,q,q,q,q,1,true);},"same CUDA FP32");
  std::cout << "HOST_CONTRACT: 6 expected rejections; no CUDA device operations\n";
}
