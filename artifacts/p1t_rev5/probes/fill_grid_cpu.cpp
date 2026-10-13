// PASS iff every tested fill extent is refused or its computed launch covers all elements.
// Prior art: ISO C++17 integral narrowing; NVIDIA CUDA12.6 dim3/grid indexing.
// Taken: finite-width grid arithmetic; ours: test the new fill_ call at 2^40+1 using the unchanged baseline helper.
// CPU arithmetic only; synthetic metadata, no resident CUDA allocation or GPU execution.
#include <cstdint>
#include <iostream>
constexpr int kT = 256;
inline int nblk(int64_t n) { return static_cast<int>((n + kT - 1) / kT); }
int main(){int failures=0;for(int64_t n:{int64_t{1},(int64_t{1}<<40)+1,(int64_t{1}<<40)+256}){int blocks=nblk(n);int64_t coverage=static_cast<int64_t>(static_cast<unsigned int>(blocks))*kT;std::cout<<"n="<<n<<" grid="<<blocks<<" covered="<<coverage<<"\n";if(coverage<n)++failures;}return failures?1:0;}
