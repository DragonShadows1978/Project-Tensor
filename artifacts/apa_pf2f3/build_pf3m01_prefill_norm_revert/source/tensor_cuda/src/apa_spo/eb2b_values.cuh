// APA-EB2b: opt-in value fetch, included before spq4's templates.
// Prior art: KIVI (Liu et al. 2024, arXiv:2402.02750), per-token V;
// ThriftAttention (Sharratt 2026, arXiv:2605.23081), block mixed precision.
// Taken: those ideas. Ours: int8 -> bf16 register tile for the unchanged
// SPQ4 PV MMA. Symmetric absmax/127 and fp32 scales are the E2a contract,
// not a reproduction of KIVI's asymmetric two-bit format.
#pragma once
#ifndef APA_EB2B_MUTANT
#define APA_EB2B_MUTANT 0
#endif
namespace eb2b {
__device__ __forceinline__ unsigned value_pair(unsigned w, int shift, float scale) {
  float a=float(int(int8_t((w>>shift)&255u)));
  float b=float(int(int8_t((w>>(shift+8))&255u)));
#if APA_EB2B_MUTANT != 6
  a=__fmul_rn(a,scale); b=__fmul_rn(b,scale);
#endif
  __nv_bfloat162 x=__floats2bfloat162_rn(a,b);
  return *reinterpret_cast<unsigned*>(&x);
}
__device__ __forceinline__ uint4 value8(const int8_t* p, float scale) {
  uint2 x=*reinterpret_cast<const uint2*>(p);
  return make_uint4(value_pair(x.x,0,scale),value_pair(x.x,16,scale),
                    value_pair(x.y,0,scale),value_pair(x.y,16,scale));
}
// Same value selection in the PV loader and operand-parity diagnostic.
// ThriftAttention 2026 block precision, KIVI 2024 per-token V; taken.
__device__ __forceinline__ bool high_flag(const uint8_t* flags,int bh,int token,int S) {
  int nb=(S+63)/64,block=token/64;
#if APA_EB2B_MUTANT == 3
  block=min((token+1)/64,nb-1);
#endif
  bool high=flags[(int64_t)bh*nb+block]!=0;
#if APA_EB2B_MUTANT == 1
  high=!high;
#endif
  return high;
}
__device__ __forceinline__ uint4 fetch_value8(const __nv_bfloat16* v,
  const int8_t* vi,const float* vs,const uint8_t* flags,int bh,int token,
  int dim,int S,int VD) {
  int64_t offset=((int64_t)bh*S+token)*VD+dim;
  if(high_flag(flags,bh,token,S))return *reinterpret_cast<const uint4*>(v+offset);
  int scale_token=token;
#if APA_EB2B_MUTANT == 2
  scale_token=(token+1)%S;
#endif
  return value8(vi+offset,vs[(int64_t)bh*S+scale_token]);
}
} // namespace eb2b
