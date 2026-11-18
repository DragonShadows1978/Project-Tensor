// APA_SPO2_D1_BEGIN merge
// Prior art: Flash-Decoding (Dao et al. 2023), stable split-K partial merge;
// Milakov/Gimelshein (2018), normalizer; Higham (2002), wider accumulation.
// Taken: rescale to common max and combine; ours: D1 chunk workspace wiring.
// FP64 merge avoids distinct FP32 numerator/denominator reduction roundoff.
namespace spo2 {
template<class T> __global__ void merge_kernel(const float* pm,const float* pl,const float* pa,
    const T* sinks,T* out,int H,int VD,int NC) {
  int row=blockIdx.x,tid=threadIdx.x;
  using Acc=typename std::conditional<std::is_same<T,float>::value,double,float>::type;
  __shared__ float mx;__shared__ Acc denom;
  const float* m=pm+(int64_t)row*NC;const float* l=pl+(int64_t)row*NC;
  if(tid==0) {
    float x=-INFINITY;for(int c=0;c<NC;++c)x=fmaxf(x,m[c]);
    if(sinks)x=fmaxf(x,ld<T>(sinks,row%H));
    mx=x;Acc sum=0;
    for(int c=0;c<NC;++c)sum+=(Acc)l[c]*(Acc)__expf(m[c]-x);
    if(sinks)sum+=(Acc)(APA_SPO2_MUTANT==5?2.:1.)*(Acc)__expf(ld<T>(sinks,row%H)-x);
    denom=sum;
  }
  __syncthreads();
  for(int d=tid;d<VD;d+=blockDim.x) {
    Acc acc=0;
    for(int c=0;c<NC;++c)acc+=(Acc)pa[((int64_t)row*NC+c)*VD+d]*(Acc)__expf(m[c]-mx);
    // APA-P1-F1 R1 (P1R-F01): only an exactly-zero denominator writes 0. denom
    // is a sum of l_c*exp(m_c-mx) terms (l_c >= 0 is itself a sum of exps).
    // APA-P1-F2 R5 (P1R2-F06), the claim stated narrowly: when every
    // intermediate score (m_c, the sink) and denom are finite, denom >= 0 and
    // the old guard denom>0 selects the same branch as denom!=0, so every such
    // result is unchanged. Finite inputs alone do not guarantee that: finite
    // operands can overflow (a FLT_MAX scale makes the scores +Inf and
    // exp(Inf-Inf) NaN), a NaN or +Inf sink makes denom NaN, and a +-Inf query
    // over all-negative logits makes every m_c -Inf. Those cases now follow
    // IEEE 754 propagation to NaN where the old guard wrote 0 (the loud
    // direction, pinned by APA-P1-F2 LF2). NaN masking is not restored.
    // Prior art: IEEE 754-2008; Milakov and Gimelshein (2018) / FlashAttention
    // (Dao et al., 2022) online-softmax merge, unchanged.
    st<T>(out,(int64_t)row*VD+d,denom!=0?(float)(acc/denom):0.f);
  }
}
}
// APA_SPO2_D1_END merge
