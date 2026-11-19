// APA_SPO4_BEGIN head_parallel
// Prior art: Flash-Decoding (Dao et al. 2023), split-K and partial merge;
// https://pytorch.org/blog/flash-decoding/. FlashInfer (Ye et al. 2025),
// https://arxiv.org/abs/2501.01005, GQA scheduling precedent; no MMA here.
// NVIDIA CUDA Best Practices (2026), independent instruction chains (ILP),
// padded shared transpose and parallel reductions, taken standard techniques.
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
// SP1.1/SPO2 (2026) dot order and fixed-partition selection are inherited.
// BLASST (Yuan et al. 2025) running-max prior art; Hillis/Steele (1986) scan;
// Milakov/Gimelshein (2018) softmax; Higham (2002) wider sums, inherited.
// Ours: per-head CTA scheduling and four independent ordered key-dot trees.
// No prior art known to me for this exact APA wiring; no new general method.
namespace spo4 {
constexpr int C=spo2::C, NT=128;
// Every x[i] uses precisely the reference lane FMA chain, then offsets
// 16,8,4,2,1 in that order, then scale. ILP is only BETWEEN keys.
template<class T> __device__ void dot4(const T* q,const T* k,int D,int key,int count,
                                      float scale,float (&x)[4],const uint8_t* use=nullptr) {
  int lane=threadIdx.x%32;
  #pragma unroll
  for(int i=0;i<4;++i)x[i]=0.f;
  for(int d=lane;d<D;d+=32) {
    float a=ld<T>(q,d);
    #pragma unroll
    for(int i=0;i<4;++i)if(key+i<count && (!use || use[key+i]))
      x[i]+=a*ld<T>(k,(int64_t)(key+i)*D+d);
  }
  #pragma unroll
  for(int off=16;off;off>>=1) {
    #pragma unroll
    for(int i=0;i<4;++i)x[i]+=__shfl_down_sync(0xffffffffu,x[i],off);
  }
  #pragma unroll
  for(int i=0;i<4;++i)x[i]*=scale;
}
__device__ float max128(float x,float* sm) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  #pragma unroll
  for(int o=16;o;o>>=1)x=fmaxf(x,__shfl_down_sync(0xffffffffu,x,o));
  if(lane==0)sm[warp]=x;__syncthreads();
  x=fmaxf(fmaxf(sm[0],sm[1]),fmaxf(sm[2],sm[3]));__syncthreads();return x;
}
template<class T,int CAP,bool W> __global__ void bulk_kernel(
    const T* q,const T* kq,float* bulk,float* cm,int H,int KVH,int S,int D,int NC,float scale) {
  int row=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int bh=(row/H)*KVH+(row%H)/(H/KVH),start=c*C,count=min(C,S-start);
  int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const T* Q=q+(int64_t)row*D;const T* K=kq+((int64_t)bh*S+start)*D;
  int64_t base=(int64_t)row*S+start;
  __shared__ float trans[W?1:64*65],red[4];
  float mx=-INFINITY;
  if constexpr(W) {
    for(int j=warp*4;j<count;j+=16) {
      float x[4];dot4(Q,K,D,j,count,scale,x);
      if(lane==0) {
        #pragma unroll
        for(int i=0;i<4;++i)if(j+i<count){bulk[base+j+i]=x[i];mx=fmaxf(mx,x[i]);}
      }
    }
  } else {
    for(int t=0;t<count;t+=64) {
      int n=min(64,count-t);
      // Padding 65 makes both row-major transpose stores and key reads
      // conflict-free for D64. Irregular dimensions remain supported.
      for(int i=tid;i<n*D;i+=NT)trans[(i%D)*65+i/D]=ld<T>(K,(int64_t)t*D+i);
      __syncthreads();
      if(tid<n) {
        float x=0.f;for(int d=0;d<D;++d)x+=ld<T>(Q,d)*trans[d*65+tid];x*=scale;
        bulk[base+t+tid]=x;mx=fmaxf(mx,x);
      }
      __syncthreads();
    }
  }
  mx=max128(mx,red);if(tid==0)cm[(int64_t)row*NC+c]=mx;
}
template<class T,int CAP,bool W,bool DIAG> __global__ void select_kernel(
    const T* q,const T* k,const T* v,const float* bulk,const float* cm,
    uint8_t* mask,float* prefix_dump,float* pm,float* pl,float* pa,
    int H,int KVH,int S,int D,int VD,int NC,int PK,float scale,float delta) {
  int row=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int bh=(row/H)*KVH+(row%H)/(H/KVH),start=c*C,count=min(C,S-start);
  int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  int64_t base=(int64_t)row*S+start,pr=(int64_t)row*NC+c;
  const T* Q=q+(int64_t)row*D;const T* K=k+((int64_t)bh*S+start)*D;
  const T* V=v+((int64_t)bh*S+start)*VD;
  using Acc=typename std::conditional<std::is_same<T,float>::value,double,float>::type;
  __shared__ float scores[C],wm[4],red[4],trans[W?1:64*65];
  __shared__ uint8_t use[C];
  __shared__ Acc sums[NT],pv[4*CAP];
  float carry=-INFINITY;int first=(start/PK)*PK/C;
  /* APA_SPO5_BEGIN carry_mutant */
    // DeMillo et al. 1978: mutation-only omission of preceding chunk.
    for(int p=first;p<c-(APA_SPO5_MUTANT==2?1:0);++p) /* APA_SPO5_END carry_mutant */carry=fmaxf(carry,cm[(int64_t)row*NC+p]);
  for(int t=0;t<C;t+=NT) {
    int j=t+tid;float x=j<count?bulk[base+j]:-INFINITY,p=x;
    #pragma unroll
    for(int o=1;o<32;o<<=1){float y=__shfl_up_sync(0xffffffffu,p,o);if(lane>=o)p=fmaxf(p,y);}
    if(lane==31)wm[warp]=p;__syncthreads();
    p=fmaxf(p,carry);for(int w=0;w<warp;++w)p=fmaxf(p,wm[w]);
    // APA_SPO5_BEGIN compare_mutant
    // DeMillo et al. 1978: equality-boundary mutation, zero in production.
    bool refine=j<count && (APA_SPO5_MUTANT==6?x>p-delta:x>=p-delta);
    // APA_SPO5_END compare_mutant
    use[j]=refine;scores[j]=x;
    if(j<count){mask[base+j]=refine;if constexpr(DIAG)prefix_dump[base+j]=p;}
    for(int w=0;w<4;++w)carry=fmaxf(carry,wm[w]);__syncthreads();
  }
  if constexpr(W) {
    for(int j=warp*4;j<count;j+=16) {
      float x[4];dot4(Q,K,D,j,count,scale,x,use);
      if(lane==0) {
        #pragma unroll
        for(int i=0;i<4;++i)if(j+i<count && use[j+i])scores[j+i]=x[i];
      }
    }
  } else {
    for(int t=0;t<count;t+=64) {
      int n=min(64,count-t);
      for(int i=tid;i<n*D;i+=NT)if(use[t+i/D])trans[(i%D)*65+i/D]=ld<T>(K,(int64_t)t*D+i);
      __syncthreads();
      if(tid<n && use[t+tid]) {
        float x=0.f;for(int d=0;d<D;++d)x+=ld<T>(Q,d)*trans[d*65+tid];scores[t+tid]=x*scale;
      }
      __syncthreads();
    }
  }
  __syncthreads();float mx=-INFINITY;
  for(int j=tid;j<count;j+=NT)mx=fmaxf(mx,scores[j]);
  mx=max128(mx,red);Acc sum=0;
  for(int j=tid;j<count;j+=NT){float w=__expf(scores[j]-mx);scores[j]=w;sum+=(Acc)w;}
  sums[tid]=sum;__syncthreads();
  for(int o=64;o;o>>=1){if(tid<o)sums[tid]+=sums[tid+o];__syncthreads();}
  if(tid==0){pm[pr]=mx;pl[pr]=(float)sums[0];}
  // Parallel PV across four disjoint key subsequences. Dot order does not
  // constrain PV. FP32 products / wider FP32-output sums inherited from SPO2.
  // APA_SPO4_BEGIN pv_ilp
  // NVIDIA CUDA Best Practices ILP (2026): overlap independent dimension
  // accumulators. Same FP32 products and per-dimension sum order as iteration1.
  Acc acc[CAP/32];
  #pragma unroll
  for(int i=0;i<CAP/32;++i)acc[i]=0;
  for(int j=warp;j<count;j+=4) {
    float w=scores[j];
    #pragma unroll
    for(int i=0;i<CAP/32;++i)if(lane+32*i<VD)
      acc[i]+=(Acc)__fmul_rn(w,ld<T>(V,(int64_t)j*VD+lane+32*i));
  }
  #pragma unroll
  for(int i=0;i<CAP/32;++i)if(lane+32*i<VD)pv[warp*CAP+lane+32*i]=acc[i];
  // APA_SPO4_END pv_ilp
  __syncthreads();
  for(int d=tid;d<VD;d+=NT)pa[pr*VD+d]=(float)((pv[d]+pv[CAP+d])+(pv[2*CAP+d]+pv[3*CAP+d]));
}
} // namespace spo4
// APA_SPO4_END head_parallel
