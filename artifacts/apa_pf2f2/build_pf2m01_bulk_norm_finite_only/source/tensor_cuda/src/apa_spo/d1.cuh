// APA_SPO5_BEGIN mutation_hooks
// Prior art: DeMillo et al. (1978), compiled mutation testing; production=0.
#ifndef APA_SPO5_MUTANT
#define APA_SPO5_MUTANT 0
#endif
// APA_SPO5_END mutation_hooks
// APA_SPO2_D1_BEGIN implementation
// Prior art: Flash-Decoding (Dao/Haziza/Massa/Sizov 2023), split-K + merge,
// https://pytorch.org/blog/flash-decoding/; FlashInfer (Ye et al. 2025),
// GQA reuse, https://arxiv.org/abs/2501.01005 (no tensor cores used here).
// BLASST (Yuan et al. 2025), running-max criterion, arXiv:2512.12087;
// SP1.1 (Project-Tensor 2026) supplies exact dot order and partition semantics.
// Hillis/Steele 1986, Blelloch 1990/1993: associative prefix scan, taken.
// Milakov/Gimelshein 2018, arXiv:1805.02867: softmax partial merge, taken.
// Ours: staged-score wiring preserving SP1.1 decisions. No novel algorithm
// claimed. No last-block-done reduction, decoupled look-back, or MMA used.
namespace spo2 {
constexpr int C=256, TILE=16, NT=128;
// Coalesced 16-byte reads, each KV tile read once across all grouped heads.
// CUDA vectorized memory access (NVIDIA CUDA programming guide), taken.
template<class T> __device__ void stage(const T* src,T* dst,int width,int count,const int* use=nullptr) {
  constexpr int V=16/sizeof(T);
  if(width%V==0) {
    for(int i=threadIdx.x;i<count*(width/V);i+=NT) {
      int j=i/(width/V);
      if(!use || use[j]) reinterpret_cast<uint4*>(dst)[i]=reinterpret_cast<const uint4*>(src)[i];
    }
  } else {
    for(int i=threadIdx.x;i<count*width;i+=NT)
      if(!use || use[i/width]) dst[i]=src[i];
  }
}
template<class T,bool W> __device__ float dot(const T* q,const T* kv,int D,float scale) {
  float x=0.f;
  if constexpr(W) {
    for(int d=threadIdx.x%32;d<D;d+=32) x+=ld<T>(q,d)*ld<T>(kv,d);
    #pragma unroll
    for(int off=16;off>0;off>>=1)x+=__shfl_down_sync(0xffffffffu,x,off);
    x=__shfl_sync(0xffffffffu,x,0);
  } else {
    for(int d=0;d<D;++d)x+=ld<T>(q,d)*ld<T>(kv,d);
  }
  return x*scale;
}
// Prior art: bank-conflict-free shared transposition, NVIDIA CUDA Best
// Practices matrix transpose example; taken layout technique, same FMA order.
template<class T> __device__ float dot_trans(const T* q,const float* kt,int j,int D,int tile_size,float scale) {
  float x=0.f;for(int d=0;d<D;++d)x+=ld<T>(q,d)*kt[d*tile_size+j];return x*scale;
}
template<class T,int CAP,bool W> __global__ void bulk_kernel(
    const T* q,const T* kq,float* bulk,float* cm,int H,int KVH,int S,int D,int NC,float scale) {
  constexpr int TILE=W?16:64;
  __shared__ __align__(16) T tile[TILE*CAP];
  __shared__ float trans[W?1:TILE*CAP];
  __shared__ float red[NT];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z,gsize=H/KVH,b=bh/KVH,h0=(bh%KVH)*gsize;
  if(c>=NC)return;
  int start=c*C,count=min(C,S-start),tid=threadIdx.x,lane=tid%32;
  for(int t=0;t<count;t+=TILE) {
    int n=min(TILE,count-t);
    stage(kq+((int64_t)bh*S+start+t)*D,tile,D,n);__syncthreads();
    if constexpr(!W) {
      for(int i=tid;i<n*D;i+=NT)trans[(i%D)*TILE+i/D]=ld<T>(tile,i);
      __syncthreads();
    }
    for(int g=0;g<gsize;++g) {
      int row=b*H+h0+g;
      if constexpr(W) {
        for(int j=tid/32;j<n;j+=4) {
          float s=dot<T,true>(q+(int64_t)row*D,tile+j*D,D,scale);
          if(lane==0)bulk[(int64_t)row*S+start+t+j]=s;
        }
      } else {
        if(tid<n)bulk[(int64_t)row*S+start+t+tid]=dot_trans<T>(q+(int64_t)row*D,trans,tid,D,TILE,scale);
      }
    }
    __syncthreads();
  }
  // CTA-scoped synchronization makes this CTA's staged writes visible.
  for(int g=0;g<gsize;++g) {
    int row=b*H+h0+g;float mx=-INFINITY;
    for(int j=tid;j<count;j+=NT)mx=fmaxf(mx,bulk[(int64_t)row*S+start+j]);
    red[tid]=mx;__syncthreads();
    for(int off=NT/2;off;off>>=1){if(tid<off)red[tid]=fmaxf(red[tid],red[tid+off]);__syncthreads();}
    if(tid==0)cm[(int64_t)row*NC+c]=red[0];__syncthreads();
  }
}
// Mutations are compile-time, diagnostic test copies only (HOUSE section 8).
#ifndef APA_SPO2_MUTANT
#define APA_SPO2_MUTANT 0
#endif
template<class T,int CAP,bool W,bool DIAG> __global__ void select_kernel(
    const T* q,const T* k,const T* v,const float* bulk,const float* cm,
    float* score,uint8_t* mask,float* prefix_dump,float* pm,float* pl,float* pa,void* accum_workspace,
    int H,int KVH,int S,int D,int VD,int NC,int PK,float scale,float delta) {
  constexpr int TILE=W?16:64;
  __shared__ __align__(16) T tile[TILE*CAP];
  __shared__ float trans[W?1:TILE*CAP];
  __shared__ float tmp[C],red[NT],wm[4];
  __shared__ int any[C];
  // FP32 outputs need wider sums at the registered one-rounding floor.
  // For lower precision output the FP32 sum is gated separately at its floor.
  using Acc=typename std::conditional<std::is_same<T,float>::value,double,float>::type;
  Acc* wide_acc=(Acc*)accum_workspace;
  __shared__ Acc sum_red[NT];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z,G=H/KVH,b=bh/KVH,h0=(bh%KVH)*G;
  if(c>=NC)return;
  int start=c*C,count=min(C,S-start),tid=threadIdx.x,lane=tid%32,warp=tid/32;
  for(int j=tid;j<C;j+=NT)any[j]=0;
  __syncthreads();
  for(int g=0;g<G;++g) {
    int row=b*H+h0+g;int64_t base=(int64_t)row*S+start;
    float carry=-INFINITY;
    int first=(start/PK)*PK/C;
    if constexpr(APA_SPO2_MUTANT==1)first=0;
    /* APA_SPO5_BEGIN carry_mutant */
    // DeMillo et al. 1978: mutation-only omission of preceding chunk.
    for(int p=first;p<c-(APA_SPO5_MUTANT==2?1:0);++p) /* APA_SPO5_END carry_mutant */carry=fmaxf(carry,cm[(int64_t)row*NC+p]);
    for(int t=0;t<C;t+=NT) {
      int j=t+tid;float x=j<count?bulk[base+j]:-INFINITY,p=x;
      #pragma unroll
      for(int off=1;off<32;off<<=1){float y=__shfl_up_sync(0xffffffffu,p,off);if(lane>=off)p=fmaxf(p,y);}
      if(lane==31)wm[warp]=p;__syncthreads();
      p=fmaxf(p,carry);for(int w=0;w<warp;++w)p=fmaxf(p,wm[w]);
      if constexpr(APA_SPO2_MUTANT==2) {
        p=carry;for(int z=t;z<j;++z)p=fmaxf(p,bulk[base+z]);
      }
      // APA_SPO5_BEGIN compare_mutant
    // DeMillo et al. 1978: equality-boundary mutation, zero in production.
    bool refine=j<count && (APA_SPO5_MUTANT==6?x>p-delta:x>=p-delta);
    // APA_SPO5_END compare_mutant
      if constexpr(APA_SPO2_MUTANT==3)refine=j<count && x>p-delta;
      if constexpr(APA_SPO2_MUTANT==6)if(j==0 && c>first)refine=false;
      if(j<count) {
        mask[base+j]=refine;score[base+j]=x;
        if(refine && (APA_SPO2_MUTANT!=4 || g==0))any[j]=1;
        if constexpr(DIAG)prefix_dump[base+j]=p;
      }
      for(int w=0;w<4;++w)carry=fmaxf(carry,wm[w]);__syncthreads();
    }
  }
  // Gather each selected K row once for the union of all heads' masks.
  for(int t=0;t<count;t+=TILE) {
    int n=min(TILE,count-t);
    stage(k+((int64_t)bh*S+start+t)*D,tile,D,n,any+t);__syncthreads();
    if constexpr(!W) {
      for(int i=tid;i<n*D;i+=NT)if(any[t+i/D])trans[(i%D)*TILE+i/D]=ld<T>(tile,i);
      __syncthreads();
    }
    for(int g=0;g<G;++g) {
      int row=b*H+h0+g;int64_t base=(int64_t)row*S+start+t;
      if constexpr(W) {
        for(int j=warp;j<n;j+=4)if(mask[base+j] && any[t+j]) {
          float ex=dot<T,true>(q+(int64_t)row*D,tile+j*D,D,scale);
          if(lane==0)score[base+j]=ex;
        }
      } else {
        if(tid<n && mask[base+tid] && any[t+tid])score[base+tid]=dot_trans<T>(q+(int64_t)row*D,trans,tid,D,TILE,scale);
      }
    }
    __syncthreads();
  }
  // Stable chunk softmax (Milakov/Gimelshein normalizer, all scores known).
  for(int g=0;g<G;++g) {
    int row=b*H+h0+g;int64_t base=(int64_t)row*S+start,pr=(int64_t)row*NC+c;
    float mx=-INFINITY;for(int j=tid;j<count;j+=NT)mx=fmaxf(mx,score[base+j]);
    red[tid]=mx;__syncthreads();
    for(int o=NT/2;o;o>>=1){if(tid<o)red[tid]=fmaxf(red[tid],red[tid+o]);__syncthreads();}
    mx=red[0];__syncthreads();Acc sum=0;
    for(int j=tid;j<count;j+=NT){float w=__expf(score[base+j]-mx);score[base+j]=w;sum+=w;}
    sum_red[tid]=sum;__syncthreads();
    for(int o=NT/2;o;o>>=1){if(tid<o)sum_red[tid]+=sum_red[tid+o];__syncthreads();}
    if(tid==0){pm[pr]=mx;pl[pr]=(float)sum_red[0];}
    for(int d=tid;d<VD;d+=NT)wide_acc[pr*VD+d]=0;
    __syncthreads();
  }
  // V tiles are read once, then reused across all grouped query heads.
  for(int t=0;t<count;t+=TILE) {
    int n=min(TILE,count-t);stage(v+((int64_t)bh*S+start+t)*VD,tile,VD,n);__syncthreads();
    for(int g=0;g<G;++g) {
      int row=b*H+h0+g;int64_t pr=(int64_t)row*NC+c;
      if(tid<n)tmp[tid]=score[(int64_t)row*S+start+t+tid];__syncthreads();
      for(int d=tid;d<VD;d+=NT) {
        // Prior art: wider accumulation, Higham (2002), Accuracy and Stability.
        // FP32 products, FP64 sum; only PV/normalizer changes, dots unchanged.
        Acc acc=wide_acc[pr*VD+d];for(int j=0;j<n;++j)acc+=(Acc)__fmul_rn(tmp[j],ld<T>(tile,j*VD+d));
        wide_acc[pr*VD+d]=acc;
      }
      __syncthreads();
    }
  }
  for(int g=0;g<G;++g) {
    int64_t pr=(int64_t)(b*H+h0+g)*NC+c;
    for(int d=tid;d<VD;d+=NT)pa[pr*VD+d]=(float)wide_acc[pr*VD+d];
  }
}
} // namespace spo2
// APA_SPO2_D1_END implementation

#include "parallel.cuh"

// APA_SPO5_BEGIN tensor_include
#include "tensor.cuh"
// APA_SPO5_END tensor_include

// APA_SPQ1_BEGIN include
#include "packed.cuh"
// APA_SPQ1_END include
