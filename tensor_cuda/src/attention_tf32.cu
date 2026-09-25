// PT-TF32-1: explicit FP32-storage attention variants; defaults unchanged.
// Prior art: NVIDIA Ampere TF32 (2020), CUDA 11 WMMA m16n16k8, taken APIs.
// https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html
// FlashAttention (Dao et al. 2022) / FlashAttention-2 (Dao 2023), taken via
// Project-Tensor BP-KERNEL-2/3/4 (2026): owner tiles, recomputation, online
// softmax, output-dot VJP, grouped heads and APA absolute z-score selection.
// https://arxiv.org/abs/2307.08691 ; docs/BP_KERNEL_{2,3,4}_LEDGER.md
// Ours: separate TF32 fragments, FP32 shared operands/coefficient storage,
// lifetime-based scratch reuse and strict FP32 opt-in launch/diagnostic guards.
// p and dS NEVER pass through BF16. Their MMA operands DO round to TF32;
// FP32 storage/accumulation is not full-FP32 multiplication precision.
#include "tc/tf32.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <type_traits>

namespace tc {
namespace pt_tf32 {
constexpr int P=128, T=16;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using namespace nvcuda;
using AF=wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major>;
using BR=wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major>;
using BC=wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major>;
using CF=wmma::fragment<wmma::accumulator,16,16,8,float>;

// CUDA 11 requires explicit TF32 conversion. Convert registers, preserving
// shared FP32 Q/K for selected scalar scores and dO for the output-dot term.
template<class F> __device__ void round_fragment(F& f) {
  #pragma unroll
  for(int i=0;i<F::num_elements;++i) f.x[i]=wmma::__float_to_tf32(f.x[i]);
}
// PT-TF32-2. Prior art: Ootomo & Yokota (2022), arXiv:2203.03341,
// residual decomposition for tensor-core products. Taken: hi/lo operands;
// ours: three products at APA predicate and cancellation sites, with a
// separate FP32 correction accumulator. Not full FP32/SGEMM equivalence.
// A*B ~= Ah*Bh + Ah*Bl + Al*Bh; omitted Al*Bl is O(u_tf32^2).
template<class F> __device__ void split_fragment(F& hi,F& lo) {
  #pragma unroll
  for(int i=0;i<F::num_elements;++i) {
    float original=hi.x[i];
    hi.x[i]=wmma::__float_to_tf32(original);
    lo.x[i]=wmma::__float_to_tf32(original-hi.x[i]);
  }
}
__device__ void scores(const float* a,const float* b,float* dst,int width) {
  AF af,al; BC bf,bl; CF cf,correction;
  wmma::fill_fragment(cf,0.f); wmma::fill_fragment(correction,0.f);
  for(int d=0;d<width;d+=8) {
    wmma::load_matrix_sync(af,a+d,P); wmma::load_matrix_sync(bf,b+d,P);
    split_fragment(af,al); split_fragment(bf,bl);
    wmma::mma_sync(cf,af,bf,cf);
    wmma::mma_sync(correction,af,bl,correction);
    wmma::mma_sync(correction,al,bf,correction);
  }
  #pragma unroll
  for(int n=0;n<CF::num_elements;++n) cf.x[n]+=correction.x[n];
  wmma::store_matrix_sync(dst,cf,T,wmma::mem_row_major);
}
template<bool PRECISE=false>
__device__ void outer(const float* a,const float* b,int col,CF& c) {
  AF af,al; BR bf,bl; CF correction;
  if constexpr(PRECISE) wmma::fill_fragment(correction,0.f);
  #pragma unroll
  for(int k=0;k<T;k+=8) {
    wmma::load_matrix_sync(af,a+k,T); wmma::load_matrix_sync(bf,b+k*P+col,P);
    if constexpr(PRECISE) {
      split_fragment(af,al); split_fragment(bf,bl);
      wmma::mma_sync(correction,af,bl,correction);
      wmma::mma_sync(correction,al,bf,correction);
    } else {round_fragment(af); round_fragment(bf);}
    wmma::mma_sync(c,af,bf,c);
  }
  if constexpr(PRECISE) {
    #pragma unroll
    for(int n=0;n<CF::num_elements;++n) c.x[n]+=correction.x[n];
  }
}
#endif

template<bool KEY>
__global__ void backward(const float* q, const float* k, const float* kq, const float* v,
    const float* dout, const float* out, const float* lse, const float* thr,
    float* dq, float* dk, float* dv, int H,int KVH,int L,int S,int D,int VD,
    float scale,bool causal) {
#if __CUDA_ARCH__ >= 800
  // Four warps own disjoint gradient columns. Warp 0 forms dense scores;
  // all threads stage inputs and evaluate selected scores / softmax VJP.
  int tid=threadIdx.x, warp=tid/32, group=H/KVH;
  int owner=blockIdx.x*T, bh=blockIdx.y;
  int b=bh/(KEY?KVH:H), kh=KEY?bh%KVH:(bh%H)/group;
  __shared__ __align__(32) float qs[T*P], dos[T*P], ks[T*P], kqs[T*P], vs[T*P];
  __shared__ __align__(32) float sel[T*T], bulkds[T*T], prob[T*T];
  __shared__ __align__(32) float bulk[T*T], dov[T*T];
  // Input Q is dead at final store; reuse it as gradient scratch (46,144 B total).
  float* scratch=qs;
  __shared__ float rd[T];
  CF accum[2], accumv[2];
  for(int n=0;n<2;++n) {wmma::fill_fragment(accum[n],0.f);wmma::fill_fragment(accumv[n],0.f);}
  // Owner input is loaded once per block, never once per pair.
  for(int z=tid;z<T*P;z+=128) {
    int r=z/P,d=z%P, pos=owner+r;
    if constexpr(KEY) {
      int64_t row=((int64_t)b*KVH+kh)*S+pos;
      ks[z]=(pos<S && d<D)?k[row*D+d]:0.f;
      kqs[z]=(pos<S && d<D)?kq[row*D+d]:0.f;
      vs[z]=(pos<S && d<VD)?v[row*VD+d]:0.f;
    } else {
      int64_t row=(int64_t)bh*L+pos;
      qs[z]=(pos<L && d<D)?q[row*D+d]:0.f;
      dos[z]=(pos<L && d<VD)?dout[row*VD+d]:0.f;
    }
  }
  __syncthreads();
  int heads=KEY?group:1;
  for(int gh=0;gh<heads;++gh) {
    int h=KEY?kh*group+gh:bh%H;
    int limit=KEY?L:S;
    // Skip wholly invisible tiles only; partial tiles use per-pair visibility.
    int first=(KEY && causal)?max(0,owner-(S-L))/T*T:0;
    if(!KEY && causal) limit=min(S,S-L+owner+T);
    for(int start=first;start<limit;start+=T) {
      int qi=KEY?start:owner, kj=KEY?owner:start;
      for(int z=tid;z<T*P;z+=128) {
        int r=z/P,d=z%P;
        if constexpr(KEY) {
          int64_t row=((int64_t)b*H+h)*L+qi+r;
          qs[z]=(qi+r<L && d<D)?q[row*D+d]:0.f;
          dos[z]=(qi+r<L && d<VD)?dout[row*VD+d]:0.f;
        } else {
          int64_t row=((int64_t)b*KVH+kh)*S+kj+r;
          ks[z]=(kj+r<S && d<D)?k[row*D+d]:0.f;
          kqs[z]=(kj+r<S && d<D)?kq[row*D+d]:0.f;
          vs[z]=(kj+r<S && d<VD)?v[row*VD+d]:0.f;
        }
      }
      __syncthreads();
      if(tid<T) {
        float dot=0.f;
        if(qi+tid<L) {
          int64_t row=((int64_t)b*H+h)*L+qi+tid;
          for(int d=0;d<VD;++d) dot+=(dos[tid*P+d])*(out[row*VD+d]);
        }
        rd[tid]=dot;
      }
      if(warp==0) {
        scores(qs,kqs,bulk,D); scores(dos,vs,dov,VD);
      }
      __syncthreads();
      for(int z=tid;z<T*T;z+=128) {
        int i=z/T,j=z%T; int64_t row=((int64_t)b*H+h)*L+qi+i;
        bool visible=qi+i<L && kj+j<S && (!causal || kj+j<=S-L+qi+i);
        float sc=bulk[z]*scale, p=0.f, ds=0.f;
        // Prior art: singleton softmax derivative is zero (standard calculus).
        // Ours: enforce the exact identity rather than subtract rounded dP/D.
        const bool singleton=(causal?S-L+qi+i+1:S)==1;
        bool selected=visible && (singleton || fabsf(sc)>=thr[row]);
        if(selected) {
          {
            float dot=0.f;
            for(int d=0;d<D;++d) dot+=(qs[i*P+d])*(ks[j*P+d]);
            sc=dot*scale;
          }
        }
        if(visible) {
          p=singleton?1.f:__expf(sc-lse[row]);
          ds=singleton?0.f:p*(dov[z]-rd[i])*scale;
        }
        int dst=KEY?j*T+i:z;
        sel[dst]=(selected?ds:0.f);
        bulkds[dst]=(selected?0.f:ds);
        prob[dst]=(p);
      }
      __syncthreads();
      for(int n=0;n<2;++n) {
        int col=(warp+4*n)*16;
        if(col<D) {
          outer(sel,KEY?qs:ks,col,accum[n]);
          if constexpr(!KEY) outer(bulkds,kqs,col,accum[n]);
        }
        // PT-TF32-3 edge-case attribution: NVIDIA TF32 (2020), Higham (2002).
        // RN_TF32(p) is discontinuous: an FP32-score/exp error around a
        // midpoint changes dV by one bin * RN_TF32(dO). This can exceed a
        // componentwise near-zero tolerance while aggregate FP64 error passes.
        // CPU witnesses/unchanged legacy assertions are in PT_TF32_3_LEDGER.md;
        // no full-FP32-product or native legacy-test-pass claim is made here.
        if constexpr(KEY) if(col<VD) outer(prob,dos,col,accumv[n]);
      }
      __syncthreads();
    }
  }
  for(int n=0;n<2;++n) {
    int col=(warp+4*n)*16;
    if(col<D) wmma::store_matrix_sync(scratch+col,accum[n],P,wmma::mem_row_major);
  }
  __syncthreads();
  for(int z=tid;z<T*D;z+=128) {
    int r=z/D,d=z%D;
    if(owner+r<(KEY?S:L)) {
      int64_t row=KEY?((int64_t)b*KVH+kh)*S+owner+r:(int64_t)bh*L+owner+r;
      (KEY?dk:dq)[row*D+d]=(scratch[r*P+d]);
    }
  }
  if constexpr(KEY) {
    __syncthreads();
    for(int n=0;n<2;++n) {
      int col=(warp+4*n)*16;
      if(col<VD) wmma::store_matrix_sync(scratch+col,accumv[n],P,wmma::mem_row_major);
    }
    __syncthreads();
    for(int z=tid;z<T*VD;z+=128) {
      int r=z/VD,d=z%VD;
      if(owner+r<S) dv[(((int64_t)b*KVH+kh)*S+owner+r)*VD+d]=(scratch[r*P+d]);
    }
  }
#else
  asm volatile("trap;"); // Host launcher rejects pre-Ampere devices.
#endif
}

template<bool DIAG>
__global__ void forward(const float* q,const float* k,const float* kq,const float* v,
    float* out,float* lse,float* thr,unsigned char* selected,float* cycles,
    int H,int KVH,int L,int S,int D,int VD,float scale,float zthr,bool causal) {
#if __CUDA_ARCH__ >= 800
  int tid=threadIdx.x,warp=tid/32,qi=blockIdx.x*16,bh=blockIdx.y;
  int kh=(bh/H)*KVH+(bh%H)/(H/KVH);
  int64_t qb=(int64_t)bh*L*D,kb=(int64_t)kh*S*D,vb=(int64_t)kh*S*VD;
  __shared__ __align__(32) float qs[16*128],ks[16*128],kqs[16*128],vs[16*128],ps[256];
  __shared__ __align__(32) float score[256],acc[16*128];
  // KQ is dead after bulk scores. Reuse it for P@V, after the barrier.
  float* tmp=kqs;
  // PT-TF32-3. Prior art: Higham (2002), wider-precision reduction, taken.
  // Ours: protect the *saved predicate threshold*, not only Q.K or P.V.
  // FP32 sequential moment sums drift O(S*u32); rare mask flips inject whole
  // dK contributions even when out/LSE and the global flip-rate gate pass.
  // FP64 moment accumulation removes that length-dependent rounding site;
  // round once to the public FP32 state before BOTH forward/backward compare.
  // Residual score/threshold rounding remains, so this is not exact FP64 APA.
  __shared__ double sums[16],squares[16];
  __shared__ float th[16],m[16],den[16],corr[16];
  unsigned long long tick=0;float phase[7]={0,0,0,0,0,0,0};
  if constexpr(DIAG) {if(tid==0)tick=clock64();}
  for(int n=tid;n<16*128;n+=128) {
    int i=n/128,d=n%128;
    qs[n]=(qi+i<L && d<D)?q[qb+(int64_t)(qi+i)*D+d]:0.f;
    acc[n]=0.f;
  }
  if(tid<16) {sums[tid]=squares[tid]=den[tid]=0.f;m[tid]=-1e30f;}
  __syncthreads();
  if constexpr(DIAG) {if(tid==0){phase[0]+=float(clock64()-tick);tick=clock64();}}
  int limit=causal?min(S,S-L+qi+16):S;
  for(int pass=0;pass<2;++pass) {
    for(int kj=0;kj<limit;kj+=16) {
      for(int n=tid;n<16*128;n+=128) {
        int j=kj+n/128,d=n%128;
        kqs[n]=(j<S && d<D)?kq[kb+(int64_t)j*D+d]:0.f;
        if(pass) {
          ks[n]=(j<S && d<D)?k[kb+(int64_t)j*D+d]:0.f;
          vs[n]=(j<S && d<VD)?v[vb+(int64_t)j*VD+d]:0.f;
        }
      }
      __syncthreads();
      if constexpr(DIAG) {if(tid==0){phase[0]+=float(clock64()-tick);tick=clock64();}}
      if(warp==0) scores(qs,kqs,score,D);
      __syncthreads();
      if constexpr(DIAG) {if(tid==0){phase[1]+=float(clock64()-tick);tick=clock64();}}
      if(!pass) {
        if(tid<16 && qi+tid<L) {
          int count=causal?S-L+qi+tid+1:S;
          for(int j=0;j<16 && kj+j<count;++j) {
            double a=double(fabsf(score[tid*16+j]*scale));
            sums[tid]+=a;squares[tid]+=a*a;
          }
        }
        __syncthreads();
        if constexpr(DIAG) {if(tid==0){phase[2]+=float(clock64()-tick);tick=clock64();}}
      } else {
        for(int n=tid;n<256;n+=128) {
          int i=n/16,j=n%16,count=causal?S-L+qi+i+1:S;
          bool visible=qi+i<L && kj+j<count;
          float bulk=score[n]*scale;
          bool sel=visible && (count==1 || fabsf(bulk)>=th[i]);
          if constexpr(DIAG) {
            if(qi+i<L && kj+j<S) selected[((int64_t)bh*L+qi+i)*S+kj+j]=sel;
          }
          float ex=0.f;
          if(sel) for(int d=0;d<D;++d) ex+=(qs[i*128+d])*(ks[j*128+d]);
          score[n]=visible?(sel?ex*scale:bulk):-1e30f;
        }
        __syncthreads();
        if constexpr(DIAG) {if(tid==0){phase[3]+=float(clock64()-tick);tick=clock64();}}
        if(tid<16) {
          int count=causal?S-L+qi+tid+1:S;
          float nm=m[tid];
          for(int j=0;j<16;++j) nm=fmaxf(nm,score[tid*16+j]);
          float c=__expf(m[tid]-nm),total=0.f;
          for(int j=0;j<16;++j) {
            float p=(qi+tid<L && kj+j<count)?__expf(score[tid*16+j]-nm):0.f;
            total+=p;ps[tid*16+j]=(p);
          }
          corr[tid]=c;den[tid]=den[tid]*c+total;m[tid]=nm;
        }
        __syncthreads();
        if constexpr(DIAG) {if(tid==0){phase[4]+=float(clock64()-tick);tick=clock64();}}
        for(int col=warp*16;col<128;col+=64) {
          CF frag;wmma::fill_fragment(frag,0.f);
          // Correct P@V as well: its saved output feeds D_i=dO_i.out_i.
          // The same residual-decomposition prior art is cited above.
          outer<true>(ps,vs,col,frag);
          wmma::store_matrix_sync(tmp+col,frag,128,wmma::mem_row_major);
        }
        __syncthreads();
        for(int n=tid;n<16*128;n+=128) acc[n]=acc[n]*corr[n/128]+tmp[n];
        __syncthreads();
        if constexpr(DIAG) {if(tid==0){phase[5]+=float(clock64()-tick);tick=clock64();}}
      }
    }
    if(!pass) {
      if(tid<16 && qi+tid<L) {
        double count=double(causal?S-L+qi+tid+1:S),mean=sums[tid]/count;
        // Population variance of one sample is exactly zero. FP32 FMA can
        // otherwise leave a positive cancellation residual (PT-TF32-2).
        th[tid]=float(count==1.?mean:mean+double(zthr)*sqrt(fmax(squares[tid]/count-mean*mean,0.)));
        thr[(int64_t)bh*L+qi+tid]=th[tid];
      }
      // Padded rows have no visible pairs; initialize their threshold too.
      if(tid<16 && qi+tid>=L)th[tid]=0.f;
      __syncthreads();
      if constexpr(DIAG) {if(tid==0){phase[2]+=float(clock64()-tick);tick=clock64();}}
    }
  }
  for(int n=tid;n<16*128;n+=128) {
    int i=n/128,d=n%128;
    if(qi+i<L && d<VD) {
      const int count=causal?S-L+qi+i+1:S;
      out[((int64_t)bh*L+qi+i)*VD+d]=count==1?v[vb+d]:acc[n]/den[i];
    }
  }
  if(tid<16 && qi+tid<L)lse[(int64_t)bh*L+qi+tid]=m[tid]+logf(fmaxf(den[tid],1e-30f));
  __syncthreads();
  if constexpr(DIAG) {
    if(tid==0) {
      phase[6]+=float(clock64()-tick);
      for(int p=0;p<7;++p)cycles[((int64_t)bh*gridDim.x+blockIdx.x)*7+p]=phase[p];
    }
  }
#else
  asm volatile("trap;");
#endif
}

__global__ void a_mask(const float* q,const float* kq,const float* thr,unsigned char* mask,
    int H,int KVH,int L,int S,int D,float scale,bool causal,bool wcoop) {
  int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32,i=row%L;
  int kh=(row/L/H)*KVH+(row/L%H)/(H/KVH);
  for(int j=warp;j<S;j+=4) {
    float dot=0.f;
    if(wcoop) {
      for(int d=lane;d<D;d+=32) dot+=(q[(int64_t)row*D+d])*(kq[((int64_t)kh*S+j)*D+d]);
      for(int off=16;off>0;off>>=1)dot+=__shfl_down_sync(0xffffffff,dot,off);
    } else if(lane==0) {
      for(int d=0;d<D;++d)dot+=(q[(int64_t)row*D+d])*(kq[((int64_t)kh*S+j)*D+d]);
    }
    if(lane==0)mask[(int64_t)row*S+j]=(!causal || j<S-L+i+1) && fabsf(dot*scale)>=thr[row];
  }
}
} // namespace pt_tf32

// Prior art: BP-KERNEL-1/4 validation (Project-Tensor 2026), taken. Ours:
// FP32-only contract and Ampere guard; reject before any kernel launch.
static void tf32_check_cuda(cudaError_t e) {
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void tf32_check_inputs(const NDArray& q,const NDArray& k,
    const NDArray& kq,const NDArray& v,float scale) {
  if(q.ndim()!=4 || k.ndim()!=4 || kq.ndim()!=4 || v.ndim()!=4)
    throw std::runtime_error("PT-TF32-1: rank 4 required");
  auto B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3];
  auto KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(B<=0 || H<=0 || KVH<=0 || H%KVH || L<=0 || S<L ||
      D<=0 || D>128 || VD<=0 || VD>128 || H>65535 || B>65535/H ||
      L>INT_MAX/(B*H)-16 || S>INT_MAX-16 ||
      k.shape!=Shape{B,KVH,S,D} || kq.shape!=k.shape ||
      v.shape!=Shape{B,KVH,S,VD} || !std::isfinite(scale))
    throw std::runtime_error("PT-TF32-1: invalid geometry/scale; D/VD <=128, S>=L required");
  for(auto x:{&q,&k,&kq,&v})
    if(!x->defined() || !x->device.is_cuda() ||
       x->device.index!=q.device.index || x->dtype!=DType::Float32)
      throw std::runtime_error("PT-TF32-1: same CUDA FP32 required");
}
static void tf32_check_device(const NDArray& q) {
  int active=-1,major=0;
  tf32_check_cuda(cudaGetDevice(&active));
  if(active!=q.device.index) throw std::runtime_error("PT-TF32-1: current device mismatch");
  tf32_check_cuda(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,active));
  if(major<8) throw std::runtime_error("PT-TF32-1: TF32 WMMA requires SM80 or newer");
}

std::tuple<NDArray,NDArray,NDArray,NDArray,NDArray> apa_selective_fwd_tf32(
    const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,
    float scale,float zthr,bool causal,const std::string& variant,bool diagnostic) {
  if(variant!="a" && variant!="h_tf32") throw std::runtime_error("PT-TF32-1: expected a/h_tf32");
  tf32_check_inputs(q,k,kq,v,scale);
  if(!std::isfinite(zthr)) throw std::runtime_error("PT-TF32-1: finite zthr required");
  if(variant=="h_tf32") tf32_check_device(q);
  int B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3];
  int KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  NDArray out,lse,thr,mask,cycles;
  if(diagnostic) {
    mask=NDArray::zeros({B,H,L,S},DType::Uint8,q.device);
    cycles=NDArray::zeros({B,H,(L+15)/16,7},DType::Float32,q.device);
  }
  if(variant=="a") {
    std::tie(out,lse,thr)=apa_selective_fwd_train(q,k,kq,v,scale,zthr,causal);
    // Shipped a's decode heuristic: TC_APA_SM_COUNT=56, L=1, rows<112.
    // Taken verbatim rule from kernels.cu; receipt replay only, no dispatch change.
    bool wcoop=!(std::max(D,VD)<=64 && L==1 && B*H*L<112);
    if(diagnostic) pt_tf32::a_mask<<<B*H*L,128>>>(
        (float*)q.data_ptr(),(float*)kq.data_ptr(),(float*)thr.data_ptr(),
        (unsigned char*)mask.data_ptr(),H,KVH,L,S,D,scale,causal,wcoop);
  } else {
    out=NDArray({B,H,L,VD},DType::Float32,q.device);
    lse=NDArray({B,H,L},DType::Float32,q.device);
    thr=NDArray({B,H,L},DType::Float32,q.device);
    auto launch=[&](auto tag) {
      constexpr bool DIAG=decltype(tag)::value;
      pt_tf32::forward<DIAG><<<dim3((L+15)/16,B*H),128>>>(
          (float*)q.data_ptr(),(float*)k.data_ptr(),(float*)kq.data_ptr(),(float*)v.data_ptr(),
          (float*)out.data_ptr(),(float*)lse.data_ptr(),(float*)thr.data_ptr(),
          DIAG?(unsigned char*)mask.data_ptr():nullptr,DIAG?(float*)cycles.data_ptr():nullptr,
          H,KVH,L,S,D,VD,scale,zthr,causal);
    };
    if(diagnostic) launch(std::true_type{}); else launch(std::false_type{});
  }
  cuda_check_last("PT-TF32-1 forward");
  return {out,lse,thr,mask,cycles};
}

std::tuple<NDArray,NDArray,NDArray> apa_selective_bwd_tf32(
    const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,
    const NDArray& dO,const NDArray& lse,const NDArray& thr,const NDArray& out,
    float scale,bool causal) {
  tf32_check_inputs(q,k,kq,v,scale);
  int B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3];
  int KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(dO.shape!=Shape{B,H,L,VD} || out.shape!=dO.shape ||
      lse.shape!=Shape{B,H,L} || thr.shape!=lse.shape)
    throw std::runtime_error("PT-TF32-1: saved forward/dO shape mismatch");
  for(auto x:{&dO,&out,&lse,&thr})
    if(!x->defined() || !x->device.is_cuda() || x->device.index!=q.device.index || x->dtype!=DType::Float32)
      throw std::runtime_error("PT-TF32-1: saved forward/dO must be same CUDA FP32");
  tf32_check_device(q);
  NDArray dq(q.shape,q.dtype,q.device),dk(k.shape,k.dtype,k.device),dv(v.shape,v.dtype,v.device);
  auto half=[&](auto tag) {
    constexpr bool KEY=decltype(tag)::value;
    pt_tf32::backward<KEY><<<dim3(((KEY?S:L)+15)/16,B*(KEY?KVH:H)),128>>>(
        (float*)q.data_ptr(),(float*)k.data_ptr(),(float*)kq.data_ptr(),(float*)v.data_ptr(),
        (float*)dO.data_ptr(),(float*)out.data_ptr(),(float*)lse.data_ptr(),(float*)thr.data_ptr(),
        (float*)dq.data_ptr(),(float*)dk.data_ptr(),(float*)dv.data_ptr(),H,KVH,L,S,D,VD,scale,causal);
    cuda_check_last("PT-TF32-1 backward");
  };
  half(std::false_type{});half(std::true_type{});
  return {dq,dk,dv};
}
} // namespace tc
