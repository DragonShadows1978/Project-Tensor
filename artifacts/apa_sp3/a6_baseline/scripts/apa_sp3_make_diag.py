#!/usr/bin/env python3
"""Generate diagnostic copies after verifying every production source hash.

Prior art: Perry APA draft (2026) z-score selection; existing Project-Tensor
CUDA reductions and online softmax (Milakov/Gimelshein 2018; Dao FA2 2023).
We take the literal existing kernel bodies, inserting only mask stores. These
copies are separate from production. SP3 contributes observational plumbing.
"""
from apa_sp3_common import ROOT, BUILD, verify_sources


def extract(source, name):
    start = source.index('__global__ void ' + name + '(')
    opening = source.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def generated():
    verify_sources()
    s = (ROOT / 'tensor_cuda/src/kernels.cu').read_text()
    blend = extract(s, 'apa_blend_softmax_kernel2')
    blend = blend.replace('int window) {', 'int window, uint8_t* selected) {')
    pos = blend.index('  // Single fused pass:')
    blend = blend[:pos] + '''
  // SP3 observation only: literal decision, after unchanged statistics.
  for (int j=tid; j<S; j+=nt) {
    float bk=ld<T>(brow,j);
    bool valid=BOUNDED ? (j>=lo && j<hi) : (bk>MASK_LIM);
    selected[(int64_t)r*S+j]=valid && fabsf(bk)>=thr;
  }
''' + blend[pos:]
    selective = extract(s, 'apa_selective_kernel')
    selective = selective.replace('int is_causal, int KVH, int group) {',
                                  'int is_causal, int KVH, int group, uint8_t* selected) {')
    selective = selective.replace('      if (fabsf(bulk) >= thr) {',
        '      if (lane == 0) selected[(int64_t)row*S+j] = fabsf(bulk)>=thr;\n      if (fabsf(bulk) >= thr) {', 1)
    # Only WCOOP=true, DMAX=128 is instantiated for the model's D=96.
    selective = selective.replace('  // Pass 1: bulk scores',
        '  for (int j=s_max+tid; j<S; j+=nt) selected[(int64_t)row*S+j]=0;\n  // Pass 1: bulk scores', 1)
    pre = '''// GENERATED: literal pinned production bodies plus diagnostic stores.
// Prior art: Perry APA 2026, Milakov/Gimelshein 2018, Dao FA2 2023.
#include "tc/core.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
namespace apa_sp3 {
using namespace tc;
template<typename T> __device__ float ld(const T* p,int64_t i) { return (float)p[i]; }
template<typename T> __device__ void st(T* p,int64_t i,float v) { p[i]=(T)v; }
'''
    post = '''
// Prior art: literal warp/FMA dot order from the existing SP1 prefill kernel.
// Observational score probe only; no attention path or selector changes.
template<typename T> __global__ void bulk_scores_kernel(const T* q,const T* kq,float* o,int H,int L,int S,int D,float scale) {
 int row=blockIdx.x,lane=threadIdx.x,h=(row/L)%H,b=row/(L*H);
 float qr[4];
 #pragma unroll
 for(int t=0;t<4;++t) {int d=lane+32*t;qr[t]=d<D ? ld<T>(q,(int64_t)row*D+d):0.f;}
 for(int j=0;j<S;++j) {
   int64_t koff=(((int64_t)b*H+h)*S+j)*D;float dot=0.f;
   #pragma unroll
   for(int t=0;t<4;++t) {int d=lane+32*t;if(d<D) dot+=qr[t]*ld<T>(kq,koff+d);}
   #pragma unroll
   for(int off=16;off>0;off>>=1) dot+=__shfl_down_sync(0xffffffffu,dot,off);
   float bulk=__shfl_sync(0xffffffffu,dot,0)*scale;
   if(lane==0)o[(int64_t)row*S+j]=bulk;
 }
}
NDArray bulk_scores(const NDArray& q,const NDArray& kq,float scale) {
 int B=q.shape[0],H=q.shape[1],L=q.shape[2],S=kq.shape[2],D=q.shape[3];
 NDArray o({B,H,L,S},DType::Float32,q.device);
 #define GO(T) bulk_scores_kernel<T><<<B*H*L,32>>>((T*)q.data_ptr(),(T*)kq.data_ptr(),(float*)o.data_ptr(),H,L,S,D,scale);
 if(q.dtype==DType::BFloat16) { GO(__nv_bfloat16) } else if(q.dtype==DType::Float32) { GO(float) } else { GO(__half) }
 #undef GO
 cuda_check_last("apa_sp3_bulk_scores");return o;
}
std::pair<NDArray,NDArray> blend(const NDArray& b,const NDArray& r,float z,int Lq,int row0) {
 NDArray o(b.shape,b.dtype,b.device),m(b.shape,DType::Uint8,b.device);
 int S=b.shape[3], L=b.shape[2], rows=b.numel()/S;
 #define GO(T) if(Lq>0) apa_blend_softmax_kernel2<T,true><<<rows,256>>>((T*)b.data_ptr(),(T*)r.data_ptr(),(T*)o.data_ptr(),S,z,L,Lq,row0,0,(uint8_t*)m.data_ptr()); else apa_blend_softmax_kernel2<T,false><<<rows,256>>>((T*)b.data_ptr(),(T*)r.data_ptr(),(T*)o.data_ptr(),S,z,L,0,row0,0,(uint8_t*)m.data_ptr());
 if(b.dtype==DType::BFloat16) { GO(__nv_bfloat16) } else if(b.dtype==DType::Float32) { GO(float) } else { GO(__half) }
 #undef GO
 cuda_check_last("apa_sp3_blend_diagnostic"); return {o,m};
}
std::pair<NDArray,NDArray> selective(const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,float scale,float z,bool causal) {
 int B=q.shape[0],H=q.shape[1],L=q.shape[2],S=k.shape[2],D=q.shape[3],VD=v.shape[3];
 NDArray o({B,H,L,VD},q.dtype,q.device),m({B,H,L,S},DType::Uint8,q.device);
 #define GO(T) apa_selective_kernel<T,128,true><<<B*H*L,128>>>((T*)q.data_ptr(),(T*)k.data_ptr(),(T*)kq.data_ptr(),(T*)v.data_ptr(),(T*)o.data_ptr(),B,H,L,S,D,VD,scale,z,causal,H,1,(uint8_t*)m.data_ptr());
 if(q.dtype==DType::BFloat16) { GO(__nv_bfloat16) } else if(q.dtype==DType::Float32) { GO(float) } else { GO(__half) }
 #undef GO
 cuda_check_last("apa_sp3_selective_diagnostic"); return {o,m};
}
} // namespace
'''
    return pre + 'template<typename T,bool BOUNDED>\n' + blend + '\n' + \
        'template<typename T,int DMAX,bool WCOOP>\n' + selective + '\n' + post


if __name__ == '__main__':
    (BUILD / 'diagnostics.cu').write_text(generated())
