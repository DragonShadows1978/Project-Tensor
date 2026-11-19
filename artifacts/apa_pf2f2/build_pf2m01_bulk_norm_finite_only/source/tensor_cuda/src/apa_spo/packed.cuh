// APA_SPQ1_BEGIN packed_bulk
// Prior art: TurboQuant (Zandieh et al. 2025), arXiv:2504.19874, rotated
// Lloyd-Max codes; symmetric INT4 inherited from apa_int4_pack_kernel (2026).
// Flash-Decoding (2023), FlashInfer (Ye et al. 2025), inherited split/GQA.
// NVIDIA warp shuffle and shared staging; D1/SP1.1 (2026) ordered FMA trees.
// Ours: packed operand adapter. No new quantizer or selection rule.
namespace spq1 {
template<class T> __global__ void rotate(const T* q,const float* r,float* out,int H,int KVH,int D,bool tq) {
  int row=blockIdx.x,d=threadIdx.x;if(d>=D)return;
  if(!tq){out[(int64_t)row*D+d]=ld<T>(q,(int64_t)row*D+d);return;}
  int kh=(row%H)/(H/KVH);float x=0;
  for(int i=0;i<D;++i)x=__fmaf_rn(ld<T>(q,(int64_t)row*D+i),r[((int64_t)kh*D+d)*D+i],x);
  out[(int64_t)row*D+d]=x; // quant.py rotates row vectors by R^T.
}
__global__ void pack_tq(const float* rotated,const float* norms,const float* bnd,
                      uint8_t* codes,float* scales,int BH,int N,int D,int capacity,int start) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=(int64_t)BH*N*(D/2))return;
  int pair=i%(D/2);int64_t key=i/(D/2),dst=(key/N*capacity+start+key%N);
  int a=0,b=0;float x=rotated[key*D+2*pair],y=rotated[key*D+2*pair+1];
  for(int c=0;c<15;++c){a+=x>bnd[c];b+=y>bnd[c];}
  codes[dst*(D/2)+pair]=(uint8_t)(a|(b<<4));if(pair==0)scales[dst]=norms[key];
}
// APA_P1F1_BEGIN int4_pack
// R1 (P1R-F01): the packed INT4 key quantizer. Arithmetic identical, operation
// for operation, to kernels.cu apa_int4_pack_rows_kernel (amax by fmaxf over
// |k|, scale = amax/7, safe = scale>0 ? scale : 1, codes round-away and clamp
// to [-7,7]); one addition: if any element of the key row is non-finite, the
// stored scale is a quiet NaN. fmaxf drops NaN (IEEE 754-2008 maxNum), so the
// original stored a finite scale and a code computed from (int)NaN, losing a
// NaN key; with this carry every score of that key is NaN, as on the dense
// path. A row with only finite elements stores the same scale and codes bit
// for bit (L55: the carry touches only rows with a non-finite element).
// Prior art: symmetric absmax INT4 (apa_int4_pack_rows_kernel, Project-Tensor
// 2026), taken unchanged; IEEE 754-2008 NaN propagation. Ours: the carry.
template<class T> __global__ void pack_int4_rows(
    const T* __restrict__ k, uint8_t* __restrict__ codes,
    float* __restrict__ kscale, int64_t batch_kv, int src_s, int dst_s,
    int row0, int nrows, int D) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t logical = (int64_t)blockIdx.x * (blockDim.x >> 5) + warp;
  const int64_t total = batch_kv * nrows;
  constexpr unsigned FULL = 0xffffffffu;
  if (logical >= total) return;
  const int64_t kvbh = logical / nrows;
  const int row = row0 + (int)(logical - kvbh * nrows);
  const T* krow = k + (kvbh * src_s + row) * D;
  const int half_d = (D + 1) >> 1;
  uint8_t* crow = codes + (kvbh * dst_s + row) * half_d;
  float amax = 0.f;
  bool nonfinite = false;
  for (int d = lane; d < D; d += 32) {
    const float x = ld<T>(krow, d);
    amax = fmaxf(amax, fabsf(x));
    nonfinite |= !isfinite(x);
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_down_sync(FULL, amax, off));
  amax = __shfl_sync(FULL, amax, 0);
  nonfinite = __any_sync(FULL, nonfinite);
  const float scale = amax * kApaInt4InvQmax;
  const float safe = scale > 0.f ? scale : 1.f;
  const float recip = 1.f / safe;
  for (int b = lane; b < half_d; b += 32) {
    const float x0 = ld<T>(krow, 2 * b);
    const float x1 = 2 * b + 1 < D ? ld<T>(krow, 2 * b + 1) : 0.f;
    float c0 = roundf(x0 * recip);
    float c1 = roundf(x1 * recip);
    c0 = c0 < -7.f ? -7.f : (c0 > 7.f ? 7.f : c0);
    c1 = c1 < -7.f ? -7.f : (c1 > 7.f ? 7.f : c1);
    crow[b] = (uint8_t)((((int)c1 + 7) << 4) | ((int)c0 + 7));
  }
  if (lane == 0) kscale[kvbh * dst_s + row] = nonfinite ? __int_as_float(0x7fc00000) : safe;
}
// APA_P1F1_END int4_pack
// APA_SPQ1_BEGIN deterministic_prepare
// Prior art: TurboQuant (Zandieh et al. 2025) normalized rotation; IEEE754
// (2008) explicitly rounded operations. Ours: shape-independent row schedule.
// Norm: separately rounded multiply then add, d=0..D-1, then sqrt_rn.
// Unit: div_rn(k[d], add_rn(norm,1e-12f)). Each rotated output coordinate:
// sequential fmaf_rn(unit[i],R[coordinate,i],acc), i=0..D-1. No GEMM.
// APA-P1-F2 R2 (P1R2-F02), superseding F1's "no code change" note: an Inf
// element gives ss=Inf and norm=Inf, and unit=k/(Inf+1e-12) is 0 or NaN, so
// every rotated coordinate is NaN and pack_tq (NaN > bnd is false) stores code
// 0 with the Inf norm. The bulk score acc*Inf then takes the sign of the
// code-0 dot product, can be -Inf, and the key is skipped with weight 0, a
// finite row where dense SP is NaN. Rule now: a key row with any non-finite
// element stores a quiet NaN norm (as F1's INT4 carry stores a NaN scale), so
// every score of that key is NaN, never -Inf. The shared `norm` that the unit
// vector divides by is not changed, so the rotated output is bit-for-bit as
// before; a row whose elements are all finite stores the same norm bit for bit
// (L55: only rows with a non-finite element change).
// APA-P1-F3 R2' (P1R3-F02), superseding F2's sentence that a finite row whose
// ss overflows keeps its Inf norm: a row of finite elements whose sum of
// squares overflows fp32 (one element near 1e20 at D64 already does) has
// ss=Inf, norm=Inf, unit 0 and code 8 on every coordinate, and the bulk score
// takes the sign of the code-8 dot product times Inf, so it can be -Inf and
// the key is skipped with weight 0 while dense SP reconstructs mixed-sign
// infinities and returns NaN (REV3's r01, confirmed on the GPU). Rule now: the
// stored norm is a quiet NaN for ANY row whose computed norm is non-finite,
// whether from a non-finite element or from overflow; every score of that key
// is then NaN, the loud direction. The shared `norm` the unit vector divides
// by is still not changed (rotated output bit for bit as before), and a row
// whose computed norm is finite stores the same norm bit for bit: sqrt_rn of a
// finite non-negative ss is finite, and ss is non-finite exactly when an
// element is non-finite or the sum overflows (L55: only rows whose computed
// norm is non-finite change; LF1's rows all have finite norms).
// This kernel is the only TQ4 key quantizer of the packed route (census:
// artifacts/apa_p1f2/TQ4_CENSUS.json); pack_tq stores norms[] unchanged.
// Prior art: IEEE 754-2008 NaN/Inf propagation and overflow to Inf; TurboQuant
// (Zandieh et al., 2025) norm/rotation, unchanged; F1's INT4 non-finite
// sentinel (Project-Tensor 2026). Ours: the NaN-norm carry for this quantizer,
// now keyed on the computed norm as well as on the elements.
template<class T> __global__ void prepare(const T* k,const float* r,float* rot,float* norms,
 int H,int S,int D,int start) {
 int row=blockIdx.x,N=S-start,h=(row/N)%H,j=start+row%N,d=threadIdx.x;
 int64_t src=((int64_t)(row/N)*S+j)*D;
 __shared__ float unit[128],norm;
 if(d==0){float ss=0;bool nonfinite=false;
   for(int i=0;i<D;++i){float v=ld<T>(k,src+i);nonfinite|=!isfinite(v);ss=__fadd_rn(ss,__fmul_rn(v,v));}
   norm=__fsqrt_rn(ss);norms[row]=(nonfinite || !isfinite(norm))?__int_as_float(0x7fc00000):norm;}
 __syncthreads();
 if(d<D)unit[d]=__fdiv_rn(ld<T>(k,src+d),__fadd_rn(norm,1e-12f));
 __syncthreads();
 if(d<D){float x=0;for(int i=0;i<D;++i)x=__fmaf_rn(unit[i],r[((int64_t)h*D+d)*D+i],x);rot[(int64_t)row*D+d]=x;}
}
// APA_SPQ1_END deterministic_prepare
// APA_SPQ1_BEGIN shared_decode
// Prior art: FlashInfer (Ye et al. 2025), GQA reuse; NVIDIA warp broadcast
// and register blocking. Ours: decode once, reuse z in G accumulators.
// D128 uses the exact r1 lane FMA and 16/8/4/2/1 tree; D64 sequential FMA.
// Each byte is loaded once (even lane for D128, broadcast to odd partner),
// each coordinate is decoded once, norm is loaded once/key; all G heads reuse.
template<int D,int G,bool TQ> __global__ void bulk_shared(const float* q,const uint8_t* packed,
 const float* scales,const float* cb,float* values,float* cm,int H,int KVH,int S,int NC,int capacity,float scale) {
 __shared__ float query[G*D],table[16],maxima[G*128];
 int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
 int b=bh/KVH,h0=(bh%KVH)*G,start=c*256,count=min(256,S-start),tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int i=tid;i<G*D;i+=128)query[i]=q[(int64_t)(b*H+h0)*D+i];
 if(tid<16)table[tid]=cb[tid];__syncthreads();
 float mx[G];
 #pragma unroll
 for(int g=0;g<G;++g)mx[g]=-INFINITY;
 for(int j=(D==128?warp:tid);j<count;j+=(D==128?4:128)) {
   int64_t key=(int64_t)bh*capacity+start+j;
   float n;
   if constexpr(D==128){n=lane==0?scales[key]:0.f;n=__shfl_sync(0xffffffffu,n,0);}else n=scales[key];
   // PF2-F1 (Project-Tensor, 2026) and IEEE 754-2008: reuse prefill's quiet-NaN sentinel at the decode load.
   n=isfinite(n)?n:__int_as_float(0x7fc00000);
   float acc[G]={};
   if constexpr(D==128){
     #pragma unroll
     for(int t=0;t<D/32;++t){int d=lane+32*t;
       unsigned byte=lane%2==0?packed[key*(D/2)+d/2]:0;
       byte=__shfl_sync(0xffffffffu,byte,lane&~1);
       int ix=(byte>>(4*(d%2)))&15;
       float z=TQ?table[ix]:__fmul_rn(float(ix-7),n);
       #pragma unroll
       for(int g=0;g<G;++g)acc[g]=__fmaf_rn(query[g*D+d],z,acc[g]);
     }
     #pragma unroll
     for(int g=0;g<G;++g){
       #pragma unroll
       for(int off=16;off;off>>=1)acc[g]=__fadd_rn(acc[g],__shfl_down_sync(0xffffffffu,acc[g],off));
     }
   }else{
     #pragma unroll
     for(int pair=0;pair<D/2;++pair){unsigned byte=packed[key*(D/2)+pair];
       #pragma unroll
       for(int half=0;half<2;++half){int d=2*pair+half,ix=(byte>>(4*half))&15;float z=TQ?table[ix]:__fmul_rn(float(ix-7),n);
         #pragma unroll
         for(int g=0;g<G;++g)acc[g]=__fmaf_rn(query[g*D+d],z,acc[g]);
       }
     }
   }
   if(D==64 || lane==0){
     #pragma unroll
     for(int g=0;g<G;++g){float x=acc[g];if(TQ)x=__fmul_rn(x,n);x=__fmul_rn(x,scale);
       values[(int64_t)(b*H+h0+g)*S+start+j]=x;mx[g]=fmaxf(mx[g],x);}
   }
 }
 #pragma unroll
 for(int g=0;g<G;++g){maxima[g*128+tid]=mx[g];}
 __syncthreads();
 for(int off=64;off;off>>=1){if(tid<off){
   #pragma unroll
   for(int g=0;g<G;++g)maxima[g*128+tid]=fmaxf(maxima[g*128+tid],maxima[g*128+tid+off]);
 }__syncthreads();}
 if(tid<G)cm[(int64_t)(b*H+h0+tid)*NC+c]=maxima[tid*128];
}
// APA_SPQ1_END shared_decode
template<int CAP,bool W> __global__ void bulk(const float* q,const uint8_t* packed,
 const float* scales,const float* cb,float* values,float* cm,int H,int KVH,int S,int D,int NC,int capacity,float scale,bool tq) {
  constexpr int TILE=W?16:64;
  __shared__ uint8_t codes[TILE*CAP/2];__shared__ float norms[TILE],table[16],red[128];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int G=H/KVH,b=bh/KVH,h0=(bh%KVH)*G,start=c*256,count=min(256,S-start),tid=threadIdx.x,lane=tid%32;
  if(tid<16)table[tid]=cb[tid];__syncthreads();
  for(int t=0;t<count;t+=TILE){
    int n=min(TILE,count-t);
    // Exactly D/2 global code bytes + one fp32 scale per KV row, shared by G.
    for(int i=tid;i<n*(D/2);i+=128)codes[i]=packed[((int64_t)bh*capacity+start+t)*(D/2)+i];
    if(tid<n)norms[tid]=scales[(int64_t)bh*capacity+start+t+tid];__syncthreads();
    for(int g=0;g<G;++g){int row=b*H+h0+g;const float* Q=q+(int64_t)row*D;
      if constexpr(W){
        for(int j=tid/32;j<n;j+=4){float x=0;
          for(int d=lane;d<D;d+=32){int ix=(codes[j*(D/2)+d/2]>>(4*(d%2)))&15;
            float z=tq?table[ix]:__fmul_rn(float(ix-7),norms[j]);x=__fmaf_rn(Q[d],z,x);}
          #pragma unroll
          for(int off=16;off;off>>=1)x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,off));
          if(lane==0){if(tq)x=__fmul_rn(x,norms[j]);values[(int64_t)row*S+start+t+j]=__fmul_rn(x,scale);}
        }
      }else if(tid<n){float x=0;
        for(int d=0;d<D;++d){int ix=(codes[tid*(D/2)+d/2]>>(4*(d%2)))&15;
          float z=tq?table[ix]:__fmul_rn(float(ix-7),norms[tid]);x=__fmaf_rn(Q[d],z,x);}
        if(tq)x=__fmul_rn(x,norms[tid]);values[(int64_t)row*S+start+t+tid]=__fmul_rn(x,scale);
      }
    }__syncthreads();
  }
  for(int g=0;g<G;++g){int row=b*H+h0+g;float mx=-INFINITY;
    for(int j=tid;j<count;j+=128)mx=fmaxf(mx,values[(int64_t)row*S+start+j]);red[tid]=mx;__syncthreads();
    for(int off=64;off;off>>=1){if(tid<off)red[tid]=fmaxf(red[tid],red[tid+off]);__syncthreads();}
    if(tid==0)cm[(int64_t)row*NC+c]=red[0];__syncthreads();
  }
}
} // namespace spq1
// APA_SPQ1_END packed_bulk
