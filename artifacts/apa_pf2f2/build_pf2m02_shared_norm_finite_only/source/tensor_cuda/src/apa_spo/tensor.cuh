// APA_SPO5_BEGIN tensor_dots
// Prior art: FlashInfer (Ye et al. 2024/2025), GQA decode heads as MMA rows,
// https://flashinfer.ai/2024/02/02/introduce-flashinfer.html (taken packing).
// NVIDIA PTX ISA m16n8k16 row.col FP16/BF16 -> FP32, taken fragment layout:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-f16
// Fasi/Higham/Mikaitis/Pranesh (2021), Numerical Behavior of NVIDIA Tensor
// Cores: accumulation differs from ordered FP32; explicitly Tier-1 gated.
// BLASST (Yuan et al. 2025) running maximum, SP1 (2026) row prefix,
// Hillis/Steele (1986) scan, Flash-Decoding (2023) split merge and
// Milakov/Gimelshein (2018) stable softmax are inherited techniques.
// Ours: APA chunk selection wiring around those fragments. No prior art
// known to me for this exact assembly; no new general algorithm claimed.
namespace spo5 {
constexpr int C=256,NT=128;
// Pack two existing 16-bit operands without a numerical conversion. Padded
// heads/dimensions are literal +0. IEEE binary16/bfloat16 representation.
template<class T> __device__ unsigned pair16(const T* p,int d,int D,bool valid) {
  unsigned lo=valid && d<D ? reinterpret_cast<const unsigned short*>(p)[d] : 0;
  unsigned hi=valid && d+1<D ? reinterpret_cast<const unsigned short*>(p)[d+1] : 0;
  return lo|(hi<<16);
}
template<class T,int MG> __device__ void mma8(const T* q,const T* k,int D,int nh,int n,float (&x)[MG/16][4],const int* use=nullptr) {
  int lane=threadIdx.x%32,r=lane/4,d=(lane%4)*2;
  #pragma unroll
  for(int h=0;h<MG/16;++h)for(int i=0;i<4;++i)x[h][i]=0.f;
  for(int base=0;base<D;base+=16) {
    // FlashInfer grouped heads: load B once and reuse across M=16 tiles.
    unsigned b0=pair16(k+(int64_t)r*D,base+d,D,r<n && (!use || use[r]));
    unsigned b1=pair16(k+(int64_t)r*D,base+d+8,D,r<n && (!use || use[r]));
    #pragma unroll
    for(int ht=0;ht<MG/16;++ht) {
    int hr=r+16*ht;
    unsigned a0=pair16(q+(int64_t)hr*D,base+d,D,hr<nh);
    unsigned a1=pair16(q+(int64_t)(hr+8)*D,base+d,D,hr+8<nh);
    unsigned a2=pair16(q+(int64_t)hr*D,base+d+8,D,hr<nh);
    unsigned a3=pair16(q+(int64_t)(hr+8)*D,base+d+8,D,hr+8<nh);
    if constexpr(std::is_same<T,__half>::value) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(x[ht][0]),"+f"(x[ht][1]),"+f"(x[ht][2]),"+f"(x[ht][3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    } else {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(x[ht][0]),"+f"(x[ht][1]),"+f"(x[ht][2]),"+f"(x[ht][3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    }
  }
}
template<class T,int MG> __global__ void bulk_kernel(const T* q,const T* kq,float* bulk,float* cm,
    int H,int KVH,int S,int D,int NC,float scale) {
  int G=H/KVH,tiles=(G+MG-1)/MG,bh=blockIdx.x/tiles,head=(blockIdx.x%tiles)*MG;
  int row=(bh/KVH)*H+(bh%KVH)*G+head,nh=min(MG,G-head);
  int c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int start=c*C,count=min(C,S-start),tid=threadIdx.x,lane=tid%32,warp=tid/32;
  __shared__ float staged[MG*C],red[4];
  for(int j=warp*8;j<count;j+=32) {
    float x[MG/16][4];mma8<T,MG>(q+(int64_t)row*D,kq+((int64_t)bh*S+start+j)*D,D,nh,min(8,count-j),x);
    #pragma unroll
    for(int ht=0;ht<MG/16;++ht)for(int i=0;i<4;++i) {
      int h=16*ht+lane/4+(i/2)*8,key=j+2*(lane%4)+(i%2);
      // DeMillo et al. 1978 mutation hooks; compiled zero in production.
      if constexpr(APA_SPO5_MUTANT==3)if(h<nh)h=nh-1-h;
      if(h<nh && key<count)staged[h*C+key]=x[ht][i]*scale;
    }
  }
  __syncthreads();
  if constexpr(APA_SPO5_MUTANT==4)if(nh<16)for(int j=tid;j<count;j+=NT)staged[j]=0.f;
  __syncthreads();
  for(int h=0;h<nh;++h) {
    float mx=-INFINITY;
    for(int j=tid;j<count;j+=NT){float x=staged[h*C+j];bulk[(int64_t)(row+h)*S+start+j]=x;mx=fmaxf(mx,x);}
    mx=spo4::max128(mx,red);if(tid==0)cm[(int64_t)(row+h)*NC+c]=mx;
  }
}
template<class T,int CAP,bool DIAG,int MG> __global__ void select_kernel(
    const T* q,const T* k,const T* v,const float* bulk,const float* cm,
    uint8_t* mask,float* prefix_dump,float* pm,float* pl,float* pa,
    int H,int KVH,int S,int D,int VD,int NC,int PK,float scale,float delta) {
  int G=H/KVH,tiles=(G+MG-1)/MG,bh=blockIdx.x/tiles,head=(blockIdx.x%tiles)*MG;
  int row=(bh/KVH)*H+(bh%KVH)*G+head,nh=min(MG,G-head);
  int c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int start=c*C,count=min(C,S-start),tid=threadIdx.x,lane=tid%32,warp=tid/32;
  __shared__ float scores[MG*C],wm[4],red[4],sums[NT],pv[4*CAP];
  __shared__ int any[C];
  __shared__ uint8_t use[MG==16?16*C:1];
  for(int j=tid;j<C;j+=NT)any[j]=0;
  __syncthreads();
  for(int h=0;h<nh;++h) {
    int64_t base=(int64_t)(row+h)*S+start;
    float carry=-INFINITY;int first=(start/PK)*PK/C;
    for(int p=first;p<c-(APA_SPO5_MUTANT==2?1:0);++p)carry=fmaxf(carry,cm[(int64_t)(row+h)*NC+p]);
    for(int t=0;t<C;t+=NT) {
      int j=t+tid;float x=j<count?bulk[base+j]:-INFINITY,p=x;
      #pragma unroll
      for(int o=1;o<32;o<<=1){float y=__shfl_up_sync(0xffffffffu,p,o);if(lane>=o)p=fmaxf(p,y);}
      if(lane==31)wm[warp]=p;__syncthreads();
      p=fmaxf(p,carry);for(int w=0;w<warp;++w)p=fmaxf(p,wm[w]);
      bool refine=j<count && (APA_SPO5_MUTANT==6?x>p-delta:x>=p-delta);
      if constexpr(MG==16)use[h*C+j]=refine;
      scores[h*C+j]=x;if(refine)any[j]=1;
      if(j<count){mask[base+j]=refine;if constexpr(DIAG)prefix_dump[base+j]=p;}
      for(int w=0;w<4;++w)carry=fmaxf(carry,wm[w]);__syncthreads();
    }
  }
  for(int j=warp*8;j<count;j+=32) {
    float x[MG/16][4];mma8<T,MG>(q+(int64_t)row*D,k+((int64_t)bh*S+start+j)*D,D,nh,min(8,count-j),x,any+j);
    #pragma unroll
    for(int ht=0;ht<MG/16;++ht)for(int i=0;i<4;++i) {
      int h=16*ht+lane/4+(i/2)*8,key=j+2*(lane%4)+(i%2);
      if constexpr(APA_SPO5_MUTANT==3)if(h<nh)h=nh-1-h;
      if(h<nh && key<count && (MG==16?use[h*C+key]:mask[(int64_t)(row+h)*S+start+key]))scores[h*C+key]=x[ht][i]*scale;
    }
  }
  __syncthreads();
  // Inherited SPO4 warp-strided PV (FP32 products/sums) and split-K merge.
  // Four independent dimension chains (NVIDIA ILP); no PV MMA rounding.
  const T* V=v+((int64_t)bh*S+start)*VD;
  for(int h=0;h<nh;++h) {
    int64_t pr=(int64_t)(row+h)*NC+c;
    float* s=scores+h*C;float mx=-INFINITY;
    for(int j=tid;j<count;j+=NT)mx=fmaxf(mx,s[j]);
    mx=spo4::max128(mx,red);float sum=0;
    for(int j=tid;j<count;j+=NT){float w=__expf(s[j]-mx);s[j]=w;sum+=w;}
    sums[tid]=sum;__syncthreads();
    for(int o=64;o;o>>=1){if(tid<o)sums[tid]+=sums[tid+o];__syncthreads();}
    if(tid==0){pm[pr]=mx;pl[pr]=sums[0];}
    float acc[CAP/32];
    #pragma unroll
    for(int i=0;i<CAP/32;++i)acc[i]=0;
    for(int j=warp;j<count;j+=4) {
      float w=s[j];
      #pragma unroll
      for(int i=0;i<CAP/32;++i)if(lane+32*i<VD)acc[i]+=__fmul_rn(w,ld<T>(V,(int64_t)j*VD+lane+32*i));
    }
    #pragma unroll
    for(int i=0;i<CAP/32;++i)if(lane+32*i<VD)pv[warp*CAP+lane+32*i]=acc[i];
    __syncthreads();
    for(int d=tid;d<VD;d+=NT)pa[pr*VD+d]=(pv[d]+pv[CAP+d])+(pv[2*CAP+d]+pv[3*CAP+d]);
    __syncthreads();
  }
}
} // namespace spo5
// APA_SPO5_END tensor_dots
