// PT-TF32-2. Prior art: NVIDIA Ampere TF32 (2020), cuBLASLt (CUDA 12.6,
// 2024), taken column-major transpose identity, heuristics, workspace,
// descriptors and implementation flags. Local APAMQ-SB2 uses Lt too.
// https://docs.nvidia.com/cuda/archive/12.6.2/cublas/index.html
// Ours: bounded per-thread/device plan cache and explicit HMMA/TF32 filter.
// Capability flags are dispatch evidence; PT-TF32-4 reports speed diagnostically.
#include "tc/tf32.h"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#include <limits>

namespace tc {
namespace {
constexpr size_t workspace_bytes=32ull*1024*1024;
constexpr uint64_t required_flags=CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA |
    CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 |
    CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F;
thread_local std::tuple<int,uint64_t,size_t> last_info{-1,0,0};
// Prior art: existing Lt plan cache (PT-TF32-2, 2026), taken ownership.
// Ours: copy diagnostic maps only on explicit inspection, never per GEMM.
// This pointer is cleared before any plan eviction in the next matmul.
thread_local const std::map<std::string,int64_t>* last_dispatch=nullptr;
thread_local bool last_output_stream_ordered=false;
// Prior art: NVIDIA WMMA TF32 m16n16k8 (CUDA 11, 2020), taken.
// Ours: tail-safe row-major/broadcast fallback where Lt's alignment-dependent
// algorithm availability is not guaranteed. Model shapes always use Lt.
__global__ void ragged_gemm(const float* a,const float* b,float* out,
    int64_t M,int64_t N,int64_t K,int64_t batch,int64_t sb,float alpha,bool trans_b) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  __shared__ __align__(32) float as[16*8],bs[8*16],cs[16*16];
  int lane=threadIdx.x;int64_t row=int64_t(blockIdx.y)*16,col=int64_t(blockIdx.x)*16;
  for(int64_t bi=blockIdx.z;bi<batch;bi+=gridDim.z) {
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,16,16,8,float> cf;
    wmma::fill_fragment(cf,0.f);
    for(int64_t lo=0;lo<K;lo+=8) {
      for(int z=lane;z<128;z+=32) {
        int64_t i=row+z/8,k=lo+z%8;
        as[z]=(i<M && k<K)?a[bi*M*K+i*K+k]:0.f;
        k=lo+z/16;int64_t j=col+z%16;
        bs[z]=(k<K && j<N)?b[bi*sb+(trans_b?j*K+k:k*N+j)]:0.f;
      }
      __syncwarp();
      wmma::load_matrix_sync(af,as,8);wmma::load_matrix_sync(bf,bs,16);
      #pragma unroll
      for(int i=0;i<af.num_elements;++i)af.x[i]=wmma::__float_to_tf32(af.x[i]);
      #pragma unroll
      for(int i=0;i<bf.num_elements;++i)bf.x[i]=wmma::__float_to_tf32(bf.x[i]);
      wmma::mma_sync(cf,af,bf,cf);__syncwarp();
    }
    wmma::store_matrix_sync(cs,cf,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32) {
      int64_t i=row+z/16,j=col+z%16;
      if(i<M && j<N)out[bi*M*N+i*N+j]=alpha*cs[z];
    }
    __syncwarp();
  }
#else
  asm volatile("trap;");
#endif
}
void check(cublasStatus_t s,const char* where) {
  if(s!=CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("PT-TF32-2 ")+where+": cuBLAS status "+std::to_string(int(s)));
}
void cuda_ok(cudaError_t s) {
  if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
// Descriptors own only host resources. Destruction is also safe after failed
// construction/search; a shape failure must not leave stale cached plans.
struct Plan {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  cublasLtMatmulPreference_t pref=nullptr;
  cublasLtMatmulHeuristicResult_t result{};
  int id=-1; uint64_t flags=0;
  std::map<std::string,int64_t> info;
  ~Plan() {
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    if(a)cublasLtMatrixLayoutDestroy(a);
    if(b)cublasLtMatrixLayoutDestroy(b);
    if(c)cublasLtMatrixLayoutDestroy(c);
    if(op)cublasLtMatmulDescDestroy(op);
  }
};
using Key=std::array<int64_t,10>;
struct Context {
  cublasLtHandle_t lt=nullptr;
  void* workspace=nullptr;
  int device;
  std::map<Key,std::unique_ptr<Plan>> plans;
  explicit Context(int dev):device(dev) {
    check(cublasLtCreate(&lt),"LtCreate");
    auto status=cudaMalloc(&workspace,workspace_bytes);
    if(status!=cudaSuccess) {cublasLtDestroy(lt);lt=nullptr;cuda_ok(status);}
  }
  ~Context() {
    int old=-1;cudaGetDevice(&old);cudaSetDevice(device);
    if(workspace)cudaFree(workspace);
    plans.clear();if(lt)cublasLtDestroy(lt);
    if(old>=0 && old!=device)cudaSetDevice(old);
  }
};
thread_local std::map<int,std::unique_ptr<Context>> contexts;
uint32_t alignment(const void* p,int64_t stride,int64_t batch) {
  uintptr_t bits=reinterpret_cast<uintptr_t>(p);
  if(batch>1 && stride)bits|=uintptr_t(stride)*sizeof(float);
  uint32_t value=1;
  while(value<256 && bits%(value*2)==0)value*=2;
  return value;
}
void layout(cublasLtMatrixLayout_t* dest,int64_t rows,int64_t cols,int64_t ld,
            int batch,int64_t stride) {
  check(cublasLtMatrixLayoutCreate(dest,CUDA_R_32F,rows,cols,ld),"layout");
  check(cublasLtMatrixLayoutSetAttribute(*dest,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)),"batch count");
  check(cublasLtMatrixLayoutSetAttribute(*dest,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride,sizeof(stride)),"batch stride");
}
// PT-TF32-3. Prior art: NVIDIA cuBLASLt host planning/capability API
// (2024), taken. Ours: the self-check and real dispatch share this exact planner;
// 16-byte minimum for aligned model paths and explicit transpose readback.
std::unique_ptr<Plan> make_plan(const Key& key,cublasLtHandle_t lt) {
  auto M=key[0],N=key[1],K=key[2],batch=key[3],sb=key[4];
  bool trans_b=bool(key[5]);
  uint32_t aa=uint32_t(key[7]),ab=uint32_t(key[8]),ac=uint32_t(key[9]);
  if(aa<16 || ab<16 || ac<16)throw std::runtime_error("PT-TF32-3: aligned Lt GEMM requires 16-byte pointers/strides");
    auto plan=std::make_unique<Plan>();
    check(cublasLtMatmulDescCreate(&plan->op,CUBLAS_COMPUTE_32F_FAST_TF32,CUDA_R_32F),"matmul descriptor");
    cublasOperation_t op=trans_b?CUBLAS_OP_T:CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(plan->op,CUBLASLT_MATMUL_DESC_TRANSA,&op,sizeof(op)),"transpose");
    cublasOperation_t op_b=CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(plan->op,CUBLASLT_MATMUL_DESC_TRANSB,&op_b,sizeof(op_b)),"transpose B");
    layout(&plan->a,trans_b?K:N,trans_b?N:K,trans_b?K:N,int(batch),sb);
    layout(&plan->b,K,M,K,int(batch),M*K);
    layout(&plan->c,N,M,N,int(batch),M*N);
    // NVIDIA cuBLASLt (2024), taken descriptor readback. No guessed mode.
    int32_t compute=0,ta=-1,tb=-1;size_t got=0;
    check(cublasLtMatmulDescGetAttribute(plan->op,CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,&compute,sizeof(compute),&got),"compute readback");
    check(cublasLtMatmulDescGetAttribute(plan->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta),&got),"transpose A readback");
    check(cublasLtMatmulDescGetAttribute(plan->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb),&got),"transpose B readback");
    if(compute!=CUBLAS_COMPUTE_32F_FAST_TF32 || ta!=op || tb!=CUBLAS_OP_N)
      throw std::runtime_error("PT-TF32-3: descriptor compute/transpose mismatch");
    plan->info={{"algorithm_id",-1},{"compute_type",compute},{"fast_tf32",1},
        {"transa",ta},{"transb",tb},{"lda",trans_b?K:N},{"ldb",K},{"ldc",N},
        {"alignment_a",aa},{"alignment_b",ab},{"alignment_c",ac},
        {"M",M},{"N",N},{"K",K},{"batch",batch},{"stride_a",sb},
        {"mathmode_impl",-1},{"mathmode_query_status",-1},
        {"mathmode_query_supported",0},{"mathmode_query_attempted",0},
        {"numerical_flags_attribute",int(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)},
        {"numerical_flags_query_status",-1},{"numerical_flags_bytes",0},{"heuristic_count",0},
        {"numerical_flags",0},{"workspace_bytes",0},{"device_selected",0}};
    if(!lt)return plan; // Host descriptor-only check; no CUDA device operation.
    check(cublasLtMatmulPreferenceCreate(&plan->pref),"preference");
    check(cublasLtMatmulPreferenceSetAttribute(plan->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)),"workspace preference");
    for(auto attr:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                   CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
      uint32_t value=attr==CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES?aa:
          attr==CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES?ab:ac;
      check(cublasLtMatmulPreferenceSetAttribute(plan->pref,attr,&value,sizeof(value)),"alignment preference");
    }
    uint64_t mask=required_flags;
    check(cublasLtMatmulPreferenceSetAttribute(plan->pref,CUBLASLT_MATMUL_PREF_IMPL_MASK,&mask,sizeof(mask)),"HMMA preference");
    std::array<cublasLtMatmulHeuristicResult_t,32> candidates{};int count=0;
    check(cublasLtMatmulAlgoGetHeuristic(lt,plan->op,plan->a,plan->b,plan->c,plan->c,
          plan->pref,int(candidates.size()),candidates.data(),&count),"heuristic");
    plan->info["heuristic_count"]=count;
    bool found=false;
    for(int i=0;i<count;++i) {
      auto candidate=candidates[i];uint64_t flags=0;size_t written=0;
      if(candidate.state!=CUBLAS_STATUS_SUCCESS || candidate.workspaceSize>workspace_bytes)continue;
      auto flags_status=cublasLtMatmulAlgoCapGetAttribute(&candidate.algo,
          CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written);
      check(flags_status,"implementation flags");
      if(written!=sizeof(flags) || (flags & required_flags)!=required_flags)continue;
      // PT-TF32-4. Prior art: NVIDIA cuBLASLt CUDA 12.6 (2024), taken
      // supported capability API, not a new algorithm-selection rule.
      // Attribute 8 is absent in this API; casting it produced status 7
      // (CUBLAS_STATUS_INVALID_VALUE), a query usage error, not CUDA-core
      // execution evidence. Query named uint64_t NUMERICAL_IMPL_FLAGS only.
      // Keep legacy fields -1 (not queried), explicitly mark unsupported,
      // and receipt the real replacement query status and returned byte count.
      plan->info["numerical_flags_query_status"]=int(flags_status);
      plan->info["numerical_flags_bytes"]=int64_t(written);
      plan->result=candidate;plan->flags=flags;found=true;break;
    }
    if(!found)throw std::runtime_error("PT-TF32-2: no TF32 HMMA algorithm for GEMM geometry/alignment; no silent SGEMM fallback");
    size_t written=0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&plan->result.algo,CUBLASLT_ALGO_CONFIG_ID,
          &plan->id,sizeof(plan->id),&written),"algorithm ID");
    plan->info["algorithm_id"]=plan->id;
    plan->info["numerical_flags"]=int64_t(plan->flags);
    plan->info["workspace_bytes"]=int64_t(plan->result.workspaceSize);
    plan->info["device_selected"]=1;
    return plan;
}
} // namespace

std::tuple<int,uint64_t,size_t> get_tf32_gemm_info() {return last_info;}
std::map<std::string,int64_t> get_tf32_gemm_dispatch() {
  if(!last_dispatch)return {};
  auto info=*last_dispatch;info["output_stream_ordered"]=last_output_stream_ordered;
  return info;
}
// PT-TF32-3. Prior art: NVIDIA stream-ordered cudaMallocAsync/cudaFreeAsync
// (CUDA 11.2, 2021), and this engine's Storage::pooled lifecycle (2026), taken.
// Ours: explicitly requested TF32 GEMM outputs enter the existing default-
// stream pool even when the caller has not enabled global transient pooling.
// A cold synchronous cudaMalloc inside every measured matmul can dominate
// a Tensor Core kernel. Defaults/other dtype allocators are unchanged.
NDArray tf32_gemm_output(const Shape& shape,Device device) {
  if(!device.is_cuda())throw std::runtime_error("PT-TF32-3: TF32 output requires CUDA");
  NDArray out;
  out.shape=shape;out.dtype=DType::Float32;out.device=device;
  // Zero-byte Storage creates no allocation; retain its existing ownership
  // and destructor instead of a separate cache with unbounded live outputs.
  out.storage=std::make_shared<Storage>(0,device);
  out.storage->nbytes=size_t(out.numel())*sizeof(float);
  out.storage->pooled=true;
  cuda_ok(cudaMallocAsync(&out.storage->ptr,out.storage->nbytes,nullptr));
  return out;
}
std::map<std::string,int64_t> tf32_gemm_self_check(int64_t M,int64_t N,int64_t K,
                                                bool trans_b,bool select_on_device) {
  if(M<=0 || N<=0 || K<=0 || M>INT32_MAX || N>INT32_MAX || K>INT32_MAX ||
      M%16 || N%16 || K%8)throw std::runtime_error("PT-TF32-3: self-check requires positive aligned INT32 dimensions");
  const char* override_value=std::getenv("NVIDIA_TF32_OVERRIDE");
  if(select_on_device && override_value && std::strcmp(override_value,"0")==0)
    throw std::runtime_error("PT-TF32-3: NVIDIA_TF32_OVERRIDE=0");
  Key key{M,N,K,1,K*N,int64_t(trans_b),-1,16,16,16};
  if(!select_on_device)return make_plan(key,nullptr)->info;
  int dev=-1;cuda_ok(cudaGetDevice(&dev));key[6]=dev;
  auto& ctx=contexts[dev];if(!ctx)ctx=std::make_unique<Context>(dev);
  return make_plan(key,ctx->lt)->info;
}

void matmul_tf32(const NDArray& a,const NDArray& b,NDArray& out,float alpha,bool trans_b) {
  last_info={-1,0,0};last_dispatch=nullptr;
  const char* override_value=std::getenv("NVIDIA_TF32_OVERRIDE");
  if(override_value && std::strcmp(override_value,"0")==0)
    throw std::runtime_error("PT-TF32-2: NVIDIA_TF32_OVERRIDE=0 disables the requested TF32 path");
  int dev=-1;cuda_ok(cudaGetDevice(&dev));
  if(!a.device.is_cuda() || !b.device.is_cuda() || dev!=a.device.index || dev!=b.device.index)
    throw std::runtime_error("PT-TF32-2: GEMM requires the current CUDA device");
  int64_t M=a.shape[a.ndim()-2],K=a.shape.back(),N=out.shape.back();
  if(M<=0 || N<=0 || K<=0)throw std::runtime_error("PT-TF32-2: positive GEMM dimensions required");
  int64_t batch=a.numel()/(M*K),sb=b.ndim()==a.ndim()?K*N:0;
  if(batch>std::numeric_limits<int>::max())throw std::runtime_error("PT-TF32-2: GEMM batch exceeds INT32");
  if(M%16 || N%16 || K%8) {
    int major=0;cuda_ok(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));
    if(major<8)throw std::runtime_error("PT-TF32-2: ragged TF32 WMMA requires SM80+");
    if((M+15)/16>65535 || (N+15)/16>std::numeric_limits<int>::max())
      throw std::runtime_error("PT-TF32-2: ragged GEMM exceeds grid limits");
    ragged_gemm<<<dim3((N+15)/16,(M+15)/16,batch<65535?batch:65535),32>>>(
        (const float*)a.data_ptr(),(const float*)b.data_ptr(),(float*)out.data_ptr(),M,N,K,batch,sb,alpha,trans_b);
    cuda_ok(cudaGetLastError());last_info={-2,required_flags,0};return;
  }
  uint32_t aa=alignment(b.data_ptr(),sb,batch),ab=alignment(a.data_ptr(),M*K,batch),
           ac=alignment(out.data_ptr(),M*N,batch);
  Key key{M,N,K,batch,sb,int64_t(trans_b),dev,aa,ab,ac};
  auto& ctx=contexts[dev];if(!ctx)ctx=std::make_unique<Context>(dev);
  auto entry=ctx->plans.find(key);
  if(entry==ctx->plans.end()) {
    auto plan=make_plan(key,ctx->lt);
    // Bounded host metadata; workspace is shared on the engine's default
    // stream. Eviction does not free any buffer referenced by an in-flight GEMM.
    if(ctx->plans.size()>=128)ctx->plans.clear();
    entry=ctx->plans.emplace(key,std::move(plan)).first;
  }
  auto& plan=*entry->second;float beta=0.f;
  check(cublasLtMatmul(ctx->lt,plan.op,&alpha,b.data_ptr(),plan.a,a.data_ptr(),plan.b,
        &beta,out.data_ptr(),plan.c,out.data_ptr(),plan.c,&plan.result.algo,
        ctx->workspace,workspace_bytes,nullptr),"LtMatmul");
  last_info={plan.id,plan.flags,plan.result.workspaceSize};
  last_dispatch=&plan.info;
  last_output_stream_ordered=out.storage->pooled;
}
} // namespace tc
