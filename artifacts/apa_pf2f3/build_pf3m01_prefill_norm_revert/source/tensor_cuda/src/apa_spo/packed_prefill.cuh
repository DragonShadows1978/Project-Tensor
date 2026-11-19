// APA_P1PF2_BEGIN packed_prefill
// APA-P1-PF2: prefill (L >= 1, served for L > 1) on packed SP keys, route (a-S)
// of the P1-PF design (artifacts/apa_p1pf/DESIGN.md section 2.1, in
// tensor_fork_apa_wt_p1pf). Lead rulings 2026-11-16: R-E is replaced by this
// route; carry = the whole-row running maximum (the dense prefill rule,
// kernels.cu apa_selective_sp_kernel); the pinned packed margin table applies
// unchanged; no key-shaped transient.
//
// Structure: apa_selective_sp_kernel (kernels.cu, the dense SP prefill kernel),
// one warp per query row, keys visited in ascending order. One substitution:
// the bulk score of key j is taken from the store's 4-bit codes in rotated
// space, q . kq_j = n_j * (qrot . c_j), in the packed L = 1 decode's per-pair
// arithmetic (packed.cuh spq1::bulk / spq1::bulk_shared), with the query row
// rotated in-kernel once in spq1::rotate's order. Everything after the bulk
// score (running maximum, refine compare, exact dot from k, online softmax,
// diagnostics mask, sinks, store) is the dense kernel's source, copied.
//
// Bulk order (must equal the decode oracle's dispatch, launcher.cuh
// packed_dispatch): lanes (lane chain over d = lane + 32 t, then the
// 16/8/4/2/1 fadd tree) unless D == 64 and (G in {4, 8} or max(D, VD) <= 64),
// in which case one sequential fused chain over d = 0..63 (SEQ). The SEQ form
// computes 32 keys at a time (lane = key) and then walks them in order.
//
// Prior art: TurboQuant (Zandieh et al., 2025, arXiv 2504.19874): rotated
// Lloyd-Max 4-bit codes, the stored form, taken. Asymmetric distance
// computation in product quantization (Jegou, Douze and Schmid, 2011): scoring
// a full-precision query against codes without reconstructing, taken as the
// idea (here the query is rotated, not a per-subspace table). BLASST (Yuan et
// al., 2025, arXiv 2512.12087): the running-maximum refine threshold, taken
// unchanged. Online softmax (Milakov and Gimelshein, 2018, arXiv 1805.02867)
// and FlashAttention (Dao et al., 2022): the streaming softmax of the dense
// kernel, inherited unchanged. Ours: putting the packed decode's bulk
// arithmetic inside the dense prefill structure so prefill decisions equal the
// decode's bit for bit; the 32-keys-per-warp form for the sequential D64 order.
// Unverified - lead to check: "attention scores from quantized keys without
// dequantization rotated query".
namespace spq_pf {
// APA_PF2F1_INDEX_BEGIN
// Prior art: ISO C++17 integer promotion; CERT INT32-C (unverified - lead
// to check). Ordinary widening before addition, also callable by host tests.
__host__ __device__ inline int64_t diagnostic_tail_index(int64_t base, int step) {
  return base + step;
}
// APA_PF2F1_INDEX_END
template<class T,int CAP,bool DIAG,bool SEQ>
__global__ void prefill(
    const T* q, const T* k, const T* v, const T* sinks,
    const uint8_t* codes, const float* norms, const float* rot, const float* cb,
    T* out, uint8_t* selected, float* bulk_dump, int H, int L, int S, int D, int VD,
    int KVH, int group, int capacity, float scale, float delta, bool causal) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const int i = row % L, bh = row / L, h = bh % H, b = bh / H;
  const int64_t kvbh = (int64_t)b * KVH + h / group;
  const int smax = causal ? S - L + i + 1 : S;
  constexpr int N = (CAP + 31) / 32;
  constexpr unsigned FULL = 0xffffffffu;
  __shared__ float qrot[128], table[16];
  // In-kernel query rotation, spq1::rotate's order: coordinate d is one
  // sequential fused chain over i2 = 0..D-1 of q[i2] * R[kv, d, i2].
  for (int d = lane; d < D; d += 32) {
    float x = 0.f;
    for (int i2 = 0; i2 < D; ++i2)
      x = __fmaf_rn(ld<T>(q, (int64_t)row * D + i2), rot[((int64_t)(h / group) * D + d) * D + i2], x);
    qrot[d] = x;
  }
  if (lane < 16) table[lane] = cb[lane];
  __syncwarp();
  float qd[4];
  #pragma unroll
  for (int t = 0; t < 4; ++t) qd[t] = lane + 32 * t < D ? qrot[lane + 32 * t] : 0.f;
  const uint8_t* kc = codes + kvbh * (int64_t)capacity * (D / 2);
  const float* kn = norms + kvbh * (int64_t)capacity;
  // From here to the end of the key loop: apa_selective_sp_kernel's source,
  // with only the bulk dot replaced.
  float qr[N], acc[N];
  #pragma unroll
  for (int t = 0; t < N; ++t) {
    const int d = lane + 32 * t;
    qr[t] = d < D ? ld<T>(q, (int64_t)row * D + d) : 0.f;
    acc[t] = 0.f;
  }
  float bulk_max = -INFINITY, m = -INFINITY, denom = 0.f;
  float block = 0.f;
  for (int j = 0; j < smax; ++j) {
    const int64_t koff = (kvbh * S + j) * D;
    float bulk;
    if constexpr (SEQ) {
      // Sequential order (spq1::bulk_shared<64> / spq1::bulk<64,false>):
      // lane l scores key (j & ~31) + l, then key j's score is read from
      // lane j & 31.
      if ((j & 31) == 0) {
        const int jk = j + lane;
        float x = 0.f;
        if (jk < smax) {
          const uint8_t* c = kc + (int64_t)jk * (D / 2);
          for (int d = 0; d < D; ++d) {
            const int ix = (c[d / 2] >> (4 * (d % 2))) & 15;
            x = __fmaf_rn(qrot[d], table[ix], x);
          }
          // Prior art: P1-F3 (Project-Tensor 2026), IEEE 754-2008 NaN
          // sentinel. Reuse at the accessor so raw Inf norms propagate too.
          const float norm = kn[jk];
          x = __fmul_rn(x, norm);
          x = __fmul_rn(x, scale);
        }
        block = x;
      }
      bulk = __shfl_sync(FULL, block, j & 31);
    } else {
      // Lane order (spq1::bulk<CAP,true> / spq1::bulk_shared<128>).
      float x = 0.f;
      const uint8_t* c = kc + (int64_t)j * (D / 2);
      #pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int d = lane + 32 * t;
        if (d < D) {
          const int ix = (c[d / 2] >> (4 * (d % 2))) & 15;
          x = __fmaf_rn(qd[t], table[ix], x);
        }
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        x = __fadd_rn(x, __shfl_down_sync(FULL, x, off));
      // Same P1-F3 non-finite norm sentinel as the sequential accessor above.
      const float norm = kn[j];
      bulk = __fmul_rn(__fmul_rn(__shfl_sync(FULL, x, 0), norm), scale);
    }
#if TC_APA_P1_TEST_HOOKS
    if (bulk_dump && lane == 0) bulk_dump[(int64_t)row * S + j] = bulk;
#endif
    bulk_max = fmaxf(bulk_max, bulk);
    const bool refine = bulk >= bulk_max - delta;
    float score = bulk;
    if (refine) {
      float ex = 0.f;
      #pragma unroll
      for (int t = 0; t < N; ++t) {
        const int d = lane + 32 * t;
        if (d < D) ex += qr[t] * ld<T>(k, koff + d);
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        ex += __shfl_down_sync(FULL, ex, off);
      score = __shfl_sync(FULL, ex, 0) * scale;
    }
    if constexpr (DIAG) {
      if (lane == 0) selected[(int64_t)row * S + j] = refine ? 1 : 0;
    }
    const float next = fmaxf(m, score);
    const float correction = __expf(m - next);
    const float weight = __expf(score - next);
    denom = denom * correction + weight;
    #pragma unroll
    for (int t = 0; t < N; ++t) {
      const int d = lane + 32 * t;
      if (d < VD)
        acc[t] = acc[t] * correction + weight * ld<T>(v, (kvbh * S + j) * VD + d);
    }
    m = next;
  }
  if constexpr (DIAG) {
    for (int64_t j = diagnostic_tail_index(smax, lane); j < S; j = diagnostic_tail_index(j, 32))
      selected[(int64_t)row * S + j] = 0;
  }
  if (sinks) {
    const float sink = ld<T>(sinks, h);
    const float next = fmaxf(m, sink);
    const float correction = __expf(m - next);
    denom = denom * correction + __expf(sink - next);
    #pragma unroll
    for (int t = 0; t < N; ++t) acc[t] *= correction;
  }
  // Prior art: packed decode merge.cuh (Project-Tensor P1-F1/F2, 2026),
  // IEEE 754-2008 propagation. Keep the finite reciprocal/multiply order.
  const float inv = denom != 0.f ? 1.f / denom : 0.f;
  #pragma unroll
  for (int t = 0; t < N; ++t) {
    const int d = lane + 32 * t;
    if (d < VD) st<T>(out, (int64_t)row * VD + d, acc[t] * inv);
  }
}
} // namespace spq_pf

// Host entry. Every refusal is raised here before any allocation or launch:
// the packed buffer checks (apa_spq1_buffer_check), the codebook and rotation
// tables, and the dense SP entry's checks (kernels.cu
// apa_selective_attention_sp) restated without its L == 1 split. There is no
// vector load in the prefill kernel, so no alignment refusal is needed.
// Returns (out [B,H,L,VD], mask [B,H,L,S] u8 when diagnostics, bulk [B,H,L,S]
// fp32 when dump in a TC_APA_P1_TEST_HOOKS build). Prior art: the dense SP
// entry's validation (Project-Tensor 2026), restated; ordinary precondition
// checks. No new technique.
inline std::tuple<NDArray,NDArray,NDArray> apa_spq_prefill_run(
    const NDArray& q,const NDArray& k,const NDArray& v,const NDArray& codes,const NDArray& sc,const NDArray& r,
    const NDArray& cb,float scale,float delta,bool causal,const NDArray* sinks,bool diagnostics,
    const std::string& fmt,bool dump) {
  const char* entry="apa_spq packed prefill (packed_prefill)";
  apa_spo_require_attention_extents(entry,{{"q",&q},{"k",&k},{"v",&v}});
  if(fmt!="tq4")throw std::invalid_argument(std::string(entry)+": fmt must be tq4 (prefill is served on TQ4 packed keys only)");
#if !TC_APA_P1_TEST_HOOKS
  if(dump)throw std::invalid_argument(std::string(entry)+": dump (bulk scores) needs a TC_APA_P1_TEST_HOOKS build");
#endif
  apa_spq1_buffer_check(k,codes,sc);
  if(cb.shape!=Shape({16}) || cb.dtype!=DType::Float32 || !cb.device.is_cuda() || cb.device.index!=k.device.index)
    throw std::runtime_error("packed codebook shape/device/dtype");
  const char* flag=std::getenv("TC_APA_SP");
  if(!flag || std::strcmp(flag,"1")!=0)
    throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  if(!std::isfinite(scale) || scale<=0.f || !std::isfinite(delta) || delta<0.f)
    throw std::runtime_error("apa_sp: scale must be finite positive; delta finite nonnegative");
  if(q.ndim()!=4 || k.ndim()!=4 || v.ndim()!=4)
    throw std::runtime_error("apa_sp: q/k/kq/v must be rank four");
  if(q.dtype!=DType::Float32 && q.dtype!=DType::BFloat16)
    throw std::runtime_error("apa_spq packed prefill: q/k/v must be float32 or bfloat16 (packed K dtypes)");
  for(const NDArray* a:{&q,&k,&v}){
    if(a->dtype!=q.dtype || !a->device.is_cuda() || a->device.index!=q.device.index)
      throw std::runtime_error("apa_sp: inputs must share CUDA device and dtype");
    for(int64_t n:a->shape)
      if(n<=0 || n>std::numeric_limits<int>::max())
        throw std::runtime_error("apa_sp: dimensions must be positive and fit int");
  }
  const int64_t B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3];
  const int64_t KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(k.shape[0]!=B || k.shape[3]!=D || v.shape[0]!=B || v.shape[1]!=KVH || v.shape[2]!=S || H%KVH!=0)
    throw std::runtime_error("apa_sp: shape/GQA mismatch");
  if(causal && S<L)
    throw std::runtime_error("apa_sp: bottom-right causal requires S >= L");
  if(r.shape!=Shape({KVH,D,D}) || r.dtype!=DType::Float32 || !r.device.is_cuda() || r.device.index!=q.device.index)
    throw std::runtime_error("packed rotation shape/device/dtype");
  const int cap=(int)(D>VD?D:VD);
  if(cap>TC_APA_MAXD || B>std::numeric_limits<int>::max()/H/L)
    throw std::runtime_error("apa_sp: dimension cap or launch range exceeded");
  if(sinks && (sinks->ndim()!=1 || sinks->shape[0]!=H || sinks->dtype!=q.dtype || !sinks->device.is_cuda() ||
               sinks->device.index!=q.device.index))
    throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");
  // Mask and bulk index (int64)row*S+j; every kernel index is int64 or a
  // native int checked above (B*H*L rows, S, D, VD, capacity).
  const int rows=(int)(B*H*L),G=(int)(H/KVH);
  const bool seq=D==64 && (G==4 || G==8 || cap<=64);
  NDArray out({B,H,L,VD},q.dtype,q.device),mask,bulk;
  if(diagnostics)mask=NDArray({B,H,L,S},DType::Uint8,q.device);
  if(dump){bulk=NDArray({B,H,L,S},DType::Float32,q.device);
    cudaMemset(bulk.data_ptr(),0xff,bulk.numel()*sizeof(float));}   // NaN fill outside each row's visible keys
  auto launch=[&](auto tag,auto cap_tag,auto diag_tag,auto seq_tag){
    using TT=typename decltype(tag)::type;
    constexpr int CAP=decltype(cap_tag)::value;constexpr bool DG=decltype(diag_tag)::value,SQ=decltype(seq_tag)::value;
    spq_pf::prefill<TT,CAP,DG,SQ><<<rows,32>>>((const TT*)q.data_ptr(),(const TT*)k.data_ptr(),(const TT*)v.data_ptr(),
      sinks?(const TT*)sinks->data_ptr():nullptr,(const uint8_t*)codes.data_ptr(),(const float*)sc.data_ptr(),
      (const float*)r.data_ptr(),(const float*)cb.data_ptr(),(TT*)out.data_ptr(),
      diagnostics?(uint8_t*)mask.data_ptr():nullptr,dump?(float*)bulk.data_ptr():nullptr,
      (int)H,(int)L,(int)S,(int)D,(int)VD,(int)KVH,G,(int)codes.shape[2],scale,delta,causal);
  };
  auto by_seq=[&](auto tag,auto cap_tag,auto diag_tag){
    if(seq)launch(tag,cap_tag,diag_tag,std::true_type{});else launch(tag,cap_tag,diag_tag,std::false_type{});
  };
  auto by_cap=[&](auto tag,auto diag_tag){
    if(cap<=64)by_seq(tag,std::integral_constant<int,64>{},diag_tag);
    else if(cap<=128)by_seq(tag,std::integral_constant<int,128>{},diag_tag);
    else if(cap<=256)by_seq(tag,std::integral_constant<int,256>{},diag_tag);
    else by_seq(tag,std::integral_constant<int,512>{},diag_tag);
  };
  auto by_diag=[&](auto tag){if(diagnostics)by_cap(tag,std::true_type{});else by_cap(tag,std::false_type{});};
  struct F32{using type=float;};struct BF16{using type=__nv_bfloat16;};
  if(q.dtype==DType::Float32)by_diag(F32{});else by_diag(BF16{});
  cuda_check_last("apa_spq_packed_prefill");
  return {out,mask,bulk};
}
// Bridge for the binding (block-scope declaration in bindings.cpp's
// _tensor_cuda_spo2 module; tc/ops.h is outside this round's writable set),
// as eb3.cuh's apa_p1f1_merge_tickets bridge. No prior art claimed; plumbing.
}  // namespace tc (closed for the bridge; reopened below)
std::tuple<tc::NDArray,tc::NDArray,tc::NDArray> apa_p1pf2_packed_prefill(
    const tc::NDArray& q,const tc::NDArray& k,const tc::NDArray& v,const tc::NDArray& codes,const tc::NDArray& sc,
    const tc::NDArray& r,const tc::NDArray& cb,float scale,float delta,bool causal,const tc::NDArray* sinks,
    bool diagnostics,const std::string& fmt,bool dump){
  return tc::apa_spq_prefill_run(q,k,v,codes,sc,r,cb,scale,delta,causal,sinks,diagnostics,fmt,dump);
}
namespace tc {
// APA_P1PF2_END packed_prefill
