// APA-SP2: host governance and untimed score diagnostics. Included in namespace tc.
// Existing attention bodies, selection, dispatch and merge remain byte-identical.
float apa_sp2_delta(double epsilon, double e_q) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon > 1.0 ||
      !std::isfinite(e_q) || e_q < 0.0)
    throw std::runtime_error("apa_sp2: epsilon must be in (0,1]; e_q finite nonnegative");
  const double margin = -std::log(epsilon) + 2.0 * e_q;
  if (!std::isfinite(margin) || margin > std::numeric_limits<float>::max())
    throw std::runtime_error("apa_sp2: derived delta exceeds float32 range");
  float result = static_cast<float>(margin);
  if (static_cast<double>(result) < margin)
    result = std::nextafter(result, std::numeric_limits<float>::infinity());
  return result;
}

NDArray apa_selective_attention_sp_epsilon(const NDArray& q, const NDArray& k,
    const NDArray& kq, const NDArray& v, float scale, double epsilon, double e_q,
    bool is_causal, const NDArray* sinks, NDArray* selected) {
  // e_q is supplied by the Python registered-table lookup. No device prepass.
  return apa_selective_attention_sp(q, k, kq, v, scale,
      apa_sp2_delta(epsilon, e_q), is_causal, sinks, selected);
}

// Mirrors dot FMA/lane/shuffle order used by both existing paths. Diagnostic
// q chunks index the ORIGINAL query tensor, preserving causal row identities.
template <bool WCOOP>
__global__ void apa_selective_sp_scores_kernel(
    const float* q, const float* k, const float* kq, float* bulk, float* exact,
    int H, int KVH, int L, int S, int D, int first, int count, float scale) {
  const int row = blockIdx.x, h = (row / count) % H, b = row / count / H;
  const int i = row % count + first;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int64_t qo = ((int64_t)b * H * L + h * L + i) * D;
  const int64_t ko = ((int64_t)b * KVH + h / (H / KVH)) * S * D;
  for (int j = WCOOP ? warp : threadIdx.x; j < S; j += WCOOP ? 4 : 128) {
    float dot = 0.f, ex = 0.f;
    for (int d = WCOOP ? lane : 0; d < D; d += WCOOP ? 32 : 1) {
      dot += q[qo+d] * kq[ko+(int64_t)j*D+d];
      ex += q[qo+d] * k[ko+(int64_t)j*D+d];
    }
    if constexpr (WCOOP) {
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, off);
        ex += __shfl_down_sync(0xffffffffu, ex, off);
      }
    }
    if (!WCOOP || lane == 0) {
      bulk[(int64_t)row*S+j] = dot * scale;
      exact[(int64_t)row*S+j] = ex * scale;
    }
  }
}

static void apa_sp2_diagnostic_guard(const NDArray& q, const NDArray& kq,
    float scale) {
  const char* flag = std::getenv("TC_APA_SP");
  if (!flag || std::strcmp(flag,"1") != 0)
    throw std::runtime_error("apa_sp2: diagnostics require TC_APA_SP=1 (default OFF)");
  if (q.ndim()!=4 || kq.ndim()!=4 || q.dtype!=DType::Float32 ||
      kq.dtype!=q.dtype || !q.device.is_cuda() || !kq.device.is_cuda() ||
      q.device.index!=kq.device.index || !std::isfinite(scale) || scale<=0.f)
    throw std::runtime_error("apa_sp2: diagnostics require fp32 CUDA rank-four tensors and positive scale");
  for (const NDArray* a : {&q,&kq})
    for (int64_t d : a->shape)
      if (d<=0 || d>32768) throw std::runtime_error("apa_sp2: diagnostic dimension outside registered limits");
  if (q.shape[0]!=kq.shape[0] || q.shape[3]!=kq.shape[3] ||
      q.shape[1]%kq.shape[1]!=0 || (q.shape[3]!=64 && q.shape[3]!=128) ||
      q.shape[0]*q.shape[1]*q.shape[2]>std::numeric_limits<int>::max())
    throw std::runtime_error("apa_sp2: diagnostic geometry mismatch");
}

std::pair<NDArray,NDArray> apa_sp2_scores(const NDArray& q, const NDArray& k,
    const NDArray& kq, float scale, int first, int count, bool wcoop) {
  apa_sp2_diagnostic_guard(q,kq,scale);
  if (k.shape!=kq.shape || k.dtype!=kq.dtype || !k.device.is_cuda() ||
      k.device.index!=kq.device.index || first<0 || count<=0 || count>32 ||
      first>q.shape[2]-count)
    throw std::runtime_error("apa_sp2: invalid score chunk or exact keys");
  const int B=q.shape[0], H=q.shape[1], L=q.shape[2], D=q.shape[3];
  const int KVH=k.shape[1], S=k.shape[2];
  NDArray bulk({B,H,count,S},DType::Float32,q.device);
  NDArray exact({B,H,count,S},DType::Float32,q.device);
  auto launch = [&](auto tag) {
    apa_selective_sp_scores_kernel<decltype(tag)::value><<<B*H*count,128>>>(
        static_cast<float*>(q.data_ptr()),static_cast<float*>(k.data_ptr()),
        static_cast<float*>(kq.data_ptr()),static_cast<float*>(bulk.data_ptr()),
        static_cast<float*>(exact.data_ptr()),H,KVH,L,S,D,first,count,scale);
  };
  if (wcoop) launch(std::true_type{}); else launch(std::false_type{});
  cuda_check_last("apa_sp2_scores");
  return {bulk,exact};
}

NDArray apa_sp2_baseline_thresholds(const NDArray& q, const NDArray& kq,
    float scale, float zthr, bool causal) {
  apa_sp2_diagnostic_guard(q,kq,scale);
  if (!std::isfinite(zthr) || (causal && kq.shape[2]<q.shape[2]))
    throw std::runtime_error("apa_sp2: invalid diagnostic threshold or causal geometry");
  const int B=q.shape[0], H=q.shape[1], L=q.shape[2], D=q.shape[3];
  const int KVH=kq.shape[1], S=kq.shape[2], rows=B*H*L;
  NDArray thr({B,H,L},DType::Float32,q.device);
  auto launch = [&](auto cap, auto coop) {
    // Invoke the unchanged baseline stats body, including split-K stats order.
    apa_selective_stats_kernel<float,decltype(cap)::value,decltype(coop)::value><<<rows,128>>>(
        static_cast<float*>(q.data_ptr()),static_cast<float*>(kq.data_ptr()),
        static_cast<float*>(thr.data_ptr()),B,H,L,S,D,scale,zthr,causal,KVH,H/KVH);
  };
  if (D==64 && apa_selective_decode_shaped(rows,L))
    launch(std::integral_constant<int,64>{},std::false_type{});
  else if (D==64) launch(std::integral_constant<int,64>{},std::true_type{});
  else launch(std::integral_constant<int,128>{},std::true_type{});
  cuda_check_last("apa_sp2_baseline_thresholds");
  return thr;
}
