// APA-SP1.1 additive decode implementation, included inside namespace tc.
// Selection uses an inclusive prefix over each contiguous 2048-key partition.
// A tile is only parallel execution: prefix carry survives every tile boundary.
template <typename T, int DMAX, bool WCOOP, bool DIAGNOSTICS>
__global__ void apa_selective_sp_splitk_kernel(
    const T* q, const T* k, const T* kq, const T* v, uint8_t* selected,
    float* part_m, float* part_l, float* part_acc,
    int H, int S, int D, int VD, float scale, float delta,
    int KVH, int group, int num_parts, int part_keys) {
  const int row = blockIdx.x, part = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nt = blockDim.x, nwarp = nt >> 5;
  const unsigned FULL = 0xffffffffu;
  const int b = row / H, h = row % H;
  const int64_t kvbh = (int64_t)b * KVH + h / group;
  const int j0 = part * part_keys, j1 = min(S, j0 + part_keys);
  const int64_t pr = (int64_t)row * num_parts + part;
  float* pacc = part_acc + pr * VD;
  // L=1: bottom-right causal and noncausal both expose all S keys.
  if (j0 >= j1) {
    if (tid == 0) { part_m[pr] = -1e30f; part_l[pr] = 0.f; }
    for (int d = tid; d < VD; d += nt) pacc[d] = 0.f;
    return;
  }
  __shared__ float qsh[DMAX], warp_max[4], red[256], wpart[4][DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(q, (int64_t)row * D + d);
  __syncthreads();
  constexpr int ACCN = WCOOP ? (DMAX + 31) / 32 : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int d = 0; d < ACCN; ++d) acc[d] = 0.f;
  } else {
    #pragma unroll
    for (int d = 0; d < DMAX; ++d) acc[d] = 0.f;
  }
  float carry = -INFINITY, m = -1e30f, l = 0.f;
  const int stride = WCOOP ? nwarp : nt;
  for (int first = j0; first < j1; first += stride) {
    const int j = first + (WCOOP ? warp : tid);
    const bool valid = j < j1;
    const int64_t koff = (kvbh * S + j) * D;
    float bulk = 0.f;
    if (valid) {
      if constexpr (WCOOP) {
        for (int d = lane; d < D; d += 32) bulk += qsh[d] * ld<T>(kq, koff + d);
      } else {
        for (int d = 0; d < D; ++d) bulk += qsh[d] * ld<T>(kq, koff + d);
      }
    }
    if constexpr (WCOOP) {
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(FULL, bulk, off);
      bulk = __shfl_sync(FULL, bulk, 0);
    }
    bulk = valid ? bulk * scale : -INFINITY;
    float prefix = bulk;
    if constexpr (!WCOOP) {
      // Inclusive warp prefix in absolute key order; no floating additions.
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        float previous = __shfl_up_sync(FULL, prefix, off);
        if (lane >= off) prefix = fmaxf(prefix, previous);
      }
      if (lane == 31) warp_max[warp] = prefix;
    } else {
      if (lane == 0) warp_max[warp] = bulk;
    }
    __syncthreads();
    prefix = fmaxf(prefix, carry);
    for (int w = 0; w < warp; ++w) prefix = fmaxf(prefix, warp_max[w]);
    for (int w = 0; w < nwarp; ++w) carry = fmaxf(carry, warp_max[w]);
    const bool refine = valid && bulk >= prefix - delta;
    float score = bulk;
    if (refine) {
      float ex = 0.f;
      if constexpr (WCOOP) {
        for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(k, koff + d);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ex += __shfl_down_sync(FULL, ex, off);
        ex = __shfl_sync(FULL, ex, 0);
      } else {
        for (int d = 0; d < D; ++d) ex += qsh[d] * ld<T>(k, koff + d);
      }
      score = ex * scale;
    }
    if constexpr (DIAGNOSTICS) {
      if (valid && (!WCOOP || lane == 0)) selected[(int64_t)row * S + j] = refine ? 1 : 0;
    }
    if (valid) {
      float next = fmaxf(m, score), corr = __expf(m - next), weight = __expf(score - next);
      l = l * corr + weight;
      if constexpr (WCOOP) {
        #pragma unroll
        for (int dl = 0; dl < ACCN; ++dl) {
          int d = lane + dl * 32;
          if (d < VD) acc[dl] = acc[dl] * corr + weight * ld<T>(v, (kvbh * S + j) * VD + d);
        }
      } else {
        #pragma unroll
        for (int d = 0; d < DMAX; ++d)
          if (d < VD) acc[d] = acc[d] * corr + weight * ld<T>(v, (kvbh * S + j) * VD + d);
      }
      m = next;
    }
    // Every lane finishes reading warp_max before the next tile writes it.
    __syncthreads();
  }
  // Same partial layout and reduction order as the existing split-K kernel.
  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float pmax = red[0]; __syncthreads();
  float rescale = __expf(m - pmax);
  red[tid] = (WCOOP && lane != 0) ? 0.f : l * rescale;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float pl = red[0]; __syncthreads();
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    #pragma unroll
    for (int d = 0; d < DMAX; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(FULL, val, off);
      if (lane == 0 && d < VD) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float sum = 0.f;
    for (int w = 0; w < nwarp; ++w) sum += wpart[w][d];
    pacc[d] = sum;
  }
  if (tid == 0) { part_m[pr] = pmax; part_l[pr] = pl; }
}

template <typename T>
static void apa_selective_sp_splitk_dispatch(
    const NDArray& q, const NDArray& k, const NDArray& kq, const NDArray& v,
    const NDArray* sinks, NDArray& out, NDArray* selected,
    float scale, float delta, int H, int S, int D, int VD, int KVH, int rows, int cap) {
  int num_parts, part_keys;
  // Identical to the baseline selective split-K launcher's fixed layout.
  apa_int4_fixed_partition_layout(S, num_parts, part_keys);
  NDArray pm({(int64_t)rows * num_parts}, DType::Float32, q.device);
  NDArray pl({(int64_t)rows * num_parts}, DType::Float32, q.device);
  NDArray pa({(int64_t)rows * num_parts * VD}, DType::Float32, q.device);
  auto launch = [&](auto cap_tag, auto wcoop_tag, auto diag_tag) {
    constexpr int CAP = decltype(cap_tag)::value;
    constexpr bool WCOOP = decltype(wcoop_tag)::value, DIAG = decltype(diag_tag)::value;
    dim3 grid((unsigned)rows, (unsigned)num_parts);
    apa_selective_sp_splitk_kernel<T,CAP,WCOOP,DIAG><<<grid,128>>>(
        static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
        static_cast<T*>(kq.data_ptr()), static_cast<T*>(v.data_ptr()),
        selected ? static_cast<uint8_t*>(selected->data_ptr()) : nullptr,
        static_cast<float*>(pm.data_ptr()), static_cast<float*>(pl.data_ptr()), static_cast<float*>(pa.data_ptr()),
        H,S,D,VD,scale,delta,KVH,H/KVH,num_parts,part_keys);
    cuda_check_last("apa_selective_sp_splitk");
    apa_selective_merge_kernel<T,CAP><<<rows,128>>>(
        static_cast<float*>(pm.data_ptr()), static_cast<float*>(pl.data_ptr()), static_cast<float*>(pa.data_ptr()),
        sinks ? static_cast<T*>(sinks->data_ptr()) : nullptr, static_cast<T*>(out.data_ptr()), H,VD,num_parts,sinks ? 1 : 0);
    cuda_check_last("apa_selective_sp_splitk_merge");
  };
  auto caps = [&](auto diag) {
    if (cap <= 64) launch(std::integral_constant<int,64>{},std::false_type{},diag);
    else if (cap <= 128) launch(std::integral_constant<int,128>{},std::true_type{},diag);
    else if (cap <= 256) launch(std::integral_constant<int,256>{},std::true_type{},diag);
    else launch(std::integral_constant<int,512>{},std::true_type{},diag);
  };
  if (selected) caps(std::true_type{}); else caps(std::false_type{});
}

// Part B only: reproduce the unchanged baseline's stats and pass-two decision.
// This diagnostic is separately launched and never enters attention timings.
__global__ void apa_sp1_1_baseline_mask_kernel(
    const float* q, const float* kq, const float* thr, uint8_t* mask) {
  const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int h = row / 2048, i = row % 2048;
  for (int j = warp; j < 2048; j += 4) {
    float bulk = 0.f;
    for (int d = lane; d < 64; d += 32)
      bulk += q[(int64_t)row * 64 + d] * kq[((int64_t)h * 2048 + j) * 64 + d];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(0xffffffffu, bulk, off);
    if (lane == 0) mask[(int64_t)row * 2048 + j] = j <= i && fabsf(bulk * 0.125f) >= thr[row];
  }
}

std::pair<NDArray,NDArray> apa_sp1_1_baseline_diagnostics(const NDArray& q, const NDArray& kq) {
  const char* flag = std::getenv("TC_APA_SP");
  if (!flag || std::strcmp(flag,"1") != 0)
    throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  const std::vector<int64_t> shape{1,4,2048,64};
  if (q.shape != shape || kq.shape != shape || q.dtype != DType::Float32 ||
      kq.dtype != q.dtype || !q.device.is_cuda() || !kq.device.is_cuda() || q.device.index != kq.device.index)
    throw std::runtime_error("apa_sp1_1_baseline_diagnostics: registered fp32 CUDA Part B shape only");
  NDArray thr({1,4,2048},DType::Float32,q.device);
  NDArray mask({1,4,2048,2048},DType::Uint8,q.device);
  apa_selective_stats_kernel<float,64,true><<<8192,128>>>(
      static_cast<float*>(q.data_ptr()),static_cast<float*>(kq.data_ptr()),static_cast<float*>(thr.data_ptr()),
      1,4,2048,2048,64,0.125f,1.0364333894937898f,1,4,1);
  cuda_check_last("apa_sp1_1_baseline_stats_diagnostic");
  apa_sp1_1_baseline_mask_kernel<<<8192,128>>>(
      static_cast<float*>(q.data_ptr()),static_cast<float*>(kq.data_ptr()),static_cast<float*>(thr.data_ptr()),
      static_cast<uint8_t*>(mask.data_ptr()));
  cuda_check_last("apa_sp1_1_baseline_mask_diagnostic");
  return {thr,mask};
}
