// APA-Quant attention: C++ orchestration (forward + backward).
//
// This is a faithful C++ port of tensor_gpu_v2._core.apa_quant_attention, with
// the per-call Python/CuPy overhead removed: the per-head quantization loop,
// the refinement selection, the tiled online-softmax block loop, the adaptive
// per-head budget allocation and the inverse-normal-CDF all run in C++.
//
// GEMMs go through ATen matmul (cuBLAS) -- identical to PyTorch's "math" SDPA
// backend -- so a benchmark measures the APA *algorithm*, not BLAS quality.
// The two APA-specific fused elementwise ops live in apa_kernels.cu.
//
// Shapes are normalized to 4D (B, H, L, D) by the Python wrapper before the
// call, so the C++ here always sees rank-4 tensors.

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cmath>
#include <tuple>
#include <vector>

namespace apa {

// Declared in apa_kernels.cu.
at::Tensor quantize_gather(const at::Tensor& rotated,
                           const at::Tensor& boundaries,
                           const at::Tensor& codebook);
std::tuple<at::Tensor, at::Tensor> mix_scores(const at::Tensor& ranking,
                                              const at::Tensor& bulk,
                                              const at::Tensor& thr);

namespace {

constexpr double kNegInf = -1e9;  // matches the reference's masked fill value

// Acklam's rational approximation to the inverse normal CDF. Matches
// scipy.stats.norm.ppf to ~1e-9, which is the value the reference caches via
// _apa_zscore(p) = norm.ppf(1 - p).
double norm_ppf(double p) {
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425;
  const double phigh = 1.0 - plow;
  if (p <= 0.0) return -INFINITY;
  if (p >= 1.0) return INFINITY;
  if (p < plow) {
    double q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (p > phigh) {
    double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  double q = p - 0.5;
  double r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

inline double clipd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Port of _allocate_per_head_budget. Returns per-head fraction in [min,max].
std::vector<double> allocate_budget(const std::vector<double>& conc, int H,
                                    double global_budget, double min_pct = 0.01,
                                    double max_pct = 0.50) {
  double sum = 1e-30;
  for (double c : conc) sum += c;
  std::vector<double> alloc(H);
  for (int h = 0; h < H; ++h)
    alloc[h] = clipd(conc[h] / sum * H * global_budget, min_pct, max_pct);
  const double target = H * global_budget;
  for (int it = 0; it < 10; ++it) {
    double cur = 0.0;
    for (double v : alloc) cur += v;
    if (std::abs(cur - target) < 1e-6) break;
    int free_count = 0;
    for (int h = 0; h < H; ++h) {
      bool clamped = (alloc[h] <= min_pct + 1e-8) || (alloc[h] >= max_pct - 1e-8);
      if (!clamped) ++free_count;
    }
    if (free_count == 0) break;
    double adj = (target - cur) / free_count;
    for (int h = 0; h < H; ++h) {
      bool clamped = (alloc[h] <= min_pct + 1e-8) || (alloc[h] >= max_pct - 1e-8);
      if (!clamped) alloc[h] = clipd(alloc[h] + adj, min_pct, max_pct);
    }
  }
  return alloc;
}

// Port of _compute_per_head_concentration -> returns a length-H host vector.
std::vector<double> per_head_concentration(const at::Tensor& bulk_scores) {
  auto bs = bulk_scores.to(at::kFloat);
  auto smx = at::softmax(bs, -1);
  auto maxw = std::get<0>(smx.max(-1));           // (B,H,L)
  auto meanw = smx.mean(-1);                       // (B,H,L)
  auto conc = maxw / (meanw + 1e-10);              // (B,H,L)
  auto per_head = conc.mean(at::IntArrayRef({0, 2}));  // (H,)
  auto host = per_head.to(at::kCPU).to(at::kDouble).contiguous();
  const double* p = host.data_ptr<double>();
  return std::vector<double>(p, p + host.numel());
}

// Quantize keys per head: unit -> rotate -> searchsorted+gather -> rotate back
// -> rescale by norm. Computed in fp32 for stability, cast back to key dtype.
at::Tensor quantize_keys(const at::Tensor& key, const at::Tensor& rotations,
                         const at::Tensor& codebook, const at::Tensor& boundaries) {
  auto kf = key.to(at::kFloat);
  auto norms = kf.pow(2).sum(-1, /*keepdim=*/true).clamp_min(0).sqrt();  // (B,H,S,1)
  auto positive = norms > 0;
  auto safe = at::where(positive, norms, at::ones_like(norms));
  auto unit = at::where(positive, kf / safe, at::zeros_like(kf));
  auto rot = rotations.to(at::kFloat);                       // (H,D,D)
  auto rotated = at::matmul(unit, rot.transpose(-2, -1));     // (B,H,S,D)
  auto centroids = quantize_gather(rotated.contiguous(), boundaries.to(at::kFloat),
                                   codebook.to(at::kFloat));
  auto recon_unit = at::matmul(centroids, rot);              // (B,H,S,D)
  auto recon = recon_unit * norms;                           // broadcast last dim
  return recon.to(key.scalar_type());
}

// Build a (rows,) fp32 threshold tensor for a block of query rows.
//
//   abs = |ranking|, with causal-masked positions zeroed (matches reference).
//   full_path        -> threshold = -inf (everything refines == exact attention)
//   adaptive (z_h)   -> mean + z_h[head] * std        (population std)
//   S > 256          -> mean + z * std
//   otherwise        -> exact top-k via kthvalue
at::Tensor compute_threshold(const at::Tensor& ranking,
                             const c10::optional<at::Tensor>& causal_mask,
                             bool full_path, bool adaptive,
                             const c10::optional<at::Tensor>& z_head,  // (H,) fp32
                             double z_scalar, int64_t S, int64_t refine_k) {
  const int64_t B = ranking.size(0), H = ranking.size(1), L = ranking.size(2);
  auto opts_f = ranking.options().dtype(at::kFloat);

  if (full_path) {
    return at::full({B, H, L}, -INFINITY, opts_f);
  }

  auto absr = ranking.abs().to(at::kFloat);
  if (causal_mask.has_value()) {
    absr = at::where(*causal_mask, at::zeros_like(absr), absr);
  }

  if (adaptive) {
    auto mean_ab = absr.mean(-1);                       // (B,H,L)
    auto std_ab = absr.std(-1, /*unbiased=*/false);     // (B,H,L)
    auto zh = z_head->to(opts_f).view({1, H, 1});
    return mean_ab + zh * std_ab;
  }
  if (S > 256) {
    auto mean_ab = absr.mean(-1);
    auto std_ab = absr.std(-1, /*unbiased=*/false);
    return mean_ab + z_scalar * std_ab;
  }
  // Exact top-k. threshold_idx = S - refine_k (ascending partition index);
  // kthvalue is 1-based, so k = threshold_idx + 1.
  int64_t k = S - refine_k + 1;
  k = std::max<int64_t>(1, std::min<int64_t>(S, k));
  auto thr = std::get<0>(absr.kthvalue(k, -1, /*keepdim=*/false));  // (B,H,L)
  return thr;
}

// Apply causal + additive attn_mask to a (B,H,rows,S) score tensor in place
// of the caller's choosing. row_base is the global index of the first row.
at::Tensor apply_masks(at::Tensor scores, bool is_causal,
                       const c10::optional<at::Tensor>& attn_mask,
                       int64_t row_base, int64_t rows, int64_t S,
                       c10::optional<at::Tensor>* out_causal_mask) {
  if (is_causal) {
    auto dev = scores.device();
    auto ri = at::arange(row_base, row_base + rows,
                         at::TensorOptions().dtype(at::kLong).device(dev))
                  .view({rows, 1});
    auto ci = at::arange(0, S, at::TensorOptions().dtype(at::kLong).device(dev))
                  .view({1, S});
    auto cmask = (ci > ri);  // (rows,S) bool
    scores = scores.masked_fill(cmask, kNegInf);
    if (out_causal_mask) *out_causal_mask = cmask;
  } else if (out_causal_mask) {
    *out_causal_mask = c10::nullopt;
  }
  if (attn_mask.has_value()) {
    auto m = *attn_mask;
    if (m.dim() == 2) {
      scores = scores + m.slice(0, row_base, row_base + rows);
    } else {
      scores = scores + m.slice(2, row_base, row_base + rows);
    }
  }
  return scores;
}

}  // namespace

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
//
// Returns {output, key_quant, refine_mask (uint8, B,H,L,S), use_tiled (bool)}.
// The Python autograd.Function saves key_quant and refine_mask for backward.
std::vector<at::Tensor> apa_forward(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& rotations, const at::Tensor& codebook,
    const at::Tensor& boundaries, double refine_percentile, bool is_causal,
    const c10::optional<at::Tensor>& attn_mask, double scale, double dropout_p,
    int64_t block_size, bool adaptive_heads, bool training) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "APA forward requires CUDA tensors");
  TORCH_CHECK(query.dim() == 4, "expected 4D (B,H,L,D) query");

  const int64_t B = query.size(0), H = query.size(1), L = query.size(2),
                D = query.size(3);
  const int64_t S = key.size(2);
  refine_percentile = std::max(0.0, std::min(1.0, refine_percentile));
  const int64_t refine_k = std::max<int64_t>(1, (int64_t)(S * refine_percentile));

  const bool full_precision = (refine_percentile >= 1.0);
  // The reference only builds a per-head budget when refine_k < S, so a budget
  // that already spans the whole sequence (refine_k >= S) always takes the
  // exact full-precision path regardless of adaptive_heads.
  const bool full_path_global = full_precision || (refine_k >= S);

  // Quantized keys (skip entirely in the exact full-precision path).
  at::Tensor key_quant =
      full_precision ? key : quantize_keys(key, rotations, codebook, boundaries);

  const int64_t attn_bytes = B * H * L * S * 4;
  const bool use_tiled = attn_bytes > (256LL * 1024 * 1024);

  auto kq_t = key_quant.transpose(-2, -1).contiguous();  // (B,H,D,S)
  auto k_t = key.transpose(-2, -1).contiguous();

  // Precompute adaptive per-head z if requested (uses full or first-block bulk).
  bool adaptive = adaptive_heads && H > 1 && refine_k < S && !full_precision;
  c10::optional<at::Tensor> z_head;
  const double z_scalar = full_path_global ? 0.0 : norm_ppf(1.0 - refine_percentile);

  auto refine_mask = at::empty({B, H, L, S}, query.options().dtype(at::kByte));
  at::Tensor output;

  if (!use_tiled) {
    auto bulk = at::matmul(query, kq_t) * scale;     // (B,H,L,S)
    auto ranking = at::matmul(query, k_t) * scale;
    c10::optional<at::Tensor> causal_mask;
    bulk = apply_masks(bulk, is_causal, attn_mask, 0, L, S, &causal_mask);
    ranking = apply_masks(ranking, is_causal, attn_mask, 0, L, S, nullptr);

    if (adaptive) {
      auto conc = per_head_concentration(bulk);
      auto pct = allocate_budget(conc, (int)H, refine_percentile);
      std::vector<float> zh(H);
      for (int h = 0; h < H; ++h) zh[h] = (float)norm_ppf(1.0 - pct[h]);
      z_head = at::from_blob(zh.data(), {H}, at::TensorOptions().dtype(at::kFloat))
                   .clone()
                   .to(query.device());
    }

    auto thr = compute_threshold(ranking, causal_mask, full_path_global, adaptive,
                                 z_head, z_scalar, S, refine_k)
                   .reshape({-1})
                   .contiguous();
    at::Tensor scores, mask;
    std::tie(scores, mask) = mix_scores(ranking, bulk, thr);
    refine_mask = mask;

    auto attn_w = at::softmax(scores, -1);
    if (dropout_p > 0 && training) attn_w = at::dropout(attn_w, dropout_p, true);
    output = at::matmul(attn_w, value);
  } else {
    output = at::zeros({B, H, L, D}, query.options());
    auto m_prev = at::full({B, H, L, 1}, -INFINITY, query.options());
    auto l_prev = at::zeros({B, H, L, 1}, query.options());

    for (int64_t i = 0; i < L; i += block_size) {
      int64_t end_i = std::min(i + block_size, L);
      int64_t bl = end_i - i;
      auto Qi = query.slice(2, i, end_i);
      auto bulk = at::matmul(Qi, kq_t) * scale;       // (B,H,bl,S)
      auto ranking = at::matmul(Qi, k_t) * scale;
      c10::optional<at::Tensor> causal_mask;
      bulk = apply_masks(bulk, is_causal, attn_mask, i, bl, S, &causal_mask);
      ranking = apply_masks(ranking, is_causal, attn_mask, i, bl, S, nullptr);

      if (adaptive && i == 0) {
        auto conc = per_head_concentration(bulk);
        auto pct = allocate_budget(conc, (int)H, refine_percentile);
        std::vector<float> zh(H);
        for (int h = 0; h < H; ++h) zh[h] = (float)norm_ppf(1.0 - pct[h]);
        z_head = at::from_blob(zh.data(), {H}, at::TensorOptions().dtype(at::kFloat))
                     .clone()
                     .to(query.device());
      }

      auto thr = compute_threshold(ranking, causal_mask, full_path_global, adaptive,
                                   z_head, z_scalar, S, refine_k)
                     .reshape({-1})
                     .contiguous();
      at::Tensor scores, mask;
      std::tie(scores, mask) = mix_scores(ranking, bulk, thr);
      refine_mask.slice(2, i, end_i).copy_(mask);

      // Online softmax accumulation.
      auto m_curr = std::get<0>(scores.max(-1, /*keepdim=*/true));
      auto m_bp = m_prev.slice(2, i, end_i);
      auto l_bp = l_prev.slice(2, i, end_i);
      auto o_bp = output.slice(2, i, end_i);
      auto m_new = at::maximum(m_bp, m_curr);
      auto exp_s = at::exp(scores - m_new);
      auto corr = at::exp(m_bp - m_new);
      auto l_new = corr * l_bp + exp_s.sum(-1, /*keepdim=*/true);
      auto pv = at::matmul(exp_s, value);
      output.slice(2, i, end_i).copy_((corr * l_bp * o_bp + pv) / l_new);
      m_prev.slice(2, i, end_i).copy_(m_new);
      l_prev.slice(2, i, end_i).copy_(l_new);
    }
    if (dropout_p > 0 && training) output = at::dropout(output, dropout_p, true);
  }

  return {output, key_quant, refine_mask};
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
//
// Faithful port of the reference _backward: gradients are computed from the
// full-precision softmax weights, and grad_q is split between the full key
// (refined positions) and the quantized key (bulk positions).
std::vector<at::Tensor> apa_backward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& key_quant,
    const at::Tensor& refine_mask, double scale, bool is_causal,
    const c10::optional<at::Tensor>& attn_mask, bool use_tiled,
    int64_t block_size) {
  const int64_t B = query.size(0), H = query.size(1), L = query.size(2);
  const int64_t S = key.size(2);
  auto mask_bool = refine_mask.to(at::kBool);

  auto grad_q = at::zeros_like(query);
  auto grad_k = at::zeros_like(key);
  auto grad_v = at::zeros_like(value);
  auto k_t = key.transpose(-2, -1).contiguous();
  auto v_t = value.transpose(-2, -1).contiguous();

  auto block_grad = [&](int64_t i, int64_t end_i) {
    auto Qi = query.slice(2, i, end_i);
    auto dOi = grad_out.slice(2, i, end_i);
    auto rmask = mask_bool.slice(2, i, end_i);

    auto scores = at::matmul(Qi, k_t) * scale;  // (B,H,bl,S)
    c10::optional<at::Tensor> causal_mask;
    scores = apply_masks(scores, is_causal, attn_mask, i, end_i - i, S, &causal_mask);

    auto attn_w = at::softmax(scores, -1);
    // grad_v += attn_w^T @ dO
    grad_v += at::matmul(attn_w.transpose(-2, -1), dOi);
    auto dAttn = at::matmul(dOi, v_t);  // (B,H,bl,S)
    auto sum_dA = (attn_w * dAttn).sum(-1, /*keepdim=*/true);
    auto dS = attn_w * (dAttn - sum_dA) * scale;
    if (causal_mask.has_value()) dS = dS.masked_fill(*causal_mask, 0.0);

    auto zeros = at::zeros_like(dS);
    auto dS_full = at::where(rmask, dS, zeros);
    auto dS_bulk = at::where(rmask, zeros, dS);
    grad_q.slice(2, i, end_i) +=
        at::matmul(dS_full, key) + at::matmul(dS_bulk, key_quant);
    grad_k += at::matmul(dS.transpose(-2, -1), Qi);
  };

  if (!use_tiled) {
    block_grad(0, L);
  } else {
    for (int64_t i = 0; i < L; i += block_size) {
      block_grad(i, std::min(i + block_size, L));
    }
  }
  return {grad_q, grad_k, grad_v};
}

}  // namespace apa

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &apa::apa_forward, "APA-Quant attention forward (CUDA)");
  m.def("backward", &apa::apa_backward, "APA-Quant attention backward (CUDA)");
  m.def("quantize_gather", &apa::quantize_gather, "searchsorted + codebook gather");
  m.def("mix_scores", &apa::mix_scores, "fused refinement score mix");
}
