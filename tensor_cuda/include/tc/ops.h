// ops.h: differentiable operations on Tensor. Each builds the forward NDArray
// result and registers a grad_fn implementing the vector-Jacobian product.

#pragma once

#include "tc/autograd.h"

#include <tuple>
#include <vector>

namespace tc {
namespace ops {

// Arithmetic (broadcasting). Scalar overloads avoid allocating a full tensor.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor add_scalar(const Tensor& a, double s);
Tensor mul_scalar(const Tensor& a, double s);
Tensor neg(const Tensor& a);

// Math / activations.
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor silu(const Tensor& a);
Tensor abs(const Tensor& a);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor tan(const Tensor& a);
Tensor asin(const Tensor& a);
Tensor acos(const Tensor& a);
Tensor atan(const Tensor& a);
Tensor sinh(const Tensor& a);
Tensor cosh(const Tensor& a);
Tensor log2(const Tensor& a);
Tensor log10(const Tensor& a);
Tensor sign(const Tensor& a);
Tensor floor(const Tensor& a);
Tensor ceil(const Tensor& a);
Tensor round(const Tensor& a);
Tensor isnan(const Tensor& a);
Tensor isinf(const Tensor& a);
Tensor isfinite(const Tensor& a);
Tensor nan_to_num(const Tensor& a, double nan, double posinf, double neginf);
Tensor reciprocal(const Tensor& a);
Tensor clamp(const Tensor& a, double lo, double hi);
Tensor maximum(const Tensor& a, const Tensor& b);
Tensor minimum(const Tensor& a, const Tensor& b);
Tensor slice(const Tensor& a, int dim, int64_t start, int64_t len);

// Linear algebra.
Tensor matmul(const Tensor& a, const Tensor& b, float alpha = 1.f, bool trans_b = false);

// INT4 group-quantized linear: y = x @ dequant(W). The weight is FROZEN
// (packed uint8 + fp16 scales/zeros — not a differentiable parameter); the
// VJP exists for x only: dx = g @ W_kn^T with the (K,N) dequantized weight
// rebuilt transiently inside the backward and freed on return. Nothing
// weight-sized is retained in the graph, so a frozen INT4 model can sit in
// a training loop (e.g. SCRIBE's L-func reader) at zero resident overhead.
Tensor int4_linear(const Tensor& x, const Tensor& packed, const Tensor& scales,
                   const Tensor& zeros, int group_size);
Tensor int4_linear_fused(const Tensor& x, const Tensor& packed,
                         const Tensor& scales, const Tensor& zeros,
                         int group_size);
Tensor intn_linear(const Tensor& x, const Tensor& packed, const Tensor& scales,
                   const Tensor& zeros, int bits, int64_t in_features,
                   int group_size);
Tensor intn_linear_fused(const Tensor& x, const Tensor& packed,
                         const Tensor& scales, const Tensor& zeros, int bits,
                         int64_t in_features, int group_size);
Tensor mxfp4_linear(const Tensor& x, const Tensor& blocks,
                    const Tensor& scales);
Tensor mxfp4_linear_expert(const Tensor& x, const Tensor& blocks,
                           const Tensor& scales, int64_t expert_idx);

// APA selective attention, differentiable + O(L) memory (graft-native training).
// Selection is a stop-gradient (kq detached); q,k,v receive gradients.
Tensor apa_selective_train(const Tensor& q, const Tensor& k, const Tensor& kq,
                           const Tensor& v, float scale, float zthr,
                           bool is_causal);

// Fused causal softmax (inference-only: backward throws).
Tensor causal_softmax(const Tensor& scores);

// Fused RMSNorm over the last dim (inference-only: backward throws).
Tensor rms_norm(const Tensor& x, const Tensor& w, double eps);
Tensor rope_apply(const Tensor& x, const Tensor& cs, const Tensor& sn,
                  int64_t pos0, bool inverse = false,
                  bool pair_swap = false);
void write_rows(Tensor& buf, const Tensor& src, int64_t start);
Tensor export_rows(const Tensor& cache, int dim, int64_t start, int64_t len);
Tensor export_rope_rows(const Tensor& cache, const Tensor& cs,
                        const Tensor& sn, int dim, int64_t start,
                        int64_t len, int64_t pos0, bool inverse = false,
                        bool pair_swap = false);
std::tuple<Tensor, Tensor> export_row_pair(
    const Tensor& raw_cache, const Tensor& rope_cache, const Tensor& cs,
    const Tensor& sn, int raw_dim, int rope_dim, int64_t raw_start,
    int64_t rope_start, int64_t len, int64_t pos0, bool inverse = false,
    bool pair_swap = false);
std::tuple<std::vector<Tensor>, std::vector<Tensor>> export_row_pairs(
    const std::vector<Tensor>& raw_caches,
    const std::vector<Tensor>& rope_caches, const Tensor& cs,
    const Tensor& sn, int raw_dim, int rope_dim,
    const std::vector<int64_t>& raw_starts,
    const std::vector<int64_t>& rope_starts, int64_t len, int64_t pos0,
    bool inverse = false, bool pair_swap = false);
std::tuple<std::vector<Tensor>, std::vector<Tensor>> swap_row_pairs_with_rope(
    const std::vector<Tensor>& raw_caches,
    const std::vector<Tensor>& rope_caches,
    const std::vector<Tensor>& raw_inserts,
    const std::vector<Tensor>& rope_inserts, const Tensor& cs,
    const Tensor& sn, int raw_dim, int rope_dim, int64_t head_tokens,
    int64_t tail_start, int64_t pos0, bool pair_swap = false);
std::tuple<std::vector<Tensor>, std::vector<Tensor>> evict_row_pairs(
    const std::vector<Tensor>& raw_caches,
    const std::vector<Tensor>& rope_caches, int raw_dim, int rope_dim,
    int64_t head_tokens, int64_t drop_tokens);
std::tuple<std::vector<Tensor>, std::vector<Tensor>, int64_t>
arena_row_pair_transaction(
    const std::vector<Tensor>& raw_caches,
    const std::vector<Tensor>& rope_caches,
    const std::vector<Tensor>& raw_inserts,
    const std::vector<Tensor>& rope_inserts, const Tensor& cs,
    const Tensor& sn, int raw_dim, int rope_dim, int64_t sink_tokens,
    int64_t current_mount_tokens, int64_t arena_width,
    bool pair_swap = false);
Tensor splice_rows(const Tensor& old_cache, const Tensor& insert,
                   int dim, int64_t head_tokens, int64_t tail_start);
Tensor evict_rows(const Tensor& old_cache, int dim, int64_t head_tokens,
                  int64_t drop_tokens);

// Reductions.
Tensor sum(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor mean(const Tensor& a, const std::vector<int>& axes, bool keepdim);

// Shape.
Tensor reshape(const Tensor& a, const Shape& shape);
Tensor transpose_last(const Tensor& a);

// Composite (built from primitives, autograd flows automatically).
Tensor softmax(const Tensor& a, int axis);
Tensor mse_loss(const Tensor& pred, const Tensor& target);

// ---- Phase 2: broadened op surface ----
Tensor pow_scalar(const Tensor& a, double exponent);
Tensor sub_scalar(const Tensor& a, double s);  // a - s

// Comparisons return detached (non-differentiable) 0/1 tensors.
Tensor compare(const Tensor& a, const Tensor& b, int op);
Tensor compare_scalar(const Tensor& a, double s, int op);

Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y);
Tensor masked_fill(const Tensor& a, const Tensor& mask, double value);

// Reductions with gradients.
Tensor max(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor min(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor var(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor std(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor prod(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor argmax(const Tensor& a, int axis);   // detached int64
Tensor argmin(const Tensor& a, int axis);
Tensor cumsum(const Tensor& a, int axis);
Tensor gather(const Tensor& a, int dim, const Tensor& index);
Tensor flip(const Tensor& a, const std::vector<int>& dims);
std::tuple<Tensor, Tensor> topk(const Tensor& a, int k, bool largest);  // (values, indices)

// Shape.
Tensor permute(const Tensor& a, const std::vector<int>& dims);
Tensor transpose(const Tensor& a, int dim0, int dim1);
Tensor squeeze(const Tensor& a, int dim);
Tensor unsqueeze(const Tensor& a, int dim);
Tensor expand(const Tensor& a, const Shape& shape);
Tensor flatten(const Tensor& a, int start_dim, int end_dim);
Tensor cat(const std::vector<Tensor>& ts, int dim);
Tensor stack(const std::vector<Tensor>& ts, int dim);

// Losses / log-domain.
Tensor log_softmax(const Tensor& a, int axis);
Tensor cross_entropy(const Tensor& logits, const Tensor& onehot_target);

// Indexing.
Tensor embedding(const Tensor& weight, const Tensor& idx);  // idx: int64 Tensor

// Stop-gradient: returns a constant view sharing storage (no autograd parents).
Tensor detach(const Tensor& a);
// Differentiable dtype cast (grad cast back to the input dtype).
Tensor cast(const Tensor& a, DType dt);

// Conv/pool (NCHW). Conv2D = im2col + matmul (composed in Python nn).
Tensor im2col(const Tensor& a, int kh, int kw, int sh, int sw, int ph, int pw);
// col2im as a forward op (scatter cols into an (N,C,OH,OW) image); for ConvTranspose.
Tensor col2im(const Tensor& cols, int64_t N, int64_t C, int64_t OH, int64_t OW,
              int kh, int kw, int sh, int sw, int ph, int pw);
Tensor avg_pool2d(const Tensor& a, int kh, int kw, int sh, int sw, int ph, int pw);
Tensor max_pool2d(const Tensor& a, int kh, int kw, int sh, int sw, int ph, int pw);

}  // namespace ops
}  // namespace tc
