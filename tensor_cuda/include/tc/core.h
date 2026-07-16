// Project Tensor — standalone CUDA tensor library (no PyTorch, no CuPy).
//
// core.h: device memory (Storage), the raw n-dimensional array (NDArray) that
// every CUDA kernel operates on, dtype/device descriptors, and the host-side
// op declarations. NDArray carries NO autograd state — it is the workhorse.
// Autograd lives one layer up (autograd.h).
//
// Phase 1 design choices (see ROADMAP.md):
//   * NDArrays are always contiguous, row-major. transpose/permute physically
//     materialize; reshape is metadata-only. This keeps kernels simple and
//     correct; strided views are a later optimization.
//   * Kernels are dtype-generic by computing in fp32 (load->float, store->T),
//     so fp32 and fp16 share one code path.
//   * Max rank is 8 (TC_MAX_DIMS), enough for NCHW + attention (B,H,L,D).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tc {

constexpr int TC_MAX_DIMS = 8;

// ----------------------------------------------------------------------- dtype
enum class DType : int8_t { Float32 = 0, Float16 = 1, Int64 = 2, Bool = 3, Uint8 = 4, BFloat16 = 5 };

size_t dtype_size(DType dt);
const char* dtype_name(DType dt);
DType dtype_from_string(const std::string& s);

// ---------------------------------------------------------------------- device
enum class DeviceType : int8_t { CPU = 0, CUDA = 1 };

struct Device {
  DeviceType type = DeviceType::CUDA;
  int index = 0;
  bool is_cuda() const { return type == DeviceType::CUDA; }
  std::string str() const;
  static Device from_string(const std::string& s);
};

using Shape = std::vector<int64_t>;

int64_t numel_of(const Shape& shape);
Shape contiguous_strides(const Shape& shape);

// --------------------------------------------------------------------- Storage
// Reference-counted device (or host) memory buffer.
struct Storage {
  void* ptr = nullptr;
  size_t nbytes = 0;
  Device device;
  bool pooled = false;  // allocated via cudaMallocAsync (small) vs raw cudaMalloc (large)
  // Monotonic host-side generation shared by every view of this allocation.
  // In-place engine operations bump it after enqueueing their write, allowing
  // persistent derived tensors to invalidate without hashing device bytes.
  uint64_t revision = 0;

  Storage(size_t nbytes, Device device);
  ~Storage();
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
};

// --------------------------------------------------------------------- NDArray
// A view-able handle over a Storage: shape + dtype + element offset. In Phase 1
// the data is contiguous; shape/strides are kept in sync by contiguous_strides.
class NDArray {
 public:
  std::shared_ptr<Storage> storage;
  Shape shape;
  DType dtype = DType::Float32;
  Device device;
  int64_t offset = 0;  // in elements

  NDArray() = default;
  NDArray(const Shape& shape, DType dtype, Device device);

  static NDArray empty(const Shape& shape, DType dtype, Device device);
  static NDArray zeros(const Shape& shape, DType dtype, Device device);
  static NDArray full(const Shape& shape, double value, DType dtype, Device device);

  int64_t numel() const { return numel_of(shape); }
  int ndim() const { return static_cast<int>(shape.size()); }
  bool defined() const { return storage != nullptr; }
  void* data_ptr() const;
  uint64_t revision() const { return storage ? storage->revision : 0; }
  void mark_modified() const {
    if (storage) ++storage->revision;
  }

  // Host<->device transfer (host buffer is contiguous, matching `dtype`).
  static NDArray from_host(const void* src, const Shape& shape, DType dtype, Device device);
  void to_host(void* dst) const;

  NDArray to(Device device) const;
  NDArray astype(DType dtype) const;
  NDArray clone() const;
  NDArray reshape(const Shape& new_shape) const;  // metadata-only (contiguous)
  NDArray contiguous() const { return *this; }     // Phase 1: always contiguous
};

// ------------------------------------------------------------- raw CUDA ops
// All return freshly-allocated contiguous NDArrays. Declared here, launched in
// src/kernels.cu and src/matmul.cu.

// Elementwise binary with NumPy broadcasting. op: 0=add 1=sub 2=mul 3=div.
NDArray ew_binary(const NDArray& a, const NDArray& b, int op);
NDArray ew_scalar(const NDArray& a, double scalar, int op, bool scalar_lhs);

// Elementwise unary. op enum mirrors UnaryOp below.
enum UnaryOp {
  U_NEG, U_EXP, U_LOG, U_SQRT, U_RELU, U_SIGMOID, U_TANH, U_GELU,
  U_GELU_EXACT, U_ERF, U_SILU,
  U_RECIP, U_ABS, U_SIGN, U_SIN, U_COS,
  U_TAN, U_ASIN, U_ACOS, U_ATAN, U_SINH, U_COSH,
  U_LOG2, U_LOG10, U_FLOOR, U_CEIL, U_ROUND, U_ISNAN, U_ISINF, U_ISFINITE,
};
NDArray ew_unary(const NDArray& a, int op);
NDArray ew_pow(const NDArray& a, double exponent);
NDArray ew_clamp(const NDArray& a, double lo, double hi);  // clamp to [lo,hi]
NDArray ew_nan_to_num(const NDArray& a, double nan, double posinf, double neginf);

// Comparisons. op: 0=gt 1=ge 2=lt 3=le 4=eq 5=ne. Result is same dtype, 0/1.
NDArray compare(const NDArray& a, const NDArray& b, int op);     // broadcasting
NDArray compare_scalar(const NDArray& a, double s, int op);
// Select: cond != 0 ? x : y (all NumPy-broadcast together).
NDArray where_nd(const NDArray& cond, const NDArray& x, const NDArray& y);

// Reductions. axes empty => reduce all. Returns reduced array (keepdim aware).
NDArray reduce_sum(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_max(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_min(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_prod(const NDArray& a, const std::vector<int>& axes, bool keepdim);
// Sum `a` down to `target` shape (NumPy-broadcast reduction); used by autograd.
NDArray reduce_to(const NDArray& a, const Shape& target);

// argmax/argmin along a single axis (returns int64, axis removed).
NDArray reduce_arg(const NDArray& a, int axis, bool is_max);
// Block-per-row argmax over the LAST axis only (returns int64, axis removed).
// Decode-loop fast path (Phase 1.1): grid-stride + warp-shuffle reduction so
// a single row (e.g. one decode step's vocab-sized logits) still parallelizes
// across a full block, unlike reduce_arg's one-thread-per-row generic path.
// Tie-break matches numpy.argmax (lowest index wins).
NDArray argmax_last_axis(const NDArray& a);
// Cumulative sum along `axis` (same shape).
NDArray cumsum_nd(const NDArray& a, int axis);
// gather along `dim` using int64 `index` (index.shape == output shape).
NDArray gather_nd(const NDArray& a, int dim, const NDArray& index);
// scatter-add `src` into a zeroed `shape` tensor along `dim` by `index` (gather bwd).
NDArray scatter_add_nd(const Shape& shape, DType dtype, int dim,
                       const NDArray& index, const NDArray& src);
// reverse `a` along the given dims.
NDArray flip_nd(const NDArray& a, const std::vector<int>& dims);
// top-k along the last axis -> (values same dtype, indices int64), last dim = k.
std::tuple<NDArray, NDArray> topk_nd(const NDArray& a, int k, bool largest);

// Shape ops.
NDArray transpose2d_last(const NDArray& a);            // swap last two dims
NDArray permute(const NDArray& a, const std::vector<int>& dims);
NDArray broadcast_to(const NDArray& a, const Shape& shape);

// Concatenate along `dim`; slice [start, start+len) along `dim` (cat backward).
NDArray cat_nd(const std::vector<NDArray>& arrs, int dim);
NDArray slice_nd(const NDArray& a, int dim, int64_t start, int64_t len);
// Scatter `small` into a zeroed `big_shape` tensor at [start..) along dim (slice backward).
NDArray pad_into(const NDArray& small, const Shape& big_shape, int dim, int64_t start);

// Batched matmul over leading dims; last two dims are (M,K)x(K,N). cuBLAS.
// trans_b reads b as (N,K) row-major via OP_T (no transpose copy); alpha is
// folded into the GEMM (applied in the fp32 accumulator, before the 16-bit store).
NDArray matmul(const NDArray& a, const NDArray& b, float alpha = 1.f, bool trans_b = false);

// Enable/disable the transients pool (call AFTER weight loading; see Storage
// in kernels.cu — persistents must stay raw or they pin pool chunks at walls).
void set_alloc_pooling(bool enabled);

// Fused bottom-right-aligned causal softmax over the last dim of (...,L,S)
// scores (S >= L). Masked columns are never read; exact-zero like the eager
// -1e4-bias path. Inference-only at the Tensor level.
NDArray causal_softmax(const NDArray& scores);

// One-thread-per-ray Amanatides-Woo traversal over a resident uint8 voxel
// grid. Outputs are detached hit/material/voxel/face/distance arrays.
std::tuple<NDArray, NDArray, NDArray, NDArray, NDArray, NDArray> dda_raycast(
    const NDArray& grid, const NDArray& origins, const NDArray& directions,
    int max_steps);

// PAINT-CUDA-1 deterministic raster primitives.  Packed uint64 winner keys
// stay internal; public components use detached int64 NDArrays because uint64
// is not a public engine dtype.  Winner order is signed int32 depth, then lower
// positive one-based face ID.  Background is (INT32_MAX, 0).
std::tuple<NDArray, NDArray> raster_winner_scatter_min(
    const NDArray& pixel_indices, const NDArray& depth_keys,
    const NDArray& face_ids, int64_t pixel_count, int threads = 256);
std::tuple<NDArray, NDArray> raster_triangle_winners(
    const NDArray& clip_positions, const NDArray& faces, int height, int width,
    int face_threads = 256);
std::tuple<NDArray, NDArray> raster_winner_resolve(
    const NDArray& clip_positions, const NDArray& faces,
    const NDArray& winner_face_ids, int pixel_threads = 256);
std::tuple<NDArray, NDArray, NDArray> rasterize_clip(
    const NDArray& clip_positions, const NDArray& faces, int height, int width,
    int face_threads = 256, int pixel_threads = 256);

// PAINT-CUDA-2 bake primitives.  Back-project keeps one output row per input
// atlas sample; cosine blend consumes views in dimension-0 order.  Indexed
// atlas scatter remains outside this engine leg, so neither operation uses
// floating atomics or has a duplicate-index policy.
std::tuple<NDArray, NDArray, NDArray, NDArray> bake_back_project(
    const NDArray& atlas_positions_h, const NDArray& view,
    const NDArray& view_depth, const NDArray& view_reliable,
    const NDArray& view_cosine, const NDArray& world_to_camera,
    const NDArray& image_projection, float depth_threshold,
    int threads = 256);
std::tuple<NDArray, NDArray, NDArray> bake_cosine_blend(
    const NDArray& view_colors, const NDArray& view_cosine,
    const NDArray& view_valid, const NDArray& view_weights,
    const NDArray& view_enabled, int threads = 256);

// PAINT-CUDA-3 bounded ordered island executor. CSR construction, component
// labels, convergence/deadline checks, fallback, and receipts remain host-side.
// One island maps to one warp and lane zero preserves in-island occurrence and
// neighbor order. Returns cloned (colors, mask, final uncolored per island).
std::tuple<NDArray, NDArray, NDArray> inpaint_island_passes(
    const NDArray& positions, const NDArray& vertex_colors,
    const NDArray& vertex_mask, const NDArray& neighbor_offsets,
    const NDArray& neighbors, const NDArray& island_offsets,
    const NDArray& island_occurrences, int pass_count_cap,
    int threads = 128);

// Precomputed camera terms for the fused terrain renderer.  The Python
// wrapper derives these from the frozen Project-Scorch camera convention;
// this keeps the per-frame host work to a few scalar/vector calculations while
// the kernel still generates every per-pixel ray itself.
struct TerrainRenderCamera {
  float position[3];
  float forward[3];
  float right[3];
  float up[3];
  float half_width;
  float half_height;
  int width;
  int height;
};

// `direction` points in the light's travel direction.  Shading uses
// dot(normal, -direction), matching the frozen WO-7B contract.
struct TerrainRenderLight {
  float direction[3];
};

struct TerrainRenderConstants {
  int max_steps;
  // Smooth-mode-only density prefilter selector.  Zero is the literal WO-8A
  // path; levels 1 and 2 consume a separately cached u8 density field.
  int density_filter = 0;
  // WO-9B procedural surface detail selector.  Zero retains the literal
  // WO-7B/8A/8C kernel paths; one selects a separate detail-only launch.
  int detail = 0;
  // WO-9E bedrock cut-face and outside-domain horizon selector.  Zero keeps
  // the established render launches untouched; one adds a grounding pass.
  int grounding = 0;
  float z_horizon = 0.0f;
  float fog_start = 600.0f;
  float fog_full = 2400.0f;
};

// One axis-aligned voxel object composited into a terrain render.  Grid and
// palette storage stay GPU-resident; only this small descriptor is assembled
// on the host.  `origin` is the world-space minimum corner of local voxel
// (0, 0, 0).
struct TerrainRenderObject {
  NDArray grid;
  float origin[3];
  NDArray palette;
};

// Build the normalized u8 filtered-density cache for one source revision.
// Level 1 is a separable centered 3-tap box; level 2 is the best measured
// centered candidate from the r2 design rail, a 9-tap binomial/Gaussian.
NDArray terrain_filter_density(const NDArray& materials, int density_filter);

// One-thread-per-pixel terrain renderer.  Inputs are a resident uint8
// materials grid and a resident uint8 palette shaped (N, 3).  The outputs are
// detached RGB uint8 (H, W, 3) and depth float32 (H, W), with -1 on a miss.
std::tuple<NDArray, NDArray> terrain_render(
    const NDArray& materials, const TerrainRenderCamera& camera,
    const TerrainRenderLight& light, const NDArray& palette,
    const TerrainRenderConstants& constants,
    const NDArray* filtered_density = nullptr);

// Overlay non-empty axis-aligned object grids onto an already-rendered terrain
// image.  Per-pixel depth comparison preserves the nearer terrain/object hit.
void terrain_render_objects_overlay(
    NDArray& rgb, NDArray& depth, const TerrainRenderCamera& camera,
    const TerrainRenderLight& light,
    const std::vector<TerrainRenderObject>& objects, int max_steps);

// Fused RMSNorm over the last dim: out = x * rsqrt(mean(x^2) + eps) * w, fp32
// accumulate, single kernel + single output alloc (vs the 9-op chain).
// w must be fp32; out_dtype is typically x's dtype.
NDArray rms_norm(const NDArray& x, const NDArray& w, double eps, DType out_dtype);
NDArray rope_apply(const NDArray& x, const NDArray& cs, const NDArray& sn,
                   int64_t pos0, bool inverse = false,
                   bool pair_swap = false);
void write_rows(NDArray& buf, const NDArray& src, int64_t start);
NDArray export_rows(const NDArray& cache, int dim, int64_t start,
                    int64_t len);
NDArray export_rope_rows(const NDArray& cache, const NDArray& cs,
                         const NDArray& sn, int dim, int64_t start,
                         int64_t len, int64_t pos0, bool inverse = false,
                         bool pair_swap = false);
std::tuple<NDArray, NDArray> export_row_pair(
    const NDArray& raw_cache, const NDArray& rope_cache, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim, int64_t raw_start,
    int64_t rope_start, int64_t len, int64_t pos0, bool inverse = false,
    bool pair_swap = false);
std::tuple<std::vector<NDArray>, std::vector<NDArray>> export_row_pairs(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim,
    const std::vector<int64_t>& raw_starts,
    const std::vector<int64_t>& rope_starts, int64_t len, int64_t pos0,
    bool inverse = false, bool pair_swap = false);
std::tuple<std::vector<NDArray>, std::vector<NDArray>> swap_row_pairs_with_rope(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches,
    const std::vector<NDArray>& raw_inserts,
    const std::vector<NDArray>& rope_inserts, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim, int64_t head_tokens,
    int64_t tail_start, int64_t pos0, bool pair_swap = false);
std::tuple<std::vector<NDArray>, std::vector<NDArray>> evict_row_pairs(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches, int raw_dim, int rope_dim,
    int64_t head_tokens, int64_t drop_tokens);
std::tuple<std::vector<NDArray>, std::vector<NDArray>, int64_t>
arena_row_pair_transaction(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches,
    const std::vector<NDArray>& raw_inserts,
    const std::vector<NDArray>& rope_inserts, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim, int64_t sink_tokens,
    int64_t current_mount_tokens, int64_t arena_width, bool pair_swap = false);
NDArray splice_rows(const NDArray& old_cache, const NDArray& insert,
                    int dim, int64_t head_tokens, int64_t tail_start);
NDArray evict_rows(const NDArray& old_cache, int dim, int64_t head_tokens,
                   int64_t drop_tokens);

// INT4 group-quantized linear: y = x @ dequant(W)^T.
//   x       : (..., K) fp16/fp32 activations (K == in_features)
//   packed  : (N, K/2) uint8 — two 4-bit weights per byte, even=low nibble,
//             odd=high nibble; N == out_features
//   scales  : (N, K/group_size) fp16 per-group scale
//   zeros   : (N, K/group_size) fp16 per-group zero point (min)
// Dequant rule mirrors the reference QuantizedLinear: w = q*scale + zero, with
// q the 4-bit code in [0,15]. Output dtype = x.dtype, shape (..., N).
NDArray int4_dequant(const NDArray& packed, const NDArray& scales,
                     const NDArray& zeros, int group_size, DType out_dtype);
NDArray int4_linear(const NDArray& x, const NDArray& packed,
                    const NDArray& scales, const NDArray& zeros, int group_size);
// Fused dequant-GEMM variant: same result as int4_linear but dequantizes the
// int4 weight into shared-memory tiles inside the GEMM, avoiding the full (K,N)
// fp16 weight transient. Opt-in (a custom GEMM may lose to cuBLAS at large N).
NDArray int4_linear_fused(const NDArray& x, const NDArray& packed,
                          const NDArray& scales, const NDArray& zeros, int group_size);

// Trinity-compatible symmetric INT8 weight / FP16 activation GEMM.
//   x      : (..., K) fp16; leading dimensions are flattened then restored
//   codes  : (N, K) uint8 with signed q represented as code = q + 128
//   scales : (N, K/32) fp16; w[n,k] = fp16((code-128) * scale[n,k/32])
// K must be positive and divisible by 32. The fixed [N,K] weight broadcasts
// over every leading activation dimension. launch_config 0 is an m64n16 CTA;
// launch_config 1 is m16n64. Both stage dequantized FP16 weight tiles in
// shared memory and accumulate in FP32 without a full-weight transient.
NDArray w8a16_matmul(const NDArray& x, const NDArray& codes,
                     const NDArray& scales, int launch_config);

// Packed low-bit weight path for experimental INT2/INT3 loaders.
//   packed : (N, ceil(K*bits/8)) uint8, little-endian bit stream per row
//   bits   : 2 or 3 for this first native path
//   K      : in_features is explicit because 3-bit rows carry byte padding
//   scales/zeros : (N, K/group_size) fp16, or empty zeros for symmetric grid
// Dequant returns a transposed (K, N) matrix so matmul(x, W_kn) computes
// y = x @ dequant(W)^T. Empty zeros selects q - 2^(bits-1).
NDArray intn_dequant(const NDArray& packed, const NDArray& scales,
                     const NDArray& zeros, int bits, int64_t in_features,
                     int group_size, DType out_dtype);
NDArray intn_linear(const NDArray& x, const NDArray& packed,
                    const NDArray& scales, const NDArray& zeros, int bits,
                    int64_t in_features, int group_size);
NDArray intn_linear_fused(const NDArray& x, const NDArray& packed,
                          const NDArray& scales, const NDArray& zeros,
                          int bits, int64_t in_features, int group_size);

// GPT-OSS MXFP4 expert linear.
//   x      : (..., K) fp32/fp16/bf16 activations
//   blocks : (N, G, 16) uint8; each 16-byte group packs 32 FP4 values
//            as low/high nibbles, matching HF GPT-OSS safetensors
//   scales : (N, G) uint8; E8M0 exponent with value scale = 2^(scale - 127)
//   K      : G * 32; output shape is (..., N), dtype = x.dtype
// Inference/frozen-weight path; does not materialize the dequantized (K,N)
// expert matrix.
NDArray mxfp4_linear(const NDArray& x, const NDArray& blocks,
                     const NDArray& scales);
NDArray mxfp4_linear_expert(const NDArray& x, const NDArray& blocks,
                            const NDArray& scales, int64_t expert_idx);

// KV-cache INT4 storage (D-grouped, symmetric-8). Distinct from int4_dequant
// (weight-shaped, K-grouped, transposed-matrix output): packs along the
// innermost D of a (B,KV,S,D) cache and reads come out as a (B,KV,n,D) SLICE.
//   kv_int4_pack  : x (B,KV,S,D) compute -> packed (B,KV,S,D/2) uint8; writes
//                   the per-group scales into scales_out (B,KV,S,D/group).
//   kv_int4_unpack: dequant rows [lo,lo+n) -> (B,KV,n,D) out_dtype.
// group must divide D; D even. Symmetric-8 grid (q in [0,15], x=(q-8)*scale).
NDArray kv_int4_pack(const NDArray& x, NDArray& scales_out, int group);
NDArray kv_int4_unpack(const NDArray& packed, const NDArray& scales,
                       int group, int64_t lo, int64_t n, DType out_dtype);

// Fused Gated DeltaNet decode step (inference-only, all fp32,
// FUNCTIONAL — returns {out, new_state}; the input state is untouched
// so callers may branch/save freely): folds l2norm(q,k), gate math and
// the decay-first delta-rule update + readout into one launch.
// q,k (B,Hk,Dk) raw heads; v (B,H,Dv); a,b (B,H); A_neg/dt_bias H elems.
std::pair<NDArray, NDArray> gated_delta_step(
    const NDArray& q, const NDArray& k, const NDArray& v,
    const NDArray& a, const NDArray& b, const NDArray& A_neg,
    const NDArray& dt_bias, const NDArray& state);

// Fused sparse APA-Quant attention. q,k,kq,v: (B,H,L,D)/(B,H,S,D). For each
// query row, the refine threshold is built from the quantized (bulk) scores;
// the full-precision dot is computed ONLY for keys whose |bulk| >= threshold
// (mean+zthr*std), the rest keep their quantized score. Online softmax over the
// resulting scores. Never materializes the L x S score matrix. Inference only.
NDArray apa_selective_attention(const NDArray& q, const NDArray& k,
                                const NDArray& kq, const NDArray& v,
                                float scale, float zthr, bool is_causal);
NDArray apa_selective_attention_sink(const NDArray& q, const NDArray& k,
                                     const NDArray& kq, const NDArray& v,
                                     const NDArray& sinks, float scale,
                                     float zthr, bool is_causal);

// Flash-style non-causal scaled dot-product attention. q/k/v are contiguous
// (B,H,L,D), share dtype/device/head geometry, and D is at most 128. The
// implementation streams keys through an online softmax and never allocates
// a score matrix. Inference-only through the Tensor-level wrapper.
NDArray fused_sdpa_noncausal(const NDArray& q, const NDArray& k,
                             const NDArray& v, float scale);

// Fused non-causal APA attention with packed symmetric INT4 bulk K
// (EXP-APA-2). q/k/v are contiguous (B,H,L,D), same dtype (fp16/fp32),
// even D <= 128. Streams keys twice inside one kernel (bulk stats -> blend
// + online softmax); never allocates an Lq x Lk score tensor. zthr is the
// Gaussian-quantile z = Phi^-1(1-r); refine_all short-circuits to exact
// streaming SDPA (the r >= 1 case). Inference-only via the Tensor wrapper.
NDArray apa_int4_sdpa_noncausal(const NDArray& q, const NDArray& k,
                                const NDArray& v, float scale, float zthr,
                                bool refine_all);

// EXP-APA-4 (K2) realized-refine-fraction instrumentation. When TC_APA_FRAC=1
// the Q-tile APA kernel (r < 1) accumulates the number of refined (row,key)
// pairs into a process-global device counter and the launcher accumulates the
// total (row,key) pairs host-side. Returns {refined, total} since the last
// reset (both 0 when the instrumentation never engaged); reset=true clears
// both after reading. The legacy streaming path (TC_ATTN_QTILE=0) is the
// frozen EXP-APA-2 instrument and is deliberately NOT instrumented.
std::pair<unsigned long long, unsigned long long> apa_refine_stats(bool reset);

// APA selective TRAINING forward: O(L)-memory (never materializes the L x L
// score matrix), additionally saves per-row logsumexp + threshold for the
// backward. Returns (out, lse, thr). Selection is a stop-gradient.
std::tuple<NDArray, NDArray, NDArray> apa_selective_fwd_train(
    const NDArray& q, const NDArray& k, const NDArray& kq, const NDArray& v,
    float scale, float zthr, bool is_causal);
// APA selective backward: recomputes scores from the saved (lse, thr), streams
// dQ/dK/dV in O(L) memory. dK is nonzero on SELECTED keys only (kq detached).
// Returns (dq, dk, dv).
std::tuple<NDArray, NDArray, NDArray> apa_selective_bwd(
    const NDArray& q, const NDArray& k, const NDArray& kq, const NDArray& v,
    const NDArray& dO, const NDArray& lse, const NDArray& thr,
    float scale, bool is_causal);

// Fused APA blend+softmax over precomputed bulk/rank score matrices (each
// (..., S), from cuBLAS): per row, thr = mean(|bulk|)+zthr*std(|bulk|); score
// = |bulk|>=thr ? rank : bulk; returns softmax(score) weights. Replaces the
// abs/mean/std/where/softmax op chain with a single launch.
//
// Two bounds conventions (Phase 3.1, board item 4a — kills the O(S^2)
// additive-mask materialize+add on both bulk and rank):
//   Lq <= 0 (row_smax arg unused, pass nullptr): legacy sentinel path. Caller
//     has baked causal/window masking into bulk/rank as -1e4 bias; masked
//     keys are detected in-kernel via a MASK_LIM sentinel check.
//   Lq > 0: index-arithmetic path. No mask tensor read. row0 = absolute
//     query-chunk start (tiled callers slice queries into blocks); Lq = FULL
//     query length L (not the chunk length — the chunk length is recovered
//     internally from bulk's own shape); window <= 0 = full causal (no
//     sliding), else sliding window of that width. Bottom-right causal
//     convention matches functional.py's _causal_mask exactly (row i sees
//     keys 0..(S-Lq)+row0+i inclusive); sliding window matches
//     gpt_oss20b_tc.py's _gpt_oss_attention_mask (keys (q_abs-window, q_abs]).
NDArray apa_blend_softmax(const NDArray& bulk, const NDArray& rank,
                          float zthr, const NDArray* row_smax,
                          int Lq = 0, int64_t row0 = 0, int window = 0);
NDArray apa_blend_softmax_sink(const NDArray& bulk, const NDArray& rank,
                               const NDArray& sinks, float zthr,
                               int Lq = 0, int64_t row0 = 0, int window = 0);

// Fill / compare helpers.
NDArray ge_scalar(const NDArray& a, double s);  // (a >= s) as same dtype 0/1

// APA quantization: per-element searchsorted against `boundaries` (sorted)
// followed by a gather from `codebook` (len = boundaries+1). Same dtype out.
NDArray apa_quantize_gather(const NDArray& rotated, const NDArray& boundaries,
                            const NDArray& codebook);

// Conv helpers (NCHW). im2col -> (N, C*kh*kw, OH*OW); col2im scatter-adds back.
NDArray im2col(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw);
NDArray col2im(const NDArray& cols, const Shape& x_shape, int kh, int kw,
               int sh, int sw, int ph, int pw);
// Pooling (NCHW). maxpool writes the flat argmax (int64) for backward.
NDArray avgpool2d(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw);
NDArray avgpool2d_bwd(const NDArray& g, const Shape& x_shape, int kh, int kw,
                      int sh, int sw, int ph, int pw);
NDArray maxpool2d(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw,
                  NDArray& argmax_out);
NDArray maxpool2d_bwd(const NDArray& g, const NDArray& argmax, const Shape& x_shape);

// Embedding: gather rows of `weight` (V, ...) by int64 `idx` (any shape).
// Output shape = idx.shape ++ weight.shape[1:].
NDArray embedding_forward(const NDArray& weight, const NDArray& idx);
// Scatter-add `grad` back into a zeroed weight-shaped tensor (accumulated fp32).
NDArray embedding_backward(const NDArray& grad, const NDArray& idx,
                           const Shape& weight_shape, DType weight_dtype);

// In-place optimizer steps (compute in fp32, store in param dtype).
void sgd_step(NDArray& param, const NDArray& grad, NDArray& momentum_buf,
              double lr, double momentum, double weight_decay);
void adam_step(NDArray& param, const NDArray& grad, NDArray& m, NDArray& v,
               double lr, double b1, double b2, double eps, int64_t t,
               double weight_decay, bool decoupled);
void lion_step(NDArray& param, const NDArray& grad, NDArray& m,
               double lr, double b1, double b2, double weight_decay);
void radam_step(NDArray& param, const NDArray& grad, NDArray& m, NDArray& v,
                double lr, double b1, double b2, double eps, double bc1, double bc2,
                double rect, bool rectified, double weight_decay, bool decoupled);
void rmsprop_step(NDArray& param, const NDArray& grad, NDArray& sq_avg,
                  double lr, double alpha, double eps, double weight_decay);
void adagrad_step(NDArray& param, const NDArray& grad, NDArray& acc,
                  double lr, double eps, double weight_decay);
// param.data += alpha * other  (in place); param.data *= s (in place).
void axpy_(NDArray& param, const NDArray& other, double alpha);
void scale_(NDArray& param, double s);

// CUDA bookkeeping.
void cuda_sync();
void cuda_check_last(const char* where);

// Release all device blocks held idle by the caching allocator back to the
// driver. Live tensors are unaffected. Call under memory pressure or to measure
// true steady-state usage.
void empty_cache();

}  // namespace tc
