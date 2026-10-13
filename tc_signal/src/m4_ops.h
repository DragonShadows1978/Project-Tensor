// P1T-M4 host launchers (T5 norms, T6 small ops) and their tunables. Raw device pointers in, everything queued on the
// legacy default stream; nothing here allocates device memory (outputs come from the Python layer's allocator).
//
// House tunable law: every launch-shape knob of the M4 kernels lives here. None of them changes numerics except
// kNormBlock and kNormChunk, which fix the summation order of the norm statistics (see norms.cu); changing either one
// changes the last bits of the statistics (still fp32, still deterministic) and must be re-gated.
#pragma once

#include <cstdint>

namespace tcs {
namespace m4tuning {

// Threads per block of the row-norm kernel = the width of its reduction tree (one block per row). Power of two.
constexpr int kNormBlock = 256;
// Blocked summation inside a thread: a thread's strided terms are summed kNormChunk at a time into a partial sum, which
// is then added to the thread total (error bound (kNormChunk + n/kNormChunk) u instead of n u for n terms).
constexpr int kNormChunk = 32;
// Threads per block of the gather kernels (upsample, reflection pad) and of rint (one thread per output element).
constexpr int kCopyBlock = 256;

static_assert(kNormBlock > 0 && (kNormBlock & (kNormBlock - 1)) == 0, "kNormBlock must be a power of two");
static_assert(kNormChunk > 0, "kNormChunk must be positive");

}  // namespace m4tuning

// Where a row-norm output element finds its scale/bias entry.
enum class NormAffine : int {
  kPerRow = 0,      // instance norm: rows are (n, c); entry c = row % channels
  kPerElem = 1,     // layer norm: entry j (the element's index inside the normalized block)
  kPerChannel = 2,  // group norm: rows are (n, g); entry (row % groups) * ch_per_group + j / T
};

// One normalization problem: `rows` contiguous rows of length L, each normalized over itself.
struct RowNormGeom {
  int64_t rows;
  int64_t L;
  float eps;
  NormAffine affine;
  int64_t channels;      // kPerRow: C
  int64_t groups;        // kPerChannel: G
  int64_t ch_per_group;  // kPerChannel: C / G
  int64_t T;             // kPerChannel: time length (elements per channel)
};

// y = ((x - K) - d) * rstd * scale + bias per row, fp32 shifted two-pass statistics (norms.cu). scale/bias may be NULL.
void row_norm_forward(int device, const RowNormGeom& g, const float* x, const float* scale, const float* bias,
                      float* y);

// y[r, i] = x[r, i / scale] for rows r of length T, output length T * scale (bit copy).
void upsample_nearest1d_forward(int device, int64_t rows, int64_t T, int64_t scale, const float* x, float* y);

// ONNX/torch reflect pad of the last axis by (pad_begin, pad_end), each < T (bit copy).
void reflection_pad1d_forward(int device, int64_t rows, int64_t T, int64_t pad_begin, int64_t pad_end, const float* x,
                              float* y);

// y = rintf(x) elementwise (IEEE 754 roundTiesToEven, sign of zero kept, NaN and +-Inf pass through).
void rint_forward(int device, int64_t n, const float* x, float* y);

}  // namespace tcs
