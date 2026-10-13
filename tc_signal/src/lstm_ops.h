// Host launcher for tc_signal's ONNX-semantics LSTM (P1T-M3, T4). Raw device pointers in; everything is queued on the
// legacy default stream; no device memory is allocated here (the projected-input and gate scratch is one operand the
// Python layer allocates; the recurrent state lives in the Y_h / Y_c outputs). cuBLAS's own internal workspace for the
// LSTM handle is the one exception (see lstm.cu, "cuBLAS handle").
//
// Prior art: see lstm.cu (ONNX LSTM operator spec; Hochreiter & Schmidhuber 1997; Gers, Schmidhuber & Cummins 2000;
// input projection hoisted out of the recurrence as in Appleyard, Kocisky & Blunsom 2016 / cuDNN RNN).
#pragma once

#include <cstdint>

namespace tcs {

// Gate blocks per direction along the 4H axis of W, R, Wb and Rb (ONNX LSTM: i, o, f, c).
constexpr int64_t kLstmGates = 4;
// B is [D, 2 * kLstmGates * H] = [Wb; Rb] per direction.
constexpr int64_t kLstmBiasHalves = 2;

struct LstmGeom {
  int64_t T, N, I;         // x [T, N, I] (ONNX layout 0)
  int64_t H;               // hidden_size
  int64_t D;               // direction slots: 1 (forward or reverse) or 2 (bidirectional)
  int64_t first_reverse;   // slots d >= first_reverse run backwards in time (forward: 1, reverse: 0, bidirectional: 1)
};

// Scratch layout (one fp32 operand): G_x [D, T, N, 4H] (projected inputs, Wb included), then G_h [D, N, 4H] (the
// per-step recurrent product h_{t-1} . R^T).
inline int64_t lstm_gx_numel(const LstmGeom& g) { return g.D * g.T * g.N * kLstmGates * g.H; }
inline int64_t lstm_gh_numel(const LstmGeom& g) { return g.D * g.N * kLstmGates * g.H; }
inline int64_t lstm_scratch_numel(const LstmGeom& g) { return lstm_gx_numel(g) + lstm_gh_numel(g); }

// Y [T, D, N, H], Y_h / Y_c [D, N, H]. B, h0 and c0 may be NULL (zeros). Y_h / Y_c double as the recurrent state.
void lstm_forward(int device, const LstmGeom& g, const float* x, const float* W, const float* R, const float* B,
                  const float* h0, const float* c0, float* Y, float* Y_h, float* Y_c, float* scratch);

struct LstmCublasConfig {
  int math_mode;           // cublasMath_t
  int atomics_mode;        // cublasAtomicsMode_t
  int pointer_mode;        // cublasPointerMode_t
  bool legacy_stream;      // the handle's stream is 0 (the legacy default stream)
  bool default_math;       // math_mode == CUBLAS_DEFAULT_MATH
  bool atomics_not_allowed;
};
// The (lazily created) LSTM cuBLAS handle's settings on `device`.
LstmCublasConfig lstm_cublas_config(int device);

}  // namespace tcs
