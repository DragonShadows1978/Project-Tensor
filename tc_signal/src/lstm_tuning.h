// tc_signal LSTM tunables (P1T-M3). House tunable law: no launch-shape literals at the use sites; every LSTM knob
// lives here (M2's tuning.h is shared and left untouched). None of these changes numerics: every output element of every
// kernel is produced by exactly one thread in a fixed order (grid-stride loops), so a knob changes speed only.
//
// Prior art: none needed for a constants module (no prior art known to me).
#pragma once

namespace tcs {
namespace lstm_tuning {

// Threads per block for the fused LSTM cell kernel (one thread per (direction, batch, hidden unit)).
constexpr int kCellBlock = 256;
// Threads per block for the kernel that pre-fills the projected-input buffer with Wb (one thread per element).
constexpr int kBiasFillBlock = 256;
// Upper bound on blocks per launch; the kernels are grid-stride loops, so larger problems loop instead.
constexpr long long kGridCap = 1LL << 20;

}  // namespace lstm_tuning
}  // namespace tcs
