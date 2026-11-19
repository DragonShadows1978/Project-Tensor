// APA_EB4_BEGIN tuning
// APA-EB4 tuning knobs (Tunable law: every scheduling literal of the EB4 decode path lives here; none changes an fp32
// operation on the output or on the tk8 decision). Prior art: NVIDIA CUDA C++ Programming Guide (occupancy, L2
// prefetch); CUTLASS-style named tile/stage constants. Ours: the values, chosen by EB4 development timing (ledger,
// "Development"). EB3's knobs (eb3_tuning.cuh) are reused unchanged for every kernel EB4 shares with EB3.
#pragma once
namespace eb4 {
namespace tune {
// Bulk-kernel cross-kernel L2 prefetch of select's V operand (as EB3's XPF), budgeted against L2_FRACTION of L2:
//   0 none, 1 bf16 V of the chunk (EB3's choice), 2 int8 V + fp32 scales of the chunk (the representation most
//   blocks read under tk8; flags are not known yet when the bulk kernel runs).
#ifdef APA_EB4_DEV_BULK_XPF   // development builds only (ledger "Development"); never set in a registered build
constexpr int BULK_XPF = APA_EB4_DEV_BULK_XPF;
#else
constexpr int BULK_XPF = 2;
#endif
// select CTA-start prefetch of its chunk's V, per 64-key block, after reading the block flags:
//   0 none, 1 bf16 rows for bf16 blocks and int8 rows + scales for int8 blocks, 2 bf16 rows for bf16 blocks only.
#ifdef APA_EB4_DEV_SEL_VPF    // development builds only
constexpr int SEL_VPF = APA_EB4_DEV_SEL_VPF;
#else
constexpr int SEL_VPF = 1;
#endif
// With SEL_VPF 1 and the bulk's int8 prefetch active (BULK_XPF 2 within budget), select skips the int8 rows the
// bulk already prefetched and prefetches only bf16 blocks (1) or still prefetches both (0).
#ifdef APA_EB4_DEV_SEL_VPF_SKIP_XPF  // development builds only
constexpr bool SEL_VPF_SKIP_XPF = APA_EB4_DEV_SEL_VPF_SKIP_XPF;
#else
constexpr bool SEL_VPF_SKIP_XPF = true;
#endif
// Where the tk8 choose runs (both exact, same flags bitwise; G1.2 checks against EB2b):
//   0 the last bulk CTA of each (batch, KV head) (threadfence ticket; Batcher bitonic sort as EB2b);
//   1 every select CTA, redundantly, before S1 (stable descending rank of its own 4 blocks by counting).
#ifdef APA_EB4_DEV_CHOOSE_SITE  // development builds only
constexpr int CHOOSE_SITE = APA_EB4_DEV_CHOOSE_SITE;
#else
constexpr int CHOOSE_SITE = 1;
#endif
// Fraction of the device L2 the cross-kernel prefetch may use (EB3 uses 0.5 for its own budget).
constexpr double L2_FRACTION = 0.5;
// Persistent choose-ticket buffer (one unsigned per (batch, KV head)), minimum entries on first use.
constexpr int COUNTER_MIN_ENTRIES = 1024;
// Largest sort width of the in-kernel tk8 choose (S <= 64 * NSORT_MAX = 65536).
constexpr int NSORT_MAX = 1024;
} // namespace tune
} // namespace eb4
// APA_EB4_END tuning
