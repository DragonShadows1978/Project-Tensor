// APA_EB3_BEGIN tuning
// APA-EB3 tuning knobs (Tunable law: every scheduling/resource literal of the
// EB3 decode path lives here; none of them changes an fp32 operation).
// Prior art: CUDA occupancy and __launch_bounds__ (NVIDIA CUDA C++
// Programming Guide); CUTLASS-style tile/stage constants as named knobs.
// Ours: the specific values, chosen by EB3 development timing (ledger,
// 'Development' section; dev builds dev1-dev3).
#pragma once
namespace eb3 {
namespace tune {
// Query rotation: output dims per CTA. Grid = (B*KVH, D/ROT_DB). Each CTA
// stages ROT_DB rows of R[kv] (ROT_DB*D fp32) once for all G query heads.
constexpr int ROT_DB = 16;
// Minimum resident CTAs per SM requested from ptxas (__launch_bounds__).
// 256 CTAs (S=8192, 8 KV heads) on 56 SMs need >= 5 per SM for one wave.
// Development (ledger): bulk is tensor-issue bound; MINB 4/5 on the copied
// kernel was slower (19.0/24.4 us vs 17.8 us), so the swap-AB bulk keeps 1.
constexpr int BULK_MINB = 1;
constexpr int BULK_MINB_ALT = 5;      // resource census only (dev: 68.6 vs 65.5 us, not adopted)
constexpr int SEL_MINB = 5;
// Fused-merge load batching (independent L2 loads issued before each
// ordered fp32 chain; the chain order itself is fixed).
constexpr int MERGE_BATCH = 16;
// Cross-kernel L2 prefetch budget: fraction of the device L2 that the
// prefetched bulk codes+norms (and, for the bulk V prefetch, plus V) may use.
constexpr double L2_FRACTION = 0.5;
// Persistent merge-ticket buffer: minimum entries allocated on first use.
constexpr int COUNTER_MIN_ENTRIES = 1024;
// Registered path (variant 0), ids as in eb3.cuh VARIANTS:
constexpr int REG_ROT = 1;        // eb3::rotate_q<128,ROT_DB>
constexpr int REG_RPF = 1;       // rotation prefetch: 1 codes+norms, 2 codes+norms+V
constexpr int REG_BULK = 2;       // swap-AB bulk tile
constexpr bool REG_XPF = true;    // bulk prefetches select_pv's V (budgeted)
constexpr bool REG_VPF = true;    // select_pv prefetches its chunk's V
constexpr bool REG_FUSE = true;   // merge fused into select_pv
constexpr bool REG_KPF = false;   // select_pv prefetches refined-union K rows
// Registered (gated, timed) configuration of the EB3 path. Variant ids are
// the runtime `variant` argument of apa_eb3_run; 0 is the registered path,
// every other id is a descriptive ablation (never gated as the EB3 result).
constexpr int REGISTERED_VARIANT = 0;
} // namespace tune
} // namespace eb3
// APA_EB3_END tuning
