"""BWD-F4 H1 / L55: a standalone REPLICA of mean's count (Build F3's loop and Build F4's rule, copied from
tensor_cuda/src/ops.cpp at 2117296ea and in this tree), compiled with -fsanitize=signed-integer-overflow and run on the
shapes the order names. It is not the module: it shows (1) the count Build F3's loop computes on the reviewer's shape and
where the signed overflow occurs (UBSan's "runtime error: signed integer overflow"), (2) Build F4's rule on the same
shapes (no overflow; the refusal point for a non-empty selected product), and (3) over a grid of shapes and axis sets,
that wherever the exact product fits in int64, Build F3's wrapped count and Build F4's count are EQUAL (the order's
"result unchanged for every input that does not overflow"), while counting the grid points where Build F3's loop
overflowed on the way (the undefined-behaviour cases) and where Build F4 refuses.
Evidence class: replica + exhaustive grid (reasoning about the rule); the module's own behaviour is G1's rows.

  python3 -B -P scripts/bwdf4_count_replica.py <out log>

Prior art: GCC's __builtin_mul_overflow (2015); UBSan's signed-integer-overflow check (Clang/GCC); NumPy's empty-axis
count 0. Ours: the replica and the grid.
"""
import os
import subprocess
import sys

CPP = r'''
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <climits>
using Shape = std::vector<int64_t>;
// Build F3's loop (ops.cpp:1064): cnt *= a.shape()[d], signed, in axis order.
static int64_t f3_count(const Shape& s, const std::vector<int>& nax) {
  int64_t cnt = 1;
  for (int d : nax) cnt *= s[d];
  return cnt;
}
// Build F4's rule (ops.cpp, BWD-F4 H1): emptiness first, then a checked product; returns false = refused.
static bool f4_count(const Shape& s, const std::vector<int>& nax, int64_t* out) {
  int64_t cnt = 1;
  bool selected_empty = false;
  for (int d : nax) selected_empty = selected_empty || s[d] == 0;
  if (selected_empty) { *out = 0; return true; }
  for (int d : nax)
    if (__builtin_mul_overflow(cnt, s[d], &cnt)) return false;
  *out = cnt;
  return true;
}
static std::string txt(const Shape& s) { std::string t = "["; for (size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]); return t + "]"; }
int main() {
  struct Case { const char* name; Shape s; std::vector<int> nax; };
  std::vector<Case> cases = {
    {"reviewer [2^32,2^32,0,0] mean([0,1,2])", {1LL << 32, 1LL << 32, 0, 0}, {0, 1, 2}},
    {"non-empty axes [2^32,2^32,1,0] mean([0,1])", {1LL << 32, 1LL << 32, 1, 0}, {0, 1}},
    {"triple [2^21,2^21,2^21,0] mean([0,1,2])", {1LL << 21, 1LL << 21, 1LL << 21, 0}, {0, 1, 2}},
    {"fits [2^31,2^31,1,0] mean([0,1])", {1LL << 31, 1LL << 31, 1, 0}, {0, 1}},
  };
  for (auto& c : cases) {
    fprintf(stderr, "--- F3 loop on %s\n", c.name);   // UBSan prints to stderr
    int64_t f3 = f3_count(c.s, c.nax);
    int64_t f4 = -1; bool ok = f4_count(c.s, c.nax, &f4);
    printf("%s | shape %s | F3 count %lld | F4 %s %lld\n", c.name, txt(c.s).c_str(), (long long)f3, ok ? "count" : "REFUSES (before any work)", (long long)f4);
  }
  // Grid: every rank 1..4 shape over these extents, every non-empty set of distinct axes (in increasing order).
  const int64_t E[] = {0, 1, 2, 3, 1LL << 20, 1LL << 31, 1LL << 32, 1LL << 62};
  const int NE = sizeof(E) / sizeof(E[0]);
  long long total = 0, fit = 0, equal = 0, unequal = 0, f3_intermediate_overflow = 0, f4_refuse = 0, f4_refuse_wrong = 0;
  for (int rank = 1; rank <= 4; ++rank) {
    long long nshape = 1; for (int i = 0; i < rank; ++i) nshape *= NE;
    for (long long code = 0; code < nshape; ++code) {
      Shape s(rank); long long c = code; for (int i = 0; i < rank; ++i) { s[i] = E[c % NE]; c /= NE; }
      for (int mask = 1; mask < (1 << rank); ++mask) {
        std::vector<int> nax; for (int i = 0; i < rank; ++i) if (mask >> i & 1) nax.push_back(i);
        ++total;
        // exact product in 128 bits (zero if any extent is zero), and whether F3's signed prefix overflowed
        __int128 ex = 1; bool zero = false; for (int d : nax) { if (s[d] == 0) zero = true; }
        if (zero) ex = 0; else for (int d : nax) { if (ex <= ((__int128)1 << 64)) ex *= s[d]; }  /* saturates above 2^64: no 128-bit overflow */
        bool fits = ex <= (__int128)INT64_MAX;
        bool prefix_ovf = false; { int64_t t = 1; for (int d : nax) { if (__builtin_mul_overflow(t, s[d], &t)) { prefix_ovf = true; break; } } }
        int64_t f4 = -1; bool ok = f4_count(s, nax, &f4);
        // F3's count with the wrap made defined (unsigned arithmetic) -- the value the signed loop gives on this target
        uint64_t u = 1; for (int d : nax) u *= (uint64_t)s[d];
        int64_t f3 = (int64_t)u;
        if (fits) {
          ++fit;
          if (prefix_ovf) ++f3_intermediate_overflow;
          if (ok && f4 == f3 && (__int128)f4 == ex) ++equal; else ++unequal;
        } else {
          ++f4_refuse;
          if (ok) ++f4_refuse_wrong;
        }
      }
    }
  }
  printf("grid: %lld (shape, axes) points; exact product fits int64: %lld -> F3-wrapped == F4 == exact in %lld, unequal in %lld; "
         "of the fitting points, %lld had a signed overflow in F3's loop on the way (a later 0 extent: the F01 undefined-behaviour case); "
         "exact product overflows int64: %lld -> F4 accepted wrongly in %lld\n",
         total, fit, equal, unequal, f3_intermediate_overflow, f4_refuse, f4_refuse_wrong);
  return (unequal == 0 && f4_refuse_wrong == 0) ? 0 : 1;
}
'''
out = sys.argv[1]
d = "/tmp/bwdf4_replica"
os.makedirs(d, exist_ok=True)
src, exe = os.path.join(d, "count_replica.cpp"), os.path.join(d, "count_replica")
open(src, "w").write(CPP)
cc = subprocess.run(["g++", "-std=c++17", "-O1", "-fsanitize=signed-integer-overflow", "-o", exe, src], capture_output=True,
                    text=True, timeout=300)
if cc.returncode:
    raise SystemExit(cc.stderr)
r = subprocess.run([exe], capture_output=True, text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
text = "STDOUT\n" + r.stdout + "\nSTDERR (UBSan)\n" + r.stderr + f"\nexit {r.returncode}\n"
open(out, "w").write(text)
print(text)
sys.exit(r.returncode)
