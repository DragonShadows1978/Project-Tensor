// Checked int64 shape / geometry / byte arithmetic for tc_signal (P1T-FIX1, review finding F03).
//
// Every plan function and every operand validation computes element counts, output lengths, scratch sizes and byte
// totals through these helpers BEFORE any narrowing cast, allocation or launch. An overflow raises ValueError naming
// the quantity; a wrapped value is never used (in particular it never turns into an "empty" or negative extent).
// The byte ceiling is INT64_MAX (= PTRDIFF_MAX on LP64): every byte total and every pointer offset must stay a valid
// signed pointer difference.
//
// Prior art: GCC/Clang `__builtin_mul_overflow` / `__builtin_add_overflow` / `__builtin_sub_overflow` (GCC 5, 2015),
// which this file uses; C23 `ckd_mul` / `ckd_add` / `ckd_sub` (<stdckdint.h>, ISO/IEC 9899:2024), the standardized
// form of the same builtins; the SEI CERT C rules INT30-C / INT32-C ("ensure that operations on (un)signed integers do
// not wrap / overflow"). Taken: the checked-operation primitive. Ours: where tc_signal applies it (the list of
// quantities in the plan functions) and the INT64_MAX byte ceiling. Unverified in this sandbox (lead to check):
// "GCC 5 __builtin_mul_overflow release", "C23 stdckdint ckd_mul", "CERT INT32-C".
#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tcs {

[[noreturn]] inline void overflow_refusal(const char* what) {
  throw pybind11::value_error(std::string("tc_signal: ") + what +
                              " overflows 64-bit arithmetic; refused (no wrapped size is ever used)");
}

inline int64_t ck_mul(int64_t a, int64_t b, const char* what) {
  int64_t r = 0;
  if (__builtin_mul_overflow(a, b, &r)) overflow_refusal(what);
  return r;
}

inline int64_t ck_add(int64_t a, int64_t b, const char* what) {
  int64_t r = 0;
  if (__builtin_add_overflow(a, b, &r)) overflow_refusal(what);
  return r;
}

inline int64_t ck_sub(int64_t a, int64_t b, const char* what) {
  int64_t r = 0;
  if (__builtin_sub_overflow(a, b, &r)) overflow_refusal(what);
  return r;
}

// P1T-FIX2, review finding N2 (the zero-extent rule). A zero extent makes the element count 0, but it does NOT make
// the other extents harmless: strides (suffix products), partial counts (prefix products) and byte strides are formed
// from them whatever the count is, and the round-2 review reproduced signed overflow in exactly those products (UBSan,
// core numel_of/contiguous_strides). So every product below multiplies EVERY extent, a zero counted as 1 (the extent
// bound), and refuses the shape if that bound (or its fp32 byte size) overflows int64, even when the count is 0.
// Callers have refused negative extents; a negative factor is also counted as 1 here (never multiplied).
// Prior art: as above (checked builtins; CERT INT32-C). The same rule (itemsize times every NONZERO extent must fit,
// even when the count is 0) is NumPy's array constructor (PyArray_NewFromDescr skips zero dimensions and multiplies
// the rest into the byte size with overflow checks): measured here, numpy 2.4.2 np.empty((0, 2**62, 4), float32) and
// np.empty((0, 2**61), float32) raise "array is too big". PyTorch 2.11 refuses the element-level part ("Stride
// calculation overflowed" for (0, 2**62, 4), "Storage size calculation overflowed" for (2**62, 4, 0)) but not the byte
// part (it accepts (0, 2**61) fp32). Taken: NumPy's rule. Ours: applying it to every tc_signal plan and validation.
// Source-level citation unverified here (lead to check): "numpy ctors.c PyArray_NewFromDescr_int is_zero dimensions
// overflow", "c10 computeStorageNbytesContiguous safe_multiplies_u64".
inline int64_t ck_extent_bound(const std::vector<int64_t>& s, size_t a, size_t b, const char* what) {
  int64_t p = 1;
  for (size_t i = a; i < b; ++i) p = ck_mul(p, s[i] > 1 ? s[i] : 1, what);
  return p;
}

// Product of s[a, b), each factor >= 0. A zero factor makes the product 0, but only after the extent bound of the
// same range has been checked (N2: a zero never hides an overflowing neighbour).
inline int64_t ck_prod(const std::vector<int64_t>& s, size_t a, size_t b, const char* what) {
  const int64_t bound = ck_extent_bound(s, a, b, what);
  for (size_t i = a; i < b; ++i)
    if (s[i] == 0) return 0;
  return bound;
}

// Bytes of `numel` fp32 elements, refused above INT64_MAX.
inline int64_t ck_fp32_bytes(int64_t numel, const char* what) {
  return ck_mul(numel, static_cast<int64_t>(sizeof(float)), what);
}

// Element count of a planned output/scratch shape, plus the check that the fp32 byte size of its EXTENT BOUND fits
// (N2: so every byte stride fits too, also for a count of 0); returns the count.
inline int64_t ck_shape_numel(const std::vector<int64_t>& s, const char* what) {
  const int64_t bound = ck_extent_bound(s, 0, s.size(), what);
  ck_fp32_bytes(bound, what);
  return ck_prod(s, 0, s.size(), what);
}

}  // namespace tcs
