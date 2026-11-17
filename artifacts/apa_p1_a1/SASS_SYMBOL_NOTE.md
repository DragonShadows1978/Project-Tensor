# SASS symbol correspondence

Raw symbol matching reports 28 missing-symbol rows: 14 compiler-generated anonymous-namespace symbols change their build hash/serial. No matched function body differs. Compare all function bodies verbatim, including encodings and control words; map ONLY the generated hash/serial in gemm_apa.cu and paint_raster.cu symbol names. Require a bijection with no collisions. No instruction text is normalized. This is symbol correspondence, not a change to the equality threshold. Original raw result retained.

Prior art: nvcc anonymous-namespace internal linkage and round-1 symbol correspondence (2026).
