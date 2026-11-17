# Deviations
- Initial rule-read used TMPDIR=/tmp, timeout30 and CUDA hidden, but omitted nice19/single-thread variables and dedicated scratch. Read-only cat only. Corrected all subsequent commands.
- Memory quick pass found no applicable REV2/F2 notes; no memory-derived review claims.
- blind_checks.py repair 1/3: assertion incorrectly expected delta=0 at epsilon=.01 and e_q=0; use epsilon=1, per derive_delta. Original exit1 receipt 010 retained.
- blind_checks.py repair 2/3: all-zero unit vector assigns central code, whose sign differs from Inf-input code0; choose q=-ones for the finite-overflow construction to exercise negative bulk. Original exit1 receipt 011 retained.
- r01_finite_norm_overflow.py repair 1/3: dense table is held back from this worktree; name the existing read-only production table, sha-pinned, as REV2 did. Initial CPU-prefix exit1 (FileNotFoundError) retained in receipt024.
- Report assembly tool slip (exit127): nested shell quoting truncated the write of finalize_review.py; no review gate ran. Repaired the literal heredoc before executing the script. Diagnosis receipt032a retained.
- Read-only probe of tensor_cuda/tests/conftest.py in receipt023 found no such file; no dependency on it. Collection itself had zero errors.
- F03 refinement: after tracing eb2b.cuh:179 to packed_rotate, the later widened product check mitigates the oversized product before allocation under modular execution. No OOB claim is retained.
- The F2 report has no quoted lead-ledger entry or path; local named ledger searches found no F2 entry. Optional path clarification requested; review continues from raw receipts.
