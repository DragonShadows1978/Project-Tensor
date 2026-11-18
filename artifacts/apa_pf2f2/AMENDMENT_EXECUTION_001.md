# Execution amendment 001, G4 receipt plumbing

Post-dry-run source review found that the inherited FLT_MAX r02 path sent a
TensorCUDA tensor to the NumPy row oracle directly. The copied raw helper
returns tensors so G1 can inspect native BF16 bits. Convert the r02 output
with the same npf helper already used for the adjacent LF2 cases. The prior
dry run exercised admission and could not detect this GPU payload error.
This is own-script repair 1 of 3. Registered fixtures, kernels, modules,
payload commands and thresholds remain as registered.

Retain the old pins, dry receipts and draft CPU/scope summaries in
staging_before_repair_01. Re-pin the corrected G4 adapter and repeat every
launcher dry run and readback. The earlier build and SASS receipts still
refer to the same modules.
