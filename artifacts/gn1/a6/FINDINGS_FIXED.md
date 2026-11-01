# REV5 finding fixes

**RED-D0: D1 is implemented but does not close receipt-forgery scope. R5-F01/F02 are not claimed fixed under all of D0.**

Evidence class: finite CPU controls through the actual A6 parent and scorer, under D0.

| Finding | Result | Fix and deciding evidence |
|---|---|---|
| R5-F01 MINOR | D1 implemented; D0 closure not claimed | D1 checks the pinned runner at command[0], plus absolute/resolved path and SHA-256. G4_a6.json: Q2_wrapped_runner_command0, D0c_runner_substitute, Recorded_pin_runner. |
| R5-F02 MINOR | D1 implemented; D0 closure not claimed | D1 fixes PATH before shell lookups, uses absolute runner/timeout/Python and Bash paths, unsets summary/window/lock overrides on non-audit lines, and binds every received PATH/FORGE_GPU_* value. G4_a6.json: Q2_path_shadow_inner_timeout, D0c_* substitutions, seven isolated program-pin controls, eight isolated environment controls, six actual environment controls. |
| R5-F03 NIT | FIXED | D2 catches Exception at both scorer read boundaries and parent dependency reads. G4_a6.json: all three NEW_*deep* cases write refusing summaries with recorded recursion reasons. |
| R5-F04 NIT | FIXED | D3 requires type(raw_exit) is int and recomputes exit = raw_exit if nonnegative, else 128 - raw_exit. G4_a6.json: minus9/exit0, missing, bool and string edits all refuse. |
| R5-F05 NIT | FIXED | D4 hashes the seal inside protection, records errors, retries publication once with a minimal error receipt, and records double-write failure. G4_a6.json: REV5_postexit_seal_failure, D4_write_retry, D4_double_write_incomplete. The final case has no parent receipt and the scorer explicitly marks it incomplete. |

D0_MODE_PROMOTION.json accepts all58 lines after59 receipt metadata edits, without a seal/child/log/program/argv change. D1 comparisons alone cannot authenticate these records against consistent after-the-fact forgery. The unaltered wrapper and PATH reproductions refuse, but the stronger D0 receipt claim is not closed.
