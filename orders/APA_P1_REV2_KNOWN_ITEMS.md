# APA-P1-REV2: the lead's known items. Read only after your first findings.json is written.

These are the items the lead already holds on this code, with the lead's ruling. Check each ruling and severity. Say
where you disagree, and whether you found the item yourself before reading this file.

| ID | Item | Lead's ruling |
|---|---|---|
| P1F1-INF | For ±Inf in k, the 4-bit rotated-code (tq4) entries take the score's sign from the code dot product. A key that dense makes non-finite can be skipped, and the row stays finite (F1's LF2 `k_inf_more` row 6; LF2S row 0). The int4 entries carry any non-finite key into NaN, so they give more non-finite rows than dense. | Open, MINOR. Fix after this review: carry any non-finite key element into a NaN stored norm, as R1 did for int4. Gate: for every Inf case, each entry's non-finite row set contains dense's. |
| P1Q-API | `quantize_sp2_keys(k, bulk_bits, *, storage=None)` has no capacity argument, so its result cannot be appended to in a decode loop. `PackedSpKeys(k, capacity=N).append(...)` works and matches a per-step rebuild bitwise (P1-Q, 192/192 checks per model). | Open, MINOR. Fix after this review: forward a `capacity=` keyword. |
| F1-D3 | The old 14-argument `tc::apa_eb2b_run` in `ops.h` is declared and never defined. Global-namespace bridges stand in for it. | NIT. `ops.h` becomes writable in the fix round. |
| F1-D15 | The public route at G=1 now refuses misaligned k/v. Build B crashed on them (sticky misaligned-address error); the dense route accepts them. | Recorded; the reviewer is asked whether a refusal on the default route is acceptable there. |
| F1-G7 | Build F keeps six unreachable `tc::spq6::bulk_mma_tq` instantiations. Removing them in F1's first attempt moved shared-memory offsets in 96 unrelated kernels. F1's G7 fails as registered. The lead's count: 28 changed and 3 added kernels, all listed, 0 unlisted. | G7 FAIL stands. No product defect is claimed. The reviewer is asked to verify the count and the unreachability. |
| F1-RCPT | `SASS_COMPARISON_F_r2.json` records the original `EXPECTED.json` sha, while it compared against the override file `EXPECTED_SASS_F_r2.json`. | Receipt NIT. |
| P1-RES | Prefill on packed keys refuses (P1-D: its L3 check was not bitwise). | Temporary limitation; a packed prefill route is a later item. |
