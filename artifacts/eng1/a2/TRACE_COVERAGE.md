# CPU trace coverage — A2

19 observed source-reader selectors; all 19 listed; **0 missing**. All 49 regression files and 2133 exact IDs ran under the unchanged REV2 attribution/classification scripts. The 36 blocked operations and trace pytest failures are retained instrument outcomes; this is a read census, not a passing regression suite.

| Listed only by name | Reason |
|---|---|
| `tensor_cuda/tests/test_apa_sp1_1.py::test_registration_and_every_existing_kernel_body_pinned` | Fails first on missing artifacts/apa_sp1/sp1_1_before/tensor_cuda__src__kernels.cu (FileNotFoundError); the later live-source read is not reached. See q2_readers_A_retry.stdout. |
| `tensor_cuda/tests/test_apa_sp3_a6.py::test_a6_bridge_all_old_kinds_future_edits_and_receipt_bytes` | Fails first on missing artifacts/apa_sp3/build/manifest.json (FileNotFoundError); the later source closure checks are not reached. See q2_readers_A_retry.stdout. |
| `tensor_cuda/tests/test_apa_sp4g_a2.py::test_a2_source_pins_pre_branch_norm_rope_v_and_scale` | Passes; directly reads /mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py, outside the worktree-only audit trace. See test_apa_sp4g_a2.py and q2_readers_B.json. |

Residual: CPU reads after an initial failing CUDA call are not observed. Imports, traceback rendering, tree walks and artifact/order receipt reads are excluded exactly as in REV2. The pyproject.toml collection read is retained under collection_reads and is not attributed to a test.
