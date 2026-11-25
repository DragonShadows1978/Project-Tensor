# BWD hardening and ENG5-S promotion record

**Change.** This record covers one merge of two engine lanes over production main `75c85de6`. Both lanes contain ENG4.
- **BWD hardening:** BWD-F1 to F4, plus BWD-F5, which repaired the test harness only.
- **ENG5-S:** the status checks, plus ENG5-S-T, its production regression test.

The product and test diff is 18 files in `tensor_cuda/` and `tc_signal/` (+3,109/−131): 8 product files (BWD's seven, plus
`packed_prefill.cuh` from ENG5-S), the tc_signal README, and 9 test files (one of them a test-support source). The rest of the merge is receipts under `artifacts/`, the
orders and the rounds' scripts.

- **Zero-extent broadcasting (BWD-F1, item BROADCAST-ZERO-EXTENT).**
  - `broadcast_shape` now follows NumPy: an extent of 0 against 1 gives 0. Production's max rule gave 1, and the binary
    kernel then read the empty operand's storage at address 0.
  - The binary ops, compare, `where`, `masked_fill`, `broadcast_to`, `reduce_to`, `einsum` and `axpy_` refuse
    incompatible shapes before any device work, naming the shapes.
- **Copy ops (BWD-F1 Amendment 1, item CAT-ND-SHAPE-UNCHECKED).**
  - `cat`, `stack`, `slice`, `splice_rows`, `evict_rows` and `export_rows` check rank, dim, shape, dtype, device and byte
    bounds before allocating.
  - Production took the shape of `cat`'s output from its first operand and checked nothing on the later ones.
- **Audit hardening (BWD-AUD1 F06 to F11).**
  - tc_signal's backward refuses an `out` that requires grad, with a named message.
  - `attach_vjp` and matmul backward give contextual messages.
  - The tc_signal README documents BWD's scope.
- **Review fix rounds** (findings from the Codex reviews below):
  - F2:
    - the RoPE sine span and the table devices are checked;
    - a rank ceiling stops a rank-9 result from `stack`;
    - the ring index is overflow-safe;
    - `export_row_pair` validates both sides before any work;
    - `where` loads its condition by its own dtype;
    - byte bounds and the dtype check come before allocation;
    - `axpy_` refuses a broadcast output that differs from the destination;
    - `einsum` checks every operand.
  - F3:
    - empty RoPE spans pass;
    - contiguous `[T, D, 1]` tables are accepted;
    - reduction axes are range-checked and duplicate axes are refused, `mean` included;
    - the byte bound comes before pair work.
  - F4:
    - `mean`'s extent product is checked for overflow;
    - RoPE tables above rank 8 are refused;
    - two comments were corrected.
- **CUDA status checks (ENG5-S).**
  - One helper, `check_cuda_status` (`kernels.cu`, declared in `core.h`), raises `RuntimeError` naming the entry, the
    call, and CUDA's error name and text.
  - It now covers nine sites that ignored a CUDA status: `cuda_sync`, the out-of-memory drain, `zeros`, `from_host`,
    `to_host`, `clone`, `to`, `axpy_`, and the packed-prefill NaN fill.
- **Production defects this fixes,** each confirmed on the GPU against production's build:
  - zero-extent broadcasting read device address 0 and left a sticky illegal-address error that ends the CUDA context;
  - `cat`, `stack` and `slice` with a dim below range corrupted the host heap, and `cat` of rank-0 operands crashed;
  - out-of-range reduction axes corrupted the host heap ("malloc(): unaligned tcache chunk detected", "free(): invalid
    pointer"), and a duplicate-axis `sum` gave 6 memcheck errors;
  - the RoPE read path gave 44 memcheck errors;
  - with a CUDA failure injected in each of ten rows (eight of the nine sites; the packed-prefill fill needs a hooks
    build), production's build returned as if the call had succeeded.
- **Input domain (lead ruling D1, BWD round 3).** Some cases where production returned a value for an empty no-op are
  now refused as outside the engine's domain, and are recorded as domain refusals, not regressions:
  - ranks above `TC_MAX_DIMS`;
  - dtypes that must match by the op's contract;
  - an in-place destination whose shape differs from the broadcast result.

  A grep of the consumers found no such call in Voice or GRAPA.
- **Intended test change:** in `tc_signal/tests/test_bwd_r3_paths.py`, one assertion now expects the named refusal of an
  `out` that requires grad. That changes two IDs, `test_host_real_planner_and_no_grad[conv1d]` and
  `[conv_transpose1d]`.
- **New tests:**
  - `test_bwdf1_zero_extent.py`, `test_bwdf1_copy_ops.py`, `test_bwdf2_fixes.py`, `test_bwdf3_fixes.py`,
    `test_bwdf4_hardening.py`;
  - `tc_signal/tests/test_bwdf1_hardening.py`;
  - `test_eng5s_status_checks.py`, which builds its injection shim (`tests/support/eng5s_cuda_shim.c`) when it runs.
    Its GPU rows skip on the CPU lane with a reason naming the lane.
  - ENG5-S's own differential test reads build folders that exist only in the fork, so it stays on the fork branch and
    is not part of this merge.

**Scope.**
- **Covers:**
  - zero-extent broadcasting and its guards;
  - the copy-op checks;
  - the audit hardening;
  - the review fixes F2 to F4;
  - the nine CUDA status checks;
  - the tests above.
- **Does not cover:**
  - host-tensor compute (HOST-COMPUTE-HMM; ENG5's host guards were parked for a later engine round);
  - `int4_linear_fused` with K = 0 (INT4-K0-UNWRITTEN) and `embedding_forward` with V = 0 (EMBED-V0-DIV);
  - Storage's destructor discarding its free statuses (NATIVE-STORAGE-FREE-STATUS);
  - the successor items filed for the next engine round (index checks in gather and scatter, flip axes, malformed
    `einsum` strings, `cumsum` axis, and three inexact "cannot fire" comments);
  - a production test for the packed-prefill status site (it needs a hooks build; the site is evidenced by ENG5-S's GPU
    lines);
  - performance;
  - installing a rebuilt engine binary.

**Review: cross-model.** Claude seats wrote every round: Opus 5.5 for BWD-F1, F2, F3 and ENG5-S, and Sonnet 5.5 for
BWD-F4, F5 and ENG5-S-T. Codex gpt-6.2-sol reviewed them.
- **BWD-AUD1** (Claude, blind, read-only) audited production's BWD diff. It reported 0 BLOCKER, 3 MAJOR, 3 MINOR and 5
  NIT, and its GPU witnesses found the zero-extent defect. F06 to F11 were taken into BWD-F1.
- **BWD-F1** (`fbe04d8f9`). BWD-F1-REV1 (Codex, read-only, complete) reported 13 findings: 6 BLOCKER, 4 MAJOR and 3
  MINOR. All were fixed in F2 (`e4ad8381a`).
- **F2.** BWD-F1-REV2 (Codex confirm) reported 8 findings: 6 BLOCKER, 1 MAJOR and 1 MINOR. The provider's content
  filter stopped it after it wrote its findings.
  - Four (F03, F04, F05, F08) were closed by ruling D1. They are refusals of empty no-ops outside the domain.
  - The other four were fixed in F3 (`2117296ea`): E1 to E4, including an inherited memory defect, `sum([8])` on rank 1
    writing past the reduction spec.
- **F3.** BWD-F1-REV3 (Codex confirm, complete) gave PASS with non-blocking hardening: 8 of 8 closed, 0 blockers, and 4
  hardening items (1 MAJOR, 1 MINOR, 2 NIT). All four were fixed in F4 (`529eea9de`).
- **ENG5-S** (`fb833679f`). ENG5-S-REV1 (Codex, read-only confirm, complete) gave CONFIRMED with 0 findings. It named
  two successors outside the diff, which are filed.
- **ENG5-S-T** (`d14ac7634`) adds only a test and its shim. It is verified by its own GPU lines, below.
- **Where the receipts are:** each review's findings and report are on its review branch in the fork. Their logs are held
  back, listed with sha256 in each branch's `LEAD_HELD_BACK.tsv`.

**Gate evidence.**
- **BWD, CPU:**
  - F1 G1: 1,561 passed. 6 of 6 CPU mutants killed.
  - F2: 1,568 passed; 4 of 4 mutants killed.
  - F3: 1,625 passed; mutants p01 to p04 killed.
  - F4: 24 rows as registered; mutants h01 and h02 killed.
- **BWD, GPU** (the lead's runs):
  - On F3: G2 audit witnesses, G3, K2 and the P2 count all PASS, and mutants m03, n03, n04, n05 and p04 KILLED.
  - The whole suite per test ID, production's build against F3: 5,424 IDs, and exactly the 2 intended differences.
  - BWD-F5's per-row memcheck lines E1 to E3 PASS on F4, which is this tree's BWD content. Production's build shows
    each defect listed above.
- **ENG5-S:**
  - CPU: 612 passed.
  - GPU injection at all eleven rows: Build S raises at each one; production's build returned from every row; mutants
    s01 to s04 killed.
  - The whole suite per test ID on the GPU: 4,278 IDs, 0 differences.
- **ENG5-S-T, GPU:**
  - the test passes on Build S;
  - it catches all ten injected rows on production's build, and the control passes;
  - each of the four mutants is killed on its own rows.
- **Modules:**
  - Build F4: `8061f980133350fde37941c786444ce186097d05a26151da4d7dfa85681470e1`
  - Build S: `6b81f0c6`, full sha in ENG5-S's build receipts.

**Before push:** a merged-tree check (MAINCHK7) built this tree from its own commit `ea32d5665` (Build M7, module
`c92bf231236a8f65f46425a1fb573978c1dbfe40d0535396b3c3394e7ac1a463`). It also built production `75c85de6` fresh and
compared the two per test ID.
- **Builds and scope:** PASS.
  - Each tree equals its commit before and after the build.
  - ENG5-S's diff names its three engine files, and BWD's names its seven product files.
- **CPU:** PASS, two runs per arm.
  - 6,321 IDs on this tree against 4,657 on production. 4,655 match, with 0 mismatches and 0 noise.
  - The 2 intended IDs pass.
  - The 1,664 new-file IDs pass, or skip with the GPU-lane reason (11 rows).
- **GPU:** PASS.
  - **The whole suite per test ID:** 7,089 IDs on this tree against 5,425 on production. 5,423 match, with 0 mismatches
    and 0 noise. The 1,664 new-file IDs and the 2 intended IDs are as expected.
  - **Three reruns** close the gaps the lead found in BWD-F3's GPU receipts:
    - **Production's accepted results against this tree, one process per row, bitwise:** 174 of 180 keys passed as
      registered. The other six were the `eq` outputs, which were all zeros on both builds because the inherited
      control inputs never tie, and the check does not accept a constant output. An amendment reran those six with
      sealed inputs that tie. All six are bitwise equal between the builds, in the input dtype, not constant, and equal
      to NumPy's result.
    - **A real rank-0 `export_rows`,** with the operand's shape asserted `[]` before the call: refused with "export_rows:
      rank must be nonzero", while the rank-3 control launches.
    - **BWD-F3's two call-count lines:** PASS on this tree. Production's build shows the defects.
- **Receipts:** on the check's branch `mainchk7` in the fork (`45ca2af13`, `87cef3126`, `9d5a16c5d`). Logs and arrays
  are held back, listed with sha256 in `artifacts/mainchk7/LEAD_HELD_BACK.tsv`.

**The tree** is BWD's lane tip, merged with production `75c85de6` and with ENG5-S's tip. It leaves out ENG5-S's
fork-only differential test (`git rm --cached`; no test content edited). Apart from the 18 product and test files, it
only adds files to production's tree.

**Approval:** the owner approved this merge via Ember on 2026-11-25 (01:12 UTC). The reply was A to the decision
request `20261124-151158_Project-Tensor_bwdh-eng5s-merge-decision.md`: merge into production main and push. The verbatim
text is in the local promotion ticket and the lead's register. The engine binary install is not part of this merge and
remains a separate request.
