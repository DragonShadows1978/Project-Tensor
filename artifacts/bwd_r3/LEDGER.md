# BWD-R3 ledger

- Mechanism and operands read from DESIGN.public.md, Voice model.py:738 and :742, and the BP1/BP2 manifests. The two manifest hashes match the order and the Voice research/backward64/ops.py SHA-256 is fb3fca078cc98198dc497665257a140d19030183c70dd4328154832555642f01.
- Registration and numerical tolerance files were created and sealed before candidate comparisons. The dispatch commit differs from the accepted parent by the order file.
- L01 first calls used cmake 3.28.3, nvcc 12.0.140 via PATH, pytest 9.0.2 and Python 3.12.3. G1 will explicitly use the CUDA 12.6 toolchain used for R2.

## Prior art

- Dumoulin and Visin (2016), convolution arithmetic, gives the adjoint tap mapping. Chellapilla, Puri and Simard (2006) give im2col/GEMM. Harris (2007) gives the shared-memory reduction tree. R3 maps both ONNX layouts onto this gather and packs/reuses one group tile with explicit ascending merges.
- NVIDIA cuBLAS 12.6 (2024), results reproducibility, workspace, pedantic compute and TF32 override, and TensorCUDA BWD-R1 (2026), private strict handle and call/member consistency. R3 owns a separate sidecar handle and aligned numeric canary.
- PyTorch (Paszke et al., 2019), convolution autograd and saved-operand checks, and BWD-R2 (2026), attachment and inference routing. R3 adds convolution slots and the four missing-VJP refusals.
- Fornberg (1988), centered finite differences, and DeMillo, Lipton and Sayward (1978), mutation testing. MAINCHK3 and BWD-R2 (2026) supply per-ID pytest phase folding and isolated package validation. R3 binds these to both convolution layouts, Voice geometry and the registered edits.
- Unverified — lead to check: convolution arithmetic Dumoulin Visin 2016, Chellapilla document processing im2col, Mark Harris optimizing parallel reduction, cuBLAS 12.6 GemmEx pedantic TF32 reproducibility, PyTorch convolution_backward, Fornberg centered differences, DeMillo Lipton Sayward mutation testing.

## Gate receipts

- G1 passed under CUDA 12.6 at SM89, nice 19, -j4. The candidate links the existing forward objects plus conv_backward.cu and preserves the pinned R2 core. Source and binary hashes are in G1_SOURCES.json and G1_MODULES.json. L01 dry admission returned exit 78 with would_start=true.
- G2 host passed 28 cases, including the separate tracked-out smoke. G4-key passed all four cells with a worst plateau under 2.1e-10. The G6 host witness showed a 6.36538904× gap from TF32 rounding. The initial host script had lost shell-interpolated quotes and raised NameError before starting pytest. Repair 1 corrected argument printing and reused fresh a2 paths.
- G8 finished all arms at 12 GiB RLIMIT_AS. Against parent, +151 R3 IDs. Against production, +376 R1/R2/R3 IDs. Both comparisons had zero changed existing IDs and F10 attribution is empty.
- G7 built eight isolated native mutators by recompiling the edited CUDA TU with the G1 flags and linking the unchanged G1 objects. The isolated r3m08 source edit was killed with exit 1 after reaching the real planner. The untouched arm passed with no planner entry.
- The CPU fixture census verified 52 file hashes and 156 stored-key comparisons against the oracle on the rounded inputs, and recorded scratch queries. It did not run the candidate gradients.
- The first staged node inventory omitted the independent handle configuration case. Repair 2 added the node to the unmasked line. The successful inventory covers all GPU pytest IDs. All 31 dry lines exited 78 with would_start=true and no lease. G2–G7 GPU qualification and the CUDA pool/runtime measurements remain pending.
- G9 passed in the fresh final tree, 29 passed and 122 skipped under the inherited seccomp policy. A command typo selected g9 in the earlier recheck, which ran only the host canary and ended with KeyError. Corrected invocation g9_plain final passed without a script change.
- G10 verified the authorized path set, protected source hashes, no staged changes, no removed tracked files and an empty forbidden-string search. No deletes or GPU lease. Remaining scratch is inventoried in FILES_WRITTEN.txt.
