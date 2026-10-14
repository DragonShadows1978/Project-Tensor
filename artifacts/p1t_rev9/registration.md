# P1T-REV9 immutable registration
Date: 2026-10-13. Seat: P1T-REV9.
Order: independent review of 9df5bef..e4cf1d6, tensor_cuda/ and tc_signal/; FIX8 delta 6991939..e4cf1d6. The user order is the immutable plan.
Writes only artifacts/p1t_rev9/, logs/, docs/P1T_REV9_*.md and /tmp/p1t_rev9_*; no product edits, no subagents, no GPU execution, no deletes, read-only git only.
Gates: source review of all specified areas and 29 closure sites; independently inspect code before FIX8 verdicts; inspect sweep k/l/L2; CPU-only compile/host checks under timeout. Pass requires no open BLOCKING or MAJOR finding; report every real defect at true severity. GPU-dependent unresolved claims receive probes, never invented passes. Host checks pass only on exit 0 with every registered case exercised; timeout/nonzero is RED or INCONCLUSIVE, not a clean review. Do not broaden into baseline product behavior or Voice acceptance/performance.
Prior art: use DLPack/Python array API, CUDA, CPython, pybind11, C++ and ONNX standards for contract reasoning. All proposed probes and fixes receive source/lead references. No new product algorithm implemented.
