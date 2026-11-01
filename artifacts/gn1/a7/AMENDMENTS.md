# A7 execution amendments

1. Pre-gate native-symbol diagnosis resolves the missing Python query binding without crossing scope: the pinned engine ELF depends on libcudart.so.12 and dlsym through its handle exposes CUDA runtime queries. D2 will call those existing runtime APIs from the tooling, with CUDA header source citations and checked statuses. No engine change, rebuild, device run or substitute provider. Python binding absence remains a recorded diagnostic, not a scope-mismatch finding. Registration acceptance thresholds are unchanged.
