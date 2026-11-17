# Instrument receipt clarification

`gpu_queried: false` in instruments.json describes the instrument-discovery command: no hardware probe or nvidia-smi command was run. It does not mean CPU tests never call a masked CUDA API. Original production tests query/reject unavailable CUDA devices and some test their CUDA API contracts with existing test shims. Every seat process had CUDA_VISIBLE_DEVICES empty; no visible-device workload ran. First actual device execution remains the lead's staged G4 run.
