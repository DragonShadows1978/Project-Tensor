# ENG1-A1 registration (immutable)

Acceptance: all G1–G4 as ordered. No E3 ID gates numerical accuracy. All 40 E3 cases record float64-reference error, emulation-reference error and engine-emulation distance. Existing targets unchanged and relabeled accuracy targets (not error bounds).

Review reproductions: RMSNorm float16 near_eps error 0.0156582687232443 and LayerNorm bfloat16 near_eps error 0.204845083463340, absolute reproduction tolerance 5e-12 (CPU deterministic emulation).

E2 new stored positive scalar witnesses use existing per-element tolerance: (u_dtype + 2^-20)*abs(reference) + half-min-subnormal. No GPU-based tuning. Include review x=0.0002994537353515625, g=0.30078125, p=3 bfloat16 and a non-power-of-two float16 upstream witness.

E5 new width_zero geometry: input (0,1,6,2), kernel (3,3), stride (1,1), padding (0,0); floor formula gives OH=4, OW=0. All four callers must refuse with complete geometry message.

M15: one bfloat16-only early cast in scratch-copy ops.cpp; GPU witness must fail, candidate must pass. M16: only OW<=0 weakened to OW<0 in scratch-copy conv.cu; all four width_zero CPU tests must fail, candidate must pass. Builds CUDA12.6 -j2, sources in candidate read-only.

All staged lines prepend selected package parent in inherited PYTHONPATH and run same-env/same-cwd subprocess origin preflight against module path and sha256; any mismatch refuses. Both arms run review probe scenario. Source-reading tests listed separately and run only on candidate, excluded from arm comparison.

Stage 24 lines (22+M15+M16), expected child exits explicit. New seal retains old. Audit uses CPU stand-in and cannot count as G5. CPU subset must pass, IDs collect. Parent records child returncode only after exit.

G1 diff base 5031c765; source hashes unchanged; candidate pin 3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53. All commands TMPDIR=/tmp/eng1_a1_eng1, CUDA_VISIBLE_DEVICES empty, threads 1, nice19, timeout <20m. No deletions. Budget 3h.
