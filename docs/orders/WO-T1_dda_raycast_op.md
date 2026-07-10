# WO-T1 — tensor_cuda: add `dda_raycast` custom op (H1-A)

You are an implementation agent on Project-Tensor (repo root = cwd).
Read AGENTS.md / house rules if present, and the two driving documents:

- `/mnt/ForgeRealm/Project-Scorch/render_cuda/ENGINE_ASSESSMENT.md`
  (the H1-A hook spec — this is your contract)
- `/mnt/ForgeRealm/Project-Scorch/render_cuda/dda_kernel.cu`
  (the kernel to host — treat as the reference implementation)

Context: Project-Scorch (voxel game, this machine) needs a voxel DDA
raymarch op. Engine assessment confirmed tensor_cuda has no external
kernel-launch path, so the op lands as a first-class engine op.

## Deliverables

1. `dda_raycast(grid_u8, origins_f32, directions_f32, max_steps)
   -> (hit_u8, material_u8, voxel_i64, face_axis_i64, face_sign_i64,
   distance_f32)` as a public tensor_cuda op:
   - Inputs: grid = 3D uint8 device NDArray; origins/directions =
     (N,3) float32 device NDArrays; max_steps int.
   - Kernel: adapt `dda_kernel.cu` (one thread per ray, Amanatides-Woo)
     into the engine's kernel source tree following its existing kernel
     organization/build conventions (CMake, Release default — do not
     change global build flags).
   - Semantics must match the Scorch NumPy reference
     (`/mnt/ForgeRealm/Project-Scorch/render/dda.py`) exactly: hit
     voxel identity, material, hit-face axis+sign, traversal distance,
     miss behavior, ray-starts-inside-solid behavior.
   - Known engine quirks (from the assessment): no Int32/Int8 NDArray
     dtypes (use Int64/uint8 as in the signature); bool NDArray
     .numpy() is broken — use uint8 for the hit mask.
2. Unit test in the engine's existing test layout:
   - 8×8×8 hand-fixture known-answer cases (mirror the cases in
     `/mnt/ForgeRealm/Project-Scorch/render/test_dda.py`: axis-aligned
     hit, diagonal traversal sequence, start-inside-solid, clean miss);
   - a randomized cross-check vs a tiny Python reference marcher
     (200 rays, small grid) — exact field equality.
3. Build + full existing engine test suite green (run it; report
   verbatim counts). Your new op must not alter any existing kernel,
   header, or public API beyond adding the op.

## Rails

- ABSOLUTE: do not modify APA/attention code paths, existing kernels,
  or existing op signatures. Additive change only.
- Do not run git commands; the lead reviews and commits.
- Do not touch Project-Scorch (read-only reference for you).
- No new dependencies. Engine conventions are law — match the existing
  kernel registration/binding pattern exactly.
- If the engine's build/binding machinery genuinely cannot host the op
  without structural change, STOP and report the exact obstacle +
  minimal structural proposal (BLOCKED verdict is a valid result).
- Report: files added, build output tail, engine suite counts
  before/after, new-op test results verbatim, and the exact Python
  call signature for the Scorch side.
