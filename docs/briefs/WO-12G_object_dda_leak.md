# WO-12G — object DDA grazing-angle corner leak (engine)

Implementation agent, Project-Tensor (repo root = cwd). Additive
engine order on terrain_render's OBJECT path (WO-9A-e line; ledger
docs/briefs/WO-9A_e_ledger.md). Same rails; append to
docs/briefs/WO-12G_ledger.md. Build the .so ONLY at the very end
(consumer verification is coordinated by the lead).

## Symptom (lead receipts)

Large flat-faced voxel OBJECTS show background-colored speckle at
grazing view angles (Scorch receipt: a 32x32x40 building's side face
peppered with sky pixels; operator independently reported "sometimes
you can see through the voxels"). Terrain faces do not visibly show
this; the object-path DDA (ray transformed into object-local frame,
AABB entry, integer DDA) is the suspect — corner/edge traversal
precision at oblique entry.

## The work

1. REPRODUCE deterministically first: fixture = solid 32x32x8 slab
   object, camera at grazing angles; count interior background
   pixels (project the slab's silhouette analytically; any
   background pixel strictly inside it = leak). Record the leak
   count per angle in the ledger BEFORE fixing.
2. Diagnose the traversal defect (entry-point epsilon? tMax/tDelta
   accumulation? corner tie-breaking?) — name it.
3. Fix with the standard robust-DDA discipline (consistent epsilon
   at AABB entry, axis tie-break rules); terrain path untouched
   unless it provably shares the defect (if so, flag loudly — its
   parity fixtures must stay byte-identical or the change is a
   registered finding, not a silent edit).
4. REGISTERED GATE G-LEAK: zero interior-background pixels for the
   slab fixture across a registered sweep of 24 camera poses
   (azimuth x pitch incl. <5-degree grazing), 640x480, AND the
   existing objects parity fixtures stay byte-exact vs their NumPy
   reference (update the reference ONLY if the fix changes correct
   behavior — justify in ledger).
5. Perf: objects timing within +0.1ms of current (0.09ms for 4).
6. Engine suite green; receipts artifacts/wo12g_{before,after}.ppm
   at the worst measured angle.

Rails: tensor_cuda source additive/object-path, tests additive,
artifacts, ledger. APA + dda_raycast + terrain behavior untouchable.
No subagents/git-write/network/pip. Report: leak table, root cause,
gate table verbatim.
