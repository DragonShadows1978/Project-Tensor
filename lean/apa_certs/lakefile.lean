import Lake

open System Lake DSL

-- TX3 X5: APA certificates as Lean statements. Mathlib pinned at the house OM4 commit; the built packages are a copy
-- of the house cache (System-Claude mail 2026-10-28 09:21), used offline: no `lake update`, no `cache get`.
package ApaCerts where
  version := v!"0.1.0"
  fixedToolchain := true
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "d13f23b723b8a846827a245b89c10fc7d3f11612"

lean_lib ApaCerts where
  globs := #[`ApaCerts.+]
