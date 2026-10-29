import ApaCerts.Proofs

/-!
# X5 check (lead-written, immutable)

Each proof must have exactly its statement's type. The build fails if a name is missing or a type differs. The
axioms printed must be among `propext`, `Classical.choice` and `Quot.sound`.
-/

namespace ApaCerts.Check
open ApaCerts

example : sp_skip_final_stmt := sp_skip_final
example : sp_skip_final_nv_stmt := sp_skip_final_nv
example : sp_mask_superset_stmt := sp_mask_superset
example : sp_mask_superset_nv_stmt := sp_mask_superset_nv
example : splitk_contains_stmt := splitk_contains
example : splitk_contains_nv_stmt := splitk_contains_nv
example : z_not_single_pass_stmt := z_not_single_pass
example : z_not_single_pass_nv_stmt := z_not_single_pass_nv
example : sp2_certificate_stmt := sp2_certificate
example : sp2_certificate_nv_stmt := sp2_certificate_nv

end ApaCerts.Check

#print axioms ApaCerts.sp_skip_final
#print axioms ApaCerts.sp_skip_final_nv
#print axioms ApaCerts.sp_mask_superset
#print axioms ApaCerts.sp_mask_superset_nv
#print axioms ApaCerts.splitk_contains
#print axioms ApaCerts.splitk_contains_nv
#print axioms ApaCerts.z_not_single_pass
#print axioms ApaCerts.z_not_single_pass_nv
#print axioms ApaCerts.sp2_certificate
#print axioms ApaCerts.sp2_certificate_nv
