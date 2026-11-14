"""Explicit path-reader class from source inspection (including provenance helpers).
Prior art: REV1 D4 source-vs-import distinction; Python ast/PSF. Taken: itemized call sites; ours: arm exclusion.
"""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'artifacts/eng1/a1';T=R/'artifacts/eng1/tool'
rows=[]
def add(file,name,paths,via):
 rows.append(dict(test_id='tensor_cuda/tests/'+file+'::'+name,paths=sorted(set(paths)),via=via,classification='source_path_reader',comparison='excluded on both arms',execution='candidate regression only'))
def source_paths(file):
 j=json.loads((R/file).read_text());out=[]
 def walk(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k.startswith(('scripts/','tensor_cuda/','orders/','/mnt/')):out.append(k)
    walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(j);return out
sp3=source_paths('artifacts/apa_sp3/registration.json')
add('test_apa_sp1_1.py','test_registration_and_every_existing_kernel_body_pinned',['tensor_cuda/src/kernels.cu'],'direct read_text; registration snapshots/receipt hashes additionally checked')
add('test_apa_sp3.py','test_every_preexisting_source_and_kernel_body_pinned',sp3,'common.verify_sources: registration.source_sha256 and external_sha256 (may refuse on first changed source)')
add('test_apa_sp3.py','test_d96_prefill_contract_and_dense_pin',['tensor_cuda/src/kernels.cu'],'direct read_text')
add('test_apa_sp3.py','test_d96_splitk_contract_and_dense_pin',['tensor_cuda/src/apa_sp1_1.cuh'],'direct read_text')
add('test_apa_sp3_a4.py','test_original_model_and_scorer_bytes_unchanged',['scripts/apa_sp3_model.py'],'common.sha explicit source')
add('test_apa_sp3_a6.py','test_interposer_shell_new_kind_policy',['scripts/apa_sp3_lead_gpu.sh'],'direct read_text')
add('test_apa_sp3_a6.py','test_a6_bridge_all_old_kinds_future_edits_and_receipt_bytes',sp3+source_paths('artifacts/apa_sp3/a6_before.json'),'provenance.current_fingerprint/compatible and common.verify_sources; before.closures sources')
for suffix,name in [('a2','test_a2_registration_precedes_edits_and_a1_receipts_closures_preserved'),('a3','test_a3_registration_preservation_and_real_prefix_geometry'),('a4','test_a4_registration_preservation_and_transitive_gate'),('a5','test_a5_propagation_ungated_and_registration_preserved'),('a7','test_a7_registration_grid_rails_anchors_and_preservation')]:
 paths=source_paths('artifacts/apa_sp4g/'+suffix+'_before.json')
 # preserved() also delegates to original source validation; A5 delegates through A4.
 paths+=source_paths('artifacts/apa_sp4g/registration.json')
 if suffix=='a5': paths+=source_paths('artifacts/apa_sp4g/a4_before.json')
 add('test_apa_sp4g_'+suffix+'.py',name,paths,'common.preserved(): source hash loops and delegated source checks')
add('test_apa_sp4g_a2.py','test_a2_source_pins_pre_branch_norm_rope_v_and_scale',['/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py'],'direct external source read_text; not an arm engine comparison')
for suffix,name in [('a2','test_a2_runner_preserves_bounds_and_uses_isolated_driver'),('a3','test_a3_runner_bounds_and_no_historical_source_changes'),('a7','test_a7_foreground_shell_safety_and_rail_isolation')]:
 add('test_apa_sp4g_'+suffix+'.py',name,['scripts/apa_sp4g_'+suffix+'_lead_gpu.sh'],'direct read_text')
(T/'source_reading_tests.json').write_text(json.dumps(rows,indent=2)+'\n')
print(len(rows),'source-reader test selectors')
