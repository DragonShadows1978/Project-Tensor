"""Create-only fingerprint amendment, with exact before/after source endpoints.

Prior art: Make dependency invalidation (Feldman 1979), Nix content-addressed
builds (Dolstra et al. 2004), unverified leads to check. New: reviewed SP3
closure/transition manifest, not automated proof of semantic equivalence.
"""
import ast
import difflib
import json
from pathlib import Path
from apa_sp3_common import ART, ROOT, REG_SHA, publish, read, sha
from apa_sp3_a4_registry import MANIFEST, CORRECTION, ORDER_SHA
from apa_sp3_a4_provenance import BRIDGE, closure
from apa_sp3_gpu import base_cells, cells


def main():
    before=read(ART/'a4_before.json')['files']
    baseline=ART/'a4_baseline'
    kinds=sorted({c['kind'] for c in base_cells()})
    all_kinds=sorted({c['kind'] for c in cells()})
    legacy=[read(p) for p in (ART/'jobs').glob('*.json')]
    paths=set(before) | {p for c in cells() for p in closure(c)}
    paths |= {str(p.relative_to(ROOT)) for p in ROOT.glob('scripts/apa_sp3_a4_*')}
    paths |= {'scripts/apa_sp3_report.py','scripts/apa_sp3_lead_gpu.sh','scripts/apa_sp3_build.sh'}
    deltas={}
    reviews={
        'scripts/apa_sp3_common.py':'Only local path relocation, per-kind receipt validation and namespaced job paths; protocol/scoring contract unchanged; dependency validation stricter.',
        'scripts/apa_sp3_gpu.py':'Base registry renamed verbatim; overlay adds registered cells. Existing execute branches unchanged behind new-kind dispatch. Receipt writing/resume uses per-kind identity.',
        'scripts/apa_sp3_control.py':'Base dependencies/protocol retained. New kind fit/destination preflight, timeout metadata and receipt namespace only. Legacy worker rail unchanged.',
        'scripts/apa_sp3_lead_gpu.sh':'Allow isolated checkout; T excludes engine preload; new kinds use shorter registered rail; all legacy GPU branches and rails unchanged.',
        'scripts/apa_sp3_build.sh':'Only allow isolated checkout root; compiler options and sources unchanged. Lead must keep original built modules/manifest.',
        'scripts/apa_sp3_report.py':'Only receipt validation and additional T/ceiling/capture tables/commands; renderer not executed by measurement cells.',
        'scripts/apa_sp3_model.py':'r3 model bytes unchanged in a4. Older e95... -> c363... is mask.float() before cat; only diagnostic B blend capture changes; no scoring or unobserved forward changes.',
        'scripts/apa_sp3_a4_provenance.py':'New validation-only dependency bridge; added to legacy import closures, no measurement arithmetic.',
        'artifacts/apa_sp3/build/manifest.json':'Local a4 host rebuild is NOT whitelisted as equivalent runtime. Preserve lead original build; source eligibility is separate from runtime identity.',
    }
    for p in sorted(paths):
        if not (ROOT/p).is_file():continue
        after=sha(ROOT/p)
        old={r.get('fingerprint',{}).get(p) for r in legacy}
        old.add(before.get(p))
        if old=={after}:continue
        accepted=kinds if p in reviews else []
        if p=='scripts/apa_sp3_model.py':accepted=[k for k in kinds if k!='capture']
        if p=='artifacts/apa_sp3/build/manifest.json':accepted=[]
        # New registry/job/T modules are absent from legacy numerical closures.
        deltas[p]=dict(before_sha256=sorted(old,key=lambda v:v or ''),after_sha256=after,
                       unchanged_kinds=accepted,review=reviews.get(p,'New module/artifact outside legacy execution closures, or test-only change; no legacy waiver required.'))
    # Machine-check the load-bearing claim: original execution branch AST stays
    # identical after stripping the new-kind dispatch; base recipe is unchanged.
    def function(file,name):
        return next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    old=function(baseline/'scripts/apa_sp3_gpu.py','execute')
    new=function(ROOT/'scripts/apa_sp3_gpu.py','execute')
    assert isinstance(new.body[1],ast.If) and ast.unparse(new.body[1].test)=='kind in KINDS'
    new.body.pop(1)
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    old=function(baseline/'scripts/apa_sp3_gpu.py','cells')
    new=function(ROOT/'scripts/apa_sp3_gpu.py','base_cells');new.name='cells'
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    assert sha(ROOT/'scripts/apa_sp3_model.py')==before['scripts/apa_sp3_model.py']
    diff=[]
    for p in sorted(before):
        if (ROOT/p).is_file() and sha(ROOT/p)!=before[p] and (baseline/p).is_file():
            diff.extend(difflib.unified_diff((baseline/p).read_text().splitlines(True),
                         (ROOT/p).read_text().splitlines(True),fromfile='r3/'+p,tofile='a4/'+p))
    (ART/'a4_source_delta.patch').write_text(''.join(diff))
    publish(ART/BRIDGE,dict(schema='apa_sp3_per_kind_bridge_v1',immutable=True,
        registration_sha256=REG_SHA,order_sha256=ORDER_SHA,cell_amendment_sha256=sha(ART/MANIFEST),
        correction_sha256=sha(ART/CORRECTION),file_deltas=deltas,
        per_kind_import_closure={k:closure(dict(kind=k)) for k in all_kinds},
        external_closure='Engine source/headers/Python imports pinned by registration, adapter_import_cpu and build manifest recursively. Protocol/token/corpus pins checked separately. T remote Python/config pinned by a4_torch_sources.json; torch/transformers package versions in runtime receipt.',
        excluded_from_legacy_closures=['report renderer','mutation/test runner','build command script (actual build manifest remains required)','protocol creation script (immutable protocol and input bytes are separately checked)','new-kind algorithms not executed by legacy branches','new cell registration contents except schema validation'],
        acceptance='PASS + every per-kind closure file unchanged, or this exact reviewed before->after transition marked unchanged for that kind; unchanged dependency receipt hashes and recursively valid dependencies; identical pinned protocol. Future/unlisted edits rejected.',
        source_gate=dict(legacy_execute_AST='IDENTICAL after new-kind branch removal',base_registry_AST='IDENTICAL after rename',model_and_scorer_bytes='IDENTICAL',diff_sha256=sha(ART/'a4_source_delta.patch')),
        existing_job_inventory_sha256=sha(ART/'a4_before.json'),
        caveat='Original capture RED remains RED; new capture aggregate uses jobs_a4. This bridge does not reclassify results, prove GPU behavior, or waive changed build/module identity.',
        prior_art='Dependency-directed content hashing: Make, Feldman 1979; Nix, Dolstra et al. 2004 (unverified leads). New manual SP3 kind audit, no general semantic equivalence claim.'))
    (ART/(BRIDGE+'.sha256')).write_text(sha(ART/BRIDGE)+'\n')
    print(json.dumps(dict(status='SEALED',path=str(ART/BRIDGE),sha256=sha(ART/BRIDGE),file_deltas=len(deltas))))


if __name__=='__main__':main()
