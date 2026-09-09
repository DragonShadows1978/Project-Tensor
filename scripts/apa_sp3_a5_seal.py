"""Create-only, reviewed A5 fingerprint amendment; run once before gates.

Prior art: a4 semantic bridge seal, dependency-directed content hashing.
New: exact a4-to-a5 endpoints, never a general equivalence proof.
"""
import ast
import difflib
import json
from apa_sp3_common import ART, ROOT, REG_SHA, publish, read, sha
from apa_sp3_a4_provenance import closure
from apa_sp3_a5_registry import MANIFEST, manifest
from apa_sp3_a5_provenance import BRIDGE


def main():
    before = read(ART/'a5_before.json')
    baseline = ART/'a5_baseline'
    def function(path, name):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    old=function(baseline/'scripts/apa_sp3_gpu.py','execute')
    new=function(ROOT/'scripts/apa_sp3_gpu.py','execute')
    assert ast.unparse(new.body[1].test) == "kind == 'decode_pool'"
    new.body.pop(1)
    assert ast.dump(old)==ast.dump(new)
    for name in ('base_cells','g0_guard','work'):
        assert ast.dump(function(baseline/'scripts/apa_sp3_gpu.py',name))==ast.dump(function(ROOT/'scripts/apa_sp3_gpu.py',name))
    assert sha(ROOT/'scripts/apa_sp3_model.py')==before['files']['scripts/apa_sp3_model.py']
    kinds = sorted(before['closures'])
    all_paths = {p for kind in kinds+['decode_pool'] for p in closure(dict(kind=kind))}
    transitions = {}
    reviews = {
        'scripts/apa_sp3_gpu.py':'Only new-kind dispatch, additive registry overlay and default selector. Legacy execute, base registry, g0 guard and worker AST pinned identical after removing decode_pool branch.',
        'scripts/apa_sp3_control.py':'New-kind-only pre-lease plan/non-fit branch; original dependencies, controls and measurement paths unchanged.',
        'scripts/apa_sp3_lead_gpu.sh':'Capture preflight decision and exit only for new decode_pool NON_FIT. Legacy empty successful preflight still reaches identical lease/preload/timeout path.',
        'scripts/apa_sp3_a4_provenance.py':'Validation-only bridge extension and new pool-kind closure; projects only exact reviewed source endpoints for prior receipts.',
        'scripts/apa_sp3_a5_provenance.py':'Added dependency: validation-only exact a4/a5 source endpoint projection; no measurement arithmetic.',
        'scripts/apa_sp3_a5_registry.py':'Added dependency: immutable additive cell registration and lead default selection; existing cell records unchanged.',
        'artifacts/apa_sp3/amendment_009_decode_pool.json':'Added dependency: new cells and explicitly registered A5 policies; existing registration/order/amendments immutable.',
    }
    for p in sorted(all_paths):
        after=sha(ROOT/p)
        previous=before['files'].get(p)
        # Existing non-source closure artifacts not edited by A5.
        if p not in before['files'] and not ('a5_' in p or p.endswith(MANIFEST)):
            continue
        if previous==after:
            continue
        if p=='scripts/apa_sp3_a5_decode.py':
            continue  # new kind only; no existing receipt ever executed it
        assert p in reviews, p
        transitions[p]=dict(before_sha256=previous,after_sha256=after,
                            unchanged_kinds=kinds,review=reviews[p])
    diff=[]
    for p,digest in before['files'].items():
        if sha(ROOT/p)!=digest:
            diff.extend(difflib.unified_diff((baseline/p).read_text().splitlines(True),
                        (ROOT/p).read_text().splitlines(True),fromfile='a4/'+p,tofile='a5/'+p))
    (ART/'a5_source_delta.patch').write_text(''.join(diff))
    publish(ART/BRIDGE,dict(immutable=True,registration_sha256=REG_SHA,
            order_sha256=manifest()['order_sha256'],cell_amendment_sha256=sha(ART/MANIFEST),
            parent_effective_sha256=before['effective_a4_sha256'],
            file_transitions=transitions,
            per_kind_import_closure={k:closure(dict(kind=k)) for k in kinds+['decode_pool']},
            source_gate=dict(legacy_execute_AST='IDENTICAL after removing only new decode_pool dispatch',
                             base_registry_g0_guard_worker_AST='IDENTICAL',model_bytes='IDENTICAL',
                             source_diff_sha256=sha(ART/'a5_source_delta.patch')),
            receipt_policy='Original PASS and a4 per-kind PASS survive only reviewed source transitions. Dependency/protocol/build validation unchanged. Original RED remains RED. Future/unlisted changes reject reuse.',
            affected_receipts=[],new_kind_only=['scripts/apa_sp3_a5_decode.py'],
            outside_measurement_closures=['reporters','tests','mutation/audit/seal tools'],
            prior_art='A4 dependency-directed content hashes / manual exact transition review; Make (Feldman 1979), Nix (Dolstra et al. 2004), unverified leads to check; no general semantic equivalence proof.'))
    (ART/(BRIDGE+'.sha256')).write_text(sha(ART/BRIDGE)+'\n')
    print(json.dumps(dict(status='SEALED',path=str(ART/BRIDGE),sha256=sha(ART/BRIDGE),transitions=len(transitions))))


if __name__=='__main__':main()
