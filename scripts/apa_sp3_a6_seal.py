"""Create-only source seal. Prior art: A5 manual endpoint bridge (2026),
dependency hashing. New exact reviewed transition, no equivalence algorithm.
"""
import ast
import difflib
import json
from apa_sp3_common import ART,ROOT,REG_SHA,read,sha,publish
from apa_sp3_a4_provenance import closure
from apa_sp3_a6_registry import KINDS,MANIFEST,manifest
from apa_sp3_a6_provenance import BRIDGE


def main():
    before=read(ART/'a6_before.json');baseline=ART/'a6_baseline'
    def fn(path,name):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    old=fn(baseline/'scripts/apa_sp3_gpu.py','execute')
    new=fn(ROOT/'scripts/apa_sp3_gpu.py','execute')
    assert ast.unparse(new.body[1].test)=="kind in ('decode_clean', 'decode_repro', 'decode_bisect')"
    new.body.pop(1)
    assert ast.dump(old)==ast.dump(new)
    for name in ('base_cells','g0_guard','work'):
        assert ast.dump(fn(baseline/'scripts/apa_sp3_gpu.py',name))==ast.dump(fn(ROOT/'scripts/apa_sp3_gpu.py',name))
    for p in ('scripts/apa_sp3_model.py','scripts/apa_sp3_a5_decode.py'):
        assert sha(ROOT/p)==before['files'][p]
    kinds=sorted(before['closures'])
    reviews={
        'scripts/apa_sp3_gpu.py':'Only additive A6 overlay/dispatch and default selection; old execute minus A6 branch, worker, G0 and base registry AST identical.',
        'scripts/apa_sp3_control.py':'New-kind-only clean preflight. Interposer policy exactly reproduces old-kind behavior; new query for shell. Dependencies/rails unchanged.',
        'scripts/apa_sp3_lead_gpu.sh':'Add current A6 root; query registered interposer bool. Every prior kind retains same preload policy, lease/worker/timeout/receipt logic.',
        'scripts/apa_sp3_a4_provenance.py':'Validation-only exact A6 bridge extension and additive import dependencies; earlier projection and numerical closures preserved.',
        'scripts/apa_sp3_a6_provenance.py':'New validation-only exact reviewed endpoint projection; unknown hashes and wrong bridge reject.',
        'scripts/apa_sp3_a6_registry.py':'Create-only additive cells and interposer policy; prior registry records unchanged; default pool decode superseded by clean.',
        'artifacts/apa_sp3/amendment_011_decode_clean.json':'Immutable additive A6 registration; earlier registrations untouched.',
    }
    transitions={}
    for p in sorted({p for k in kinds for p in closure(dict(kind=k))}):
        if p=='artifacts/apa_sp3/build/manifest.json':continue
        old=before['files'].get(p);new=sha(ROOT/p)
        if old==new:continue
        assert p in reviews,p
        transitions[p]=dict(before_sha256=old,after_sha256=new,unchanged_kinds=kinds,review=reviews[p])
    diff=[]
    for p,h in before['files'].items():
        if sha(ROOT/p)!=h:
            diff.extend(difflib.unified_diff((baseline/p).read_text().splitlines(True),(ROOT/p).read_text().splitlines(True),fromfile='a5/'+p,tofile='a6/'+p))
    (ART/'a6_source_delta.patch').write_text(''.join(diff))
    publish(ART/BRIDGE,dict(immutable=True,registration_sha256=REG_SHA,
            order_sha256=manifest()['order_sha256'],cell_amendment_sha256=sha(ART/MANIFEST),
            parent_effective_sha256=before['effective_a5_sha256'],file_transitions=transitions,
            per_kind_import_closure={k:closure(dict(kind=k)) for k in kinds+sorted(KINDS)},
            source_gate=dict(legacy_execute_AST='IDENTICAL minus A6 branch',base_registry_g0_worker_AST='IDENTICAL',
                             legacy_model_and_pool_worker_bytes='IDENTICAL',source_diff_sha256=sha(ART/'a6_source_delta.patch')),
            affected_receipts=[],receipt_policy='All old source endpoints reviewed. No runtime build waiver: seat build not merged; preserve lead build. Original RED remains RED; unknown changes reject.',
            prior_art='A5 exact endpoint hash projection reused; Make Feldman 1979/Nix Dolstra 2004 unverified leads to check; no general semantic equivalence proof'))
    (ART/(BRIDGE+'.sha256')).write_text(sha(ART/BRIDGE)+'\n')
    print(json.dumps(dict(status='SEALED',sha256=sha(ART/BRIDGE),transitions=len(transitions))))


if __name__=='__main__':main()
