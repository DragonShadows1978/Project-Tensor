"""Seal reviewed A1 endpoints, before CPU gates.

Prior art: SP3 a4 (2026) exact source transition audit and Make/Feldman1979
dependency fingerprints. No new algorithm; kernel-only legacy acceptance.
"""
import ast,difflib,json
from apa_sp4g_common import *
from apa_sp4g_registry import cells,fallback_cells,by_id
from apa_sp4g_a1_provenance import BRIDGE

def main():
    before=read(A/'a1_before.json');baseline=A/'a1_baseline'
    def segment(path,name):
        s=path.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(s,node)
    checks={}
    for module,names in {'gpu':['kernel512'], 'common':['sha','read','registration','verify_sources','tokens','verify_weight','build_check','load_runtime']}.items():
        path=f'scripts/apa_sp4g_{module}.py'
        for name in names:
            assert segment(baseline/path,name)==segment(R/path,name),(path,name)
            checks[f'{module}.{name}']='BYTE_IDENTICAL'
    def kernel_branch(path):
        tree=ast.parse(segment(path,'execute'))
        return ast.dump(tree.body[0].body[1],include_attributes=False)
    assert kernel_branch(baseline/'scripts/apa_sp4g_gpu.py')==kernel_branch(R/'scripts/apa_sp4g_gpu.py')
    verify_sources();build_check()
    kernel=read(A/'jobs/kernel512.json')
    assert kernel['cell']==by_id()['kernel512'] and kernel['status']=='PASS'
    for p,h in kernel['fingerprint'].items():assert before['files'].get(p,h)==h
    for p in (A/'jobs').glob('*.json'):
        assert sha(p)==before['files'][str(p.relative_to(R))]
    diffs=[];deltas={}
    paths=set(before['files']) | {p for c in cells()+fallback_cells() for p in fingerprint(c)}
    for p in sorted(paths):
        if not (R/p).is_file():continue
        old=before['files'].get(p);new=sha(R/p)
        if old==new:continue
        deltas[p]=dict(before_sha256=old,after_sha256=new,
                      unchanged_kinds=['kernel'] if p in fingerprint(kernel['cell']) else [],
                      review='Kernel-only bridge: measurement functions, branch, inputs, build and runner unchanged; shared registry/provenance/other-kind dispatch changed. No model or margin equivalence waiver.')
        if (baseline/p).is_file() and p.endswith('.py'):
            diffs.extend(difflib.unified_diff((baseline/p).read_text().splitlines(True),(R/p).read_text().splitlines(True),fromfile='r1/'+p,tofile='a1/'+p))
    patch=A/'a1_source_delta.patch'
    with patch.open('x') as f:f.write(''.join(diffs))
    publish(A/BRIDGE,dict(schema='apa_sp4g_a1_kernel_bridge_v1',immutable=True,
        registration_sha256=REG_SHA,order_sha256=sha(R/'orders/APA_SP4G_AMENDMENT_1.md'),
        before_sha256=sha(A/'a1_before.json'),execution_amendment_sha256=sha(A/'amendment_002_a1_execution.json'),
        file_deltas=deltas,source_checks=checks,kernel_dispatch_AST='IDENTICAL',
        legacy_kernel=dict(receipt_sha256=sha(A/'jobs/kernel512.json'),cell=kernel['cell'],
                           before_fingerprint=kernel['fingerprint'],after_fingerprint=fingerprint(kernel['cell'])),
        per_kind_fingerprints={c['kind']:fingerprint(c) for c in cells()+fallback_cells()},
        source_diff_sha256=sha(patch),
        cpu_policy='Original CPU receipt remains a historical fingerprint pin. CPU_GATES_A1.json is separately mandatory at every preflight and pins the full amended execution/test closure and this bridge SHA.',
        receipt_policy='Only original PASS kernel512 crosses exact reviewed source endpoints. Original model RED retained in jobs; new model/margin receipts in jobs_a1. Unknown changes fail closed.',
        prior_art='SP3 a4(2026), Make/Feldman1979 dependency hashing. No new algorithm.'))
    with (A/(BRIDGE+'.sha256')).open('x') as f:f.write(sha(A/BRIDGE)+'\n')
    print(json.dumps(dict(status='SEALED',sha256=sha(A/BRIDGE),source_checks=checks)))

if __name__=='__main__':main()
