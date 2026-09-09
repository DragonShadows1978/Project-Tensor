"""A3 source-copy mutation gates, never replacing live code.
Prior art: DeMillo/Lipton/Sayward (1978), mutation testing (unverified lead:
Hints on Test Data Selection); SP3 (2026) fail-closed baseline. No novelty.
"""
import os,subprocess
from apa_sp4g_common import *

MUTANTS = [
    ('frob_denominator','math3','np.linalg.norm(b.astype(np.float64).ravel())','np.linalg.norm(a.astype(np.float64).ravel())','test_a3_frobenius_uses_reference_norm_and_both_bounds'),
    ('or_tolerance','math3','absmax <= MAX_ABS and relative <= REL_FROB','absmax <= MAX_ABS or relative <= REL_FROB','test_a3_frobenius_uses_reference_norm_and_both_bounds'),
    ('sink_dropped','math3','denom += np.exp(-maximum)','denom += 0','test_a3_dense_analytic_zero_scores_mask_sink_and_shared_heads'),
    ('scale_ignored','math3','np.float32(scale) * (q[b, h] @ k[b, kvh].T)','np.float32(1.) * (q[b, h] @ k[b, kvh].T)','test_a3_dense_nontrivial_scale_and_group_mapping_independent_scalar_reference'),
    ('group_mapping','math3','kvh = h // (H // k.shape[1])','kvh = h % k.shape[1]','test_a3_dense_nontrivial_scale_and_group_mapping_independent_scalar_reference'),
    ('two_changes_accepted','math3',"sum(variant[k] != NOMINAL[k] for k in NOMINAL) != 1","sum(variant[k] != NOMINAL[k] for k in NOMINAL) not in (1,2)",'test_a3_classifier_requires_one_change_both_calls_and_dense_confirmation'),
    ('native_tolerance','model',"if not comparison['bitwise']:","if not comparison['fp32_agreement']:",'test_a3_bitwise_includes_signed_zero_and_no_tolerance'),
    ('refine_mask_ignored','model','if not np.array_equal(observed, expected):','if False:','test_a3_refine_all_requires_full_eligible_mask'),
    ('source_hash_ignored','common',"or j.get('fingerprint') != fingerprint(c)",'or False','test_a3_fingerprint_dependency_and_payload_rejections'),
    ('payload_hash_ignored','common',"if sha(R / f['path']) != f['sha256']:",'if False:','test_a3_fingerprint_dependency_and_payload_rejections'),
]

def main():
    directory=A/('mutations_a3_final' if '--final' in sys.argv else 'mutations_a3');directory.mkdir(exist_ok=True)
    manifest=[]
    for name,module,old,new,test in MUTANTS:
        stem='math' if module=='math3' else module
        src=R/f'scripts/apa_sp4g_a3_{stem}.py';text=src.read_text()
        if text.count(old)!=1:raise Red('A3_MUTATION_SITE_NOT_UNIQUE: '+name)
        path=directory/(name+'.py')
        with path.open('x') as f:f.write(text.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(path),source=str(src.relative_to(R)),source_sha256=sha(src),mutant_sha256=sha(path),test=test))
    publish(directory/'manifest.json',dict(threshold=.8,mutants=manifest))
    rows=[]
    for mutant in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',APA_SP4G_A3_MUTANT=mutant['module']+':'+mutant['path'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tensor_cuda/tests/test_apa_sp4g_a3.py','-k',mutant['test']],cwd=R,env=env,capture_output=True,text=True,timeout=30)
        log=directory/(mutant['name']+'.log')
        with log.open('x') as f:f.write(run.stdout+run.stderr)
        error=run.returncode not in (0,1) or 'ERROR ' in run.stdout
        rows.append(dict(name=mutant['name'],killed=run.returncode==1 and 'FAILED ' in run.stdout and not error,
                         error=error,returncode=run.returncode,log=str(log.relative_to(R)),log_sha256=sha(log)))
    n=sum(not row['error'] for row in rows);kills=sum(row['killed'] for row in rows)
    result=dict(killed=kills,nonerror=n,rate=kills/n if n else 0,threshold=.8,rows=rows,
                evidence_class='author copied-source mutations, not blind verification')
    result['status']='PASS' if n==len(rows) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('A3_MUTATION_GATE_FAILED')

if __name__=='__main__':main()
