"""Independent copied-source defect probes after passing baseline.
Prior art: DeMillo/Lipton/Sayward1978 mutation testing (unverified lead: Hints on
Test Data Selection); SP3 mutation receipts2026. No live sources mutated.
"""
import os,subprocess
from apa_sp4g_common import *
MUTANTS=[
 ('target_boundary','model','pos=S-scored-1','pos=S-scored','test_exact_target_boundaries_and_no_final_target_outside_input'),
 ('wrong_target','model','x[np.arange(len(y)),y]','x[np.arange(len(y)),np.roll(y,1)]','test_nll_matches_independent_logaddexp_fp64_oracle'),
 ('tail_denominator','metrics','w[~m].sum()/w.sum()','w[~m].sum()/w[~m].sum()','test_tail_all_key_normalization_and_max_ratio_distinct'),
 ('percentile','metrics','(.99,.999)','(.99,.99)','test_population_percentiles_do_not_average_band_percentiles'),
 ('margin_term','metrics','math.log(100)+2*eq','math.log(100)+0*eq','test_eq_uses_both_lengths_and_upward_margin'),
 ('relaxed_match','gpu',"abs(r['fraction']-target)<=.01","abs(r['fraction']-target)<=.02",'test_match_outside_tolerance_does_not_carry'),
 ('reverse_bracket','gpu',"if r['fraction']<target:","if r['fraction']>target:",'test_bisection_direction_and_global_delta'),
 ('stale_fingerprint','common',"(j.get('fingerprint')!=fingerprint(c) and not legacy_compatible(j))",'False','test_receipt_stale_source_rejected'),
]
def main():
    label='mutations' if len(sys.argv)==1 else 'mutations_'+sys.argv[1]
    if not label.replace('_','').isalnum():raise Red('INVALID_MUTATION_SET_NAME')
    directory=A/label;directory.mkdir(exist_ok=True)
    manifest=[]
    for name,module,old,new,test in MUTANTS:
        source=R/f'scripts/apa_sp4g_{module}.py';s=source.read_text()
        if old not in s:raise Red('MUTATION_SITE_MISSING: '+name)
        p=directory/f'{name}.py'
        with p.open('x') as f:f.write(s.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(p),source_sha256=sha(source),mutant_sha256=sha(p),test=test))
    publish(directory/'manifest.json',dict(registration_sha256=REG_SHA,threshold=.8,mutants=manifest))
    results=[]
    for m in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',APA_SP4G_MUTANT=m['module']+':'+m['path'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        args=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tensor_cuda/tests/test_apa_sp4g.py','-k',m['test']]
        r=subprocess.run(args,cwd=R,env=env,capture_output=True,text=True,timeout=30)
        log=directory/f'{m["name"]}.log'
        with log.open('x') as f:f.write(r.stdout+r.stderr)
        killed=r.returncode==1 and 'FAILED ' in r.stdout and 'ERROR ' not in r.stdout
        error=r.returncode not in (0,1) or 'ERROR ' in r.stdout
        results.append(dict(name=m['name'],returncode=r.returncode,killed=killed,error=error,log=str(log.relative_to(R)),log_sha256=sha(log)))
    nonerror=sum(not r['error'] for r in results);killed=sum(r['killed'] for r in results)
    result=dict(evidence_class='author mutation baseline, not blind review',nonerror=nonerror,killed=killed,rate=killed/nonerror if nonerror else 0,threshold=.8,results=results)
    result['status']='PASS' if nonerror==len(MUTANTS) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result);print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('MUTATION_GATE_FAILED')
if __name__=='__main__':main()
