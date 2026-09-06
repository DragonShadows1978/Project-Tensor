#!/usr/bin/env python3
"""Registered semantic defect probes in disposable copies only.

Prior art: mutation testing (DeMillo, Lipton & Sayward, 1978; unverified lead
to check title "Hints on Test Data Selection"). House Rules section 8 requires
>=80% killed after passing author baseline. No production source is modified.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from apa_sp3_common import ART, ROOT, publish, sha


MUTANTS=[
 ('skip_inverted','apa_sp3_metrics','skipped = ~np.asarray(selected, bool)','skipped = np.asarray(selected, bool)',
  'test_mass_and_max_relative_weight_are_distinct_and_all_key_normalized'),
 ('wrong_normalizer','apa_sp3_metrics','wrel[skipped].sum() / wrel.sum()','wrel[skipped].sum() / len(wrel)',
  'test_mass_and_max_relative_weight_are_distinct_and_all_key_normalized'),
 ('off_by_one_targets','apa_sp3_model','targets = ids[-512:]','targets = ids[-513:-1]',
  'test_last512_scores_exactly_last512_targets'),
 ('downward_margin','apa_sp3_common','np.nextafter(f, np.float32(np.inf))','np.nextafter(f, np.float32(-np.inf))',
  'test_margin_rounds_up_and_never_uses_percentile'),
 ('stale_resume','apa_sp3_common',"j.get('fingerprint') != fingerprint()","False",
  'test_stale_or_red_receipt_is_not_resumable'),
 ('swapped_scale_delta','apa_sp3_model','q, k, kq, v, scale, self.delta,','q, k, kq, v, self.delta, scale,',
  'test_sp_binding_parameter_order_and_no_wrong_arm_dispatch'),
]


def main():
    tag=sys.argv[1] if len(sys.argv)>1 else 'initial'
    if tag not in ('initial','final','protocol2'):
        raise RuntimeError('usage: apa_sp3_mutations.py [initial|final|protocol2]')
    folder=ART/'mutations'/tag;folder.mkdir(parents=True,exist_ok=False)
    # Separate immutable seed registration, before baseline/mutants in this run.
    publish(ART/f'mutation_manifest_{tag}.json',{
        'gate':'author mutation baseline','threshold':.8,
        'sources':{m:sha(ROOT/'scripts'/(m+'.py')) for _,m,_,_,_ in MUTANTS},
        'defects':[{'name':n,'module':m,'before':b,'after':a,'test':t} for n,m,b,a,t in MUTANTS]})
    test=ROOT/'tensor_cuda/tests/test_apa_sp3.py'
    baseline=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(test)],
                            capture_output=True,text=True,timeout=60)
    if baseline.returncode:
        raise RuntimeError('mutation baseline failed: '+baseline.stdout+baseline.stderr)
    result=[]
    for name,module,before,after,case in MUTANTS:
        source=(ROOT/'scripts'/(module+'.py')).read_text()
        if source.count(before)!=1:
            raise RuntimeError('mutation site drift: '+name)
        path=folder/(name+'.py');path.write_text(source.replace(before,after))
        code=("import sys,importlib.util,pytest; sys.path.insert(0,"+repr(str(ROOT/'scripts'))+")\n"+
              "s=importlib.util.spec_from_file_location("+repr(module)+","+repr(str(path))+");"+
              "m=importlib.util.module_from_spec(s);sys.modules["+repr(module)+"]=m;s.loader.exec_module(m)\n"+
              "raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',"+repr(str(test)+'::'+case)+"]))")
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
        (folder/(name+'.log')).write_text(r.stdout+r.stderr)
        # pytest rc=1 must show a real assertion failure, not import/runtime error.
        killed=r.returncode==1 and ('AssertionError' in r.stdout or 'DID NOT RAISE' in r.stdout)
        result.append(dict(name=name,exit_code=r.returncode,killed=killed,
                           invalid=r.returncode not in (0,1) or (r.returncode==1 and not killed),
                           mutant_sha256=sha(path),test=case))
    valid=[r for r in result if not r['invalid']]
    rate=sum(r['killed'] for r in valid)/len(valid) if valid else 0
    receipt=dict(status='PASS' if rate>=.8 and len(valid)==len(MUTANTS) else 'RED',
                 baseline='PASS',kill_rate=rate,nonerror_mutants=len(valid),mutants=result,
                 source_mutations='copies only; production untouched',evidence_class='unit test; author mutation baseline')
    publish(ART/f'mutation_results_{tag}.json',receipt)
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
