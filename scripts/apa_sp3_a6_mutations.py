"""Disposable A6 mutants. Prior art: A5 injection harness, mutation testing
DeMillo/Lipton/Sayward 1978 (unverified lead: Hints on Test Data Selection).
No production source mutation; each injected module lives in its own process.
"""
import json
import subprocess
import sys
from apa_sp3_common import ART,ROOT,publish,sha

MUTANTS=[
 ('pool_after_load','apa_sp3_a6_decode',"tc.set_alloc_pooling(self.cfg['pool_before_load'])",'tc.set_alloc_pooling(False)','test_clean_worker_load_cache_copies_and_timing',1),
 ('force_layer_hook','apa_sp3_a6_decode',"if self.cfg['attention_wrapper']:",'if True:','test_decode_clean_no_attention_hook_installed',1),
 ('force_full_copy','apa_sp3_a6_decode',"if cfg['full_logits_host_copy'] and",'if True and','test_clean_worker_load_cache_copies_and_timing',1),
 ('shorten_steps','apa_sp3_a6_decode','range(S,S+steps):','range(S,S+steps-1):','test_clean_worker_load_cache_copies_and_timing',1),
 ('drop_cache','apa_sp3_a6_decode',"if cfg['use_cache']:",'if False:','test_clean_worker_load_cache_copies_and_timing',1),
 ('rail_boundary','apa_sp3_a6_decode',"estimate>=cell['worker_timeout_s']","estimate>cell['worker_timeout_s']",'test_clean_32k_plan_rail',2),
 ('raw_p5','apa_sp3_a6_report',"ids=[f'decode_clean_b4_{a}_{S}' for a in 'BC']","ids=[f'decode_pool_b4_{a}_{S}' for a in 'BC']",'test_clean_p5_no_legacy_or_8192_substitution',1),
]


def main():
    directory=ART/'a6_mutations';directory.mkdir(exist_ok=False)
    test=ROOT/'tensor_cuda/tests/test_apa_sp3_a6.py'
    pins={m:sha(ROOT/'scripts'/(m+'.py')) for _,m,*_ in MUTANTS}
    publish(directory/'registration.json',dict(threshold=.8,invalid_allowed=0,mutants=MUTANTS,
            source_sha256=pins,accepted_suite_sha256=sha(ART/'a6_cpu_accepted.log'),prior_art=__doc__))
    baseline=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(test)],capture_output=True,text=True,timeout=45)
    (directory/'baseline.log').write_text(baseline.stdout+baseline.stderr)
    if baseline.returncode:raise RuntimeError('A6 mutation baseline RED')
    results=[]
    for name,module,before,after,case,count in MUTANTS:
        source=(ROOT/'scripts'/(module+'.py')).read_text()
        if source.count(before)!=count:raise RuntimeError('mutation site drift: '+name)
        dest=directory/(name+'.py');dest.write_text(source.replace(before,after))
        code=(f'import sys,importlib.util,pytest;sys.path.insert(0,{str(ROOT/"scripts")!r})\n'
              'import apa_sp3_gpu\n'
              f'spec=importlib.util.spec_from_file_location({module!r},{str(dest)!r})\n'
              f'm=importlib.util.module_from_spec(spec);sys.modules[{module!r}]=m;spec.loader.exec_module(m)\n'
              f'raise SystemExit(pytest.main(["-q","-p","no:cacheprovider",{str(test)+"::"+case!r}]))')
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
        (directory/(name+'.log')).write_text(p.stdout+p.stderr)
        killed=p.returncode==1 and any(s in p.stdout for s in ('AssertionError','DID NOT RAISE','A6_DIAGNOSTIC_HOOK_INSTALLED','A6_TIMING_PIN','A6_CACHE_GEOMETRY'))
        results.append(dict(name=name,exit_code=p.returncode,killed=killed,
                            invalid=p.returncode not in (0,1) or p.returncode==1 and not killed))
    valid=[r for r in results if not r['invalid']]
    rate=sum(r['killed'] for r in valid)/len(valid) if valid else 0
    unchanged=all(sha(ROOT/'scripts'/(m+'.py'))==h for m,h in pins.items())
    r=dict(status='PASS' if rate>=.8 and len(valid)==len(results) and unchanged else 'RED',
           kill_rate=rate,mutants=results,original_sources_unchanged=unchanged,
           evidence_class='author mutation testing; no GPU or independent blind verification')
    publish(directory/'results.json',r);print(json.dumps(r))
    return 0 if r['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
