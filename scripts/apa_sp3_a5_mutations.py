"""Registered A5 mutants on disposable source copies only.

Prior art: mutation testing, DeMillo/Lipton/Sayward 1978, unverified lead
'Hints on Test Data Selection'. Reuse a4 injection harness, new defect sites.
"""
import json
import subprocess
import sys
from apa_sp3_common import ART, ROOT, publish, sha

MUTANTS = [
    ('drop_pool_enable','apa_sp3_a5_decode','model.tc.set_alloc_pooling(True)',
     'model.tc.set_alloc_pooling(False)','test_decode_pool_worker_pool_state_pin'),
    ('shorten_measured_steps','apa_sp3_model','range(S, S + 32):',
     'range(S, S + 31):','test_decode_pool_worker_pool_state_pin'),
    ('relax_planning_boundary','apa_sp3_a5_decode',"estimate >= cell['worker_timeout_s']",
     "estimate > cell['worker_timeout_s']",'test_pool_32k_measured_plan_rail_boundary'),
    ('default_32k_capture','apa_sp3_a5_registry','include_32k_captures or not',
     'True or not','test_default_capture_gate_and_commands_dependency_order'),
    ('raw_decode_P5','apa_sp3_a5_report',"jobs.get(f'decode_pool_b4_{arm}_32768', {})",
     "jobs.get(f'decode_b4_{arm}_32768', {})",'test_p5_uses_only_pool_on_primary_valid_measurements'),
]


def main():
    directory=ART/'a5_mutations'
    directory.mkdir(exist_ok=False)
    test=ROOT/'tensor_cuda/tests/test_apa_sp3_a5.py'
    publish(directory/'registration.json',dict(threshold=.8,no_invalid_mutants=True,
            mutants=MUTANTS,source_sha256={m:sha(ROOT/'scripts'/(m+'.py')) for _,m,_,_,_ in MUTANTS},
            accepted_suite_sha256=sha(ART/'a5_cpu_accepted.log'),prior_art=__doc__))
    baseline=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(test)],
                            capture_output=True,text=True,timeout=45)
    (directory/'baseline.log').write_text(baseline.stdout+baseline.stderr)
    if baseline.returncode:
        raise RuntimeError('A5 mutation baseline RED')
    results=[]
    for name,module,before,after,case in MUTANTS:
        source=(ROOT/'scripts'/(module+'.py')).read_text()
        if source.count(before)!=1:
            raise RuntimeError('mutation site drift: '+name)
        dest=directory/(name+'.py')
        dest.write_text(source.replace(before,after))
        code=(f'import sys, importlib.util, pytest; sys.path.insert(0,{str(ROOT/"scripts")!r})\n'
              'import apa_sp3_gpu\n'
              f'spec=importlib.util.spec_from_file_location({module!r},{str(dest)!r})\n'
              f'm=importlib.util.module_from_spec(spec);sys.modules[{module!r}]=m;spec.loader.exec_module(m)\n'
              f'raise SystemExit(pytest.main(["-q","-p","no:cacheprovider",{str(test)+"::"+case!r}]))')
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
        (directory/(name+'.log')).write_text(p.stdout+p.stderr)
        killed=p.returncode==1 and any(s in p.stdout for s in ('AssertionError','DID NOT RAISE'))
        results.append(dict(name=name,exit_code=p.returncode,killed=killed,
                            invalid=p.returncode not in (0,1) or p.returncode==1 and not killed))
    rate=sum(r['killed'] for r in results)/len(results)
    result=dict(status='PASS' if rate>=.8 and not any(r['invalid'] for r in results) else 'RED',
                kill_rate=rate,mutants=results,production_edits=False,
                original_sources_unchanged=all(sha(ROOT/'scripts'/(m+'.py'))==digest
                    for m,digest in json.loads((directory/'registration.json').read_text())['source_sha256'].items()))
    publish(directory/'results.json',result)
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
