"""Registered replacement for error mutant; prior art: A5/A6 mutation harness
and DeMillo/Lipton/Sayward 1978, unverified lead: Hints on Test Data Selection.
Original invalid lane retained; no weakened threshold or production edit.
"""
import json
import subprocess
import sys
from apa_sp3_common import ART,ROOT,publish,read,sha


def main():
    original=ART/'a6_mutations'
    directory=ART/'a6_mutations_p5_followup';directory.mkdir(exist_ok=False)
    module='apa_sp3_a6_report';source_path=ROOT/'scripts'/(module+'.py')
    test=ROOT/'tensor_cuda/tests/test_apa_sp3_a6.py'
    case=str(test)+'::test_clean_p5_no_legacy_or_8192_substitution'
    before='valid_measurement(jobs.get(n,{}),by[n])'
    after="valid_measurement(jobs.get(n.replace('decode_clean','decode_pool'),{}),by[n])"
    publish(directory/'registration.json',dict(threshold=.8,invalid_allowed=0,
            replaces='raw_p5 error mutant only; original result remains RED',
            original_results_sha256=sha(original/'results.json'),source_sha256=sha(source_path),
            test_sha256=sha(test),before=before,after=after,
            reason='Keep registered clean cell lookup valid, substitute only receipt namespace, so intended P5 defect is executable.',
            prior_art=__doc__))
    baseline=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',case],capture_output=True,text=True,timeout=30)
    (directory/'baseline.log').write_text(baseline.stdout+baseline.stderr)
    if baseline.returncode:raise RuntimeError('replacement baseline RED')
    source=source_path.read_text();assert source.count(before)==1
    dest=directory/'raw_p5.py';dest.write_text(source.replace(before,after))
    code=(f'import sys,importlib.util,pytest;sys.path.insert(0,{str(ROOT/"scripts")!r})\n'
          'import apa_sp3_gpu\n'
          f's=importlib.util.spec_from_file_location({module!r},{str(dest)!r})\n'
          f'm=importlib.util.module_from_spec(s);sys.modules[{module!r}]=m;s.loader.exec_module(m)\n'
          f'raise SystemExit(pytest.main(["-q","-p","no:cacheprovider",{case!r}]))')
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    (directory/'raw_p5.log').write_text(p.stdout+p.stderr)
    killed=p.returncode==1 and 'AssertionError' in p.stdout
    rows=[r for r in read(original/'results.json')['mutants'] if r['name']!='raw_p5']
    rows.append(dict(name='raw_p5_receipt_namespace',killed=killed,invalid=p.returncode not in (0,1) or p.returncode==1 and not killed,exit_code=p.returncode))
    unchanged=all(sha(ROOT/'scripts'/(m+'.py'))==h for m,h in read(original/'registration.json')['source_sha256'].items())
    rate=sum(r['killed'] for r in rows)/len(rows)
    result=dict(status='PASS' if rate>=.8 and not any(r['invalid'] for r in rows) and unchanged else 'RED',
                kill_rate=rate,mutants=rows,original_sources_unchanged=unchanged,
                original_error_mutant='RETAINED: a6_mutations/results.json, not included as valid defect',
                evidence_class='six original valid mutants plus preregistered executable P5 replacement; author CPU baseline only')
    publish(directory/'results.json',result);print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
