"""A6 source-copy mutation gate. Prior art: DeMillo/Lipton/Sayward (1978),
SP4G A4 (2026). Live sources never replaced; author verification, not blind.
Unverified — lead to check: Hints on Test Data Selection (1978).
"""
import subprocess
from apa_sp4g_common import *

MUTANTS = [
    ('floor_relaxed','common','inside = abs(delta) <= 2.56','inside = abs(delta) <= 3.0',
     'test_a6_floor_inclusive_and_direction'),
    ('floor_nan','common','if not all(math.isfinite(x) and x > 0 for x in (left, right)):', 'if False:',
     'test_a6_floor_nonfinite_rejected'),
    ('flat_accepted','common',"C_E_unblocked=outcome == 'AMPLIFIED'","C_E_unblocked=True",
     'test_a6_flat_and_intermediate_stop'),
    ('fingerprint_ignored','common',"or j.get('fingerprint') != expected_fingerprint","or False",
     'test_a6_fingerprint_red_payload_and_precision_pins_rejected'),
    ('coverage_dropped','common',"r.get('rows') != 2047","False",
     'test_a6_profile_requires_full_population_and_finite'),
    ('precision_wrong_arm','model','super().__init__(cell)',"super().__init__(dict(cell, arm='D32'))",
     'test_a6_A32_capture_uses_standard_precision_and_restores'),
    ('calibration_relaxed','gpu',"abs(r['fraction']-target) <= .01","abs(r['fraction']-target) <= .1",
     'test_a6_calibration_match_uses_real_B_fraction_and_tolerance'),
    ('decode_wrapped','gpu',"return nullcontext() if kind == 'decode' else deadline_guard(owner, deadline)",
     'return deadline_guard(owner, deadline)', 'test_a6_clean_decode_never_installs_guard'),
]


def main():
    directory=A/'mutations_a6';directory.mkdir(exist_ok=True)
    manifest=[]
    for name,module,old,new,test in MUTANTS:
        src=R/f'scripts/apa_sp4g_a6_{module}.py';source=src.read_text()
        if source.count(old)!=1:raise Red('A6_MUTATION_SITE_NOT_UNIQUE: '+name)
        p=directory/(name+'.py')
        with p.open('x') as f:f.write(source.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(p),source=str(src.relative_to(R)),
                             source_sha256=sha(src),mutant_sha256=sha(p),test=test))
    publish(directory/'manifest.json',dict(threshold=.8,mutants=manifest));rows=[]
    for m in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 APA_SP4G_A6_MUTANT=m['module']+':'+m['path'])
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            'tensor_cuda/tests/test_apa_sp4g_a6.py','-k',m['test']],cwd=R,env=env,capture_output=True,text=True)
        p=directory/(m['name']+'.log')
        with p.open('x') as f:f.write(run.stdout+run.stderr)
        error=run.returncode not in (0,1) or 'ERROR ' in run.stdout
        rows.append(dict(name=m['name'],killed=run.returncode==1 and 'FAILED ' in run.stdout and not error,
                         error=error,returncode=run.returncode,log=str(p.relative_to(R)),log_sha256=sha(p)))
    n=sum(not r['error'] for r in rows);k=sum(r['killed'] for r in rows)
    result=dict(killed=k,nonerror=n,rate=k/n if n else 0,threshold=.8,rows=rows)
    result['status']='PASS' if n==len(rows) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result);print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('A6_MUTATION_GATE_FAILED')


if __name__=='__main__':main()
