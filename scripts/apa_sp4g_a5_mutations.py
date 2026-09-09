"""A5 source-copy mutation gate. Prior art: DeMillo/Lipton/Sayward (1978),
SP4G A4 (2026). Live sources never replaced; author verification, not blind.
Unverified — lead to check: Hints on Test Data Selection (1978).
"""
import subprocess
from apa_sp4g_common import *

MUTANTS = [
    ('prefill_only','model','self.perplexity(ids,1024)','self.model(ids[None,:1023],last_token_only=True)',
     'test_a5_runs_scoring_driver_reaches_target_layer_and_block'),
    ('off_by_one','model','offset != S-L','offset != S-L+1',
     'test_a5_capture_shapes_include_actual_L64_cached_blocks'),
    ('splitk_for_cached','model','split = L == 1','split = L < S',
     'test_a5_dispatch_L64_is_prefill_even_with_cache'),
    ('completed_red_rejected','common',"j.get('status') not in ('PASS','RED')","j.get('status') not in ('PASS',)",
     'test_a5_completed_RED_accepted_without_waiving_gate'),
    ('worker_red_accepted','common',"raise Red('A5_D32_NOT_COMPLETED')","return j",
     'test_a5_worker_RED_is_not_completion'),
    ('fingerprint_ignored','common',"or j.get('fingerprint') != expected_fingerprint","or False",
     'test_a5_completed_stale_receipt_rejected'),
    ('rerun_without_disagreement','common',"['relative_frobenius'] > .001","['relative_frobenius'] >= .001",
     'test_a5_rerun_requires_all_three_and_strict_disagreement'),
    ('capture_wrong_layer','model',"selected = self.model.layers[self.cell['layer']].mixer","selected = self.model.layers[5].mixer",
     'test_a5_install_selects_registered_layer_and_restores'),
]


def main():
    directory=A/'mutations_a5';directory.mkdir(exist_ok=True)
    manifest=[]
    for name,module,old,new,test in MUTANTS:
        src=R/f'scripts/apa_sp4g_a5_{module}.py';source=src.read_text()
        if source.count(old)!=1:raise Red('A5_MUTATION_SITE_NOT_UNIQUE: '+name)
        p=directory/(name+'.py')
        with p.open('x') as f:f.write(source.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(p),source=str(src.relative_to(R)),
                             source_sha256=sha(src),mutant_sha256=sha(p),test=test))
    publish(directory/'manifest.json',dict(threshold=.8,mutants=manifest));rows=[]
    for m in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 APA_SP4G_A5_MUTANT=m['module']+':'+m['path'])
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            'tensor_cuda/tests/test_apa_sp4g_a5.py','-k',m['test']],cwd=R,env=env,capture_output=True,text=True)
        p=directory/(m['name']+'.log')
        with p.open('x') as f:f.write(run.stdout+run.stderr)
        error=run.returncode not in (0,1) or 'ERROR ' in run.stdout
        rows.append(dict(name=m['name'],killed=run.returncode==1 and 'FAILED ' in run.stdout and not error,
                         error=error,returncode=run.returncode,log=str(p.relative_to(R)),log_sha256=sha(p)))
    n=sum(not r['error'] for r in rows);k=sum(r['killed'] for r in rows)
    result=dict(killed=k,nonerror=n,rate=k/n if n else 0,threshold=.8,rows=rows)
    result['status']='PASS' if n==len(rows) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result);print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('A5_MUTATION_GATE_FAILED')


if __name__=='__main__':main()
