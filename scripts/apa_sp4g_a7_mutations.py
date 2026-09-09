"""Prior art: DeMillo/Lipton/Sayward (1978), SP4G (2026) source-copy
mutation testing. No live source replacement or blind-adversary claim.
"""
import subprocess
from apa_sp4g_common import *

MUTANTS=[
 ('rail_boundary','model','if now >= self.worker_end:', 'if now > self.worker_end:', 'test_a7_deadline_inclusive'),
 ('oom_classifier','model','if isinstance(error, RuntimeError) and re.search(', 'if isinstance(error, Exception) or re.search(', 'test_a7_other_errors'),
 ('rail_as_oom','common',"if r['outcome'] in ('OOM', 'NON_FIT_AFTER_OOM'):", "if r['outcome'] in ('OOM', 'NON_FIT_AFTER_OOM', 'RAIL'):", 'test_a7_only_same_arm'),
 ('fingerprint_ignored','common',"or j.get('registration_sha256') != REG_SHA or j.get('fingerprint') != fp", "or j.get('registration_sha256') != REG_SHA or False", 'test_a7_receipt_rejects'),
 ('kv_slope','common','global_bytes = 16384 * S','global_bytes = 8192 * S','test_a7_cache_formula'),
 ('prefill_shortened','model',"owner.prefill(ids[:c['S']])","owner.prefill(ids[:c['S']-1])",'test_a7_entire_prefix'),
 ('delta_wrong','model',"delta=3.0 if c['arm']=='C' else None","delta=4.0 if c['arm']=='C' else None",'test_a7_frozen_delta'),
 ('pool_off','base','tc.set_alloc_pooling(True)','tc.set_alloc_pooling(False)','test_a7_pool_on_before_load')]


def main():
    directory=A/'mutations_a7';directory.mkdir(exist_ok=True);manifest=[]
    for name,module,old,new,test in MUTANTS:
        src=R/('scripts/apa_sp4g_model.py' if module=='base' else f'scripts/apa_sp4g_a7_{module}.py')
        source=src.read_text()
        if source.count(old)!=1:raise Red('A7_MUTATION_SITE_NOT_UNIQUE: '+name)
        p=directory/(name+'.py')
        with p.open('x') as f:f.write(source.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(p),source=str(src.relative_to(R)),
                             source_sha256=sha(src),mutant_sha256=sha(p),test=test))
    publish(directory/'manifest.json',dict(threshold=.8,mutants=manifest));rows=[]
    for m in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 APA_SP4G_A7_MUTANT=m['module']+':'+m['path'])
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            'tensor_cuda/tests/test_apa_sp4g_a7.py','-k',m['test']],cwd=R,env=env,capture_output=True,text=True)
        p=directory/(m['name']+'.log')
        with p.open('x') as f:f.write(run.stdout+run.stderr)
        error=run.returncode not in (0,1) or 'ERROR ' in run.stdout
        rows.append(dict(name=m['name'],killed=run.returncode==1 and 'FAILED ' in run.stdout and not error,
                         error=error,returncode=run.returncode,log=str(p.relative_to(R)),log_sha256=sha(p)))
    n=sum(not r['error'] for r in rows);k=sum(r['killed'] for r in rows)
    result=dict(killed=k,nonerror=n,rate=k/n if n else 0,threshold=.8,rows=rows)
    result['status']='PASS' if n==len(rows) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result);print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('A7_MUTATION_GATE_FAILED')


if __name__=='__main__':main()
