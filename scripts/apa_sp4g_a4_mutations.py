"""A4 source-copy mutations, live sources never replaced.
Prior art: DeMillo/Lipton/Sayward (1978), SP3 (2026); author verification only.
"""
import os,subprocess
from apa_sp4g_common import *
MUTANTS=[
 ('cast_dropped','model',"[t.astype('float32') for t in (q,k,kq,v)]","[t for t in (q,k,kq,v)]",'test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm'),
 ('input_dtype_ignored','model',"inputs=pin(dict(q=qf,k=kf,kq=kqf,v=vf),'float32')","inputs={}",'test_a4_dtype_pin_rejects_silent_failed_cast_before_sp'),
 ('output_dtype_ignored','model',"native=pin(dict(out=out),'float32')","native={}",'test_a4_dtype_pin_rejects_native_bf16_output'),
 ('return_wrong_arm','model',"if arm=='D32':","if arm=='A32':",'test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm'),
 ('gate_relaxed','common',"abs(d-a)<=reg['tolerance']","abs(d-a)<=reg['tolerance']*1000",'test_a4_gate_inclusive_tolerance_finite_and_dtype'),
 ('gate_nan','common','math.isfinite(d) and','True or','test_a4_gate_inclusive_tolerance_finite_and_dtype'),
 ('coverage_gap','model',"if r['lo']!=at or r['n']<=0:","if False or r['n']<=0:",'test_a4_coverage_rejects_gaps_and_empty'),
 ('source_fingerprint_ignored','common',"or j.get('fingerprint')!=fingerprint(c)",'or False','test_a4_receipt_fingerprint_dependency_payload_and_red_rejected'),
]
def main():
    directory=A/('mutations_a4_v2' if '--v2' in sys.argv else 'mutations_a4');directory.mkdir(exist_ok=True);manifest=[]
    for name,module,old,new,test in MUTANTS:
        src=R/f'scripts/apa_sp4g_a4_{module}.py';source=src.read_text()
        if source.count(old)!=1:raise Red('A4_MUTATION_SITE_NOT_UNIQUE: '+name)
        p=directory/(name+'.py')
        with p.open('x') as f:f.write(source.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(p),source=str(src.relative_to(R)),source_sha256=sha(src),mutant_sha256=sha(p),test=test))
    publish(directory/'manifest.json',dict(threshold=.8,mutants=manifest));rows=[]
    for m in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',APA_SP4G_A4_MUTANT=m['module']+':'+m['path'])
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tensor_cuda/tests/test_apa_sp4g_a4.py','-k',m['test']],cwd=R,env=env,capture_output=True,text=True,timeout=30)
        p=directory/(m['name']+'.log')
        with p.open('x') as f:f.write(run.stdout+run.stderr)
        error=run.returncode not in (0,1) or 'ERROR ' in run.stdout
        rows.append(dict(name=m['name'],killed=run.returncode==1 and 'FAILED ' in run.stdout and not error,error=error,returncode=run.returncode,log=str(p.relative_to(R)),log_sha256=sha(p)))
    n=sum(not r['error'] for r in rows);k=sum(r['killed'] for r in rows)
    result=dict(killed=k,nonerror=n,rate=k/n if n else 0,threshold=.8,rows=rows)
    result['status']='PASS' if n==len(rows) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result);print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('A4_MUTATION_GATE_FAILED')
if __name__=='__main__':main()
