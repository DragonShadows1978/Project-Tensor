"""A2 independent source-copy mutants. Prior art: DeMillo/Lipton/Sayward1978
mutation testing (unverified lead: Hints on Test Data Selection), SP3 (2026).
No live sources replaced; threshold >=.80, errors never count as kills.
"""
import os,subprocess
from apa_sp4g_common import *
MUTANTS=[
 ('output_tolerance','metrics',"if not np.array_equal(out,raw['out']):","if not np.allclose(out,raw['out']):",'test_a2_replay_requires_output_and_selection_bitwise'),
 ('mask_ignored','metrics',"if not np.array_equal(mask,raw['mask']):","if False:",'test_a2_replay_requires_output_and_selection_bitwise'),
 ('bf16_lowbits','model','if not np.isfinite(x).all() or np.any(u & 65535):','if not np.isfinite(x).all():','test_a2_lossless_bf16_payload_keeps_sign_bits_and_subnormals'),
 ('population','registry',"population_rows=S-1","population_rows=S",'test_a2_cell_dag_diagnostics_order_and_actual_ppl_margins'),
 ('rail','registry',"c.get('S',0)>=16384","c.get('S',0)>16384",'test_a2_long_rows_rejected_before_model_or_preflight'),
 ('fp32_arm','model',"a if self.cell['treatment']=='A32' else d","a if self.cell['treatment']=='D32' else d",'test_a2_precision_dispatch_reuses_exact_k_and_returns_registered_arm'),
 ('source_hash','common',"j.get('fingerprint')!=fingerprint_a2(c)","False",'test_a2_new_fingerprint_unknown_transition_and_red_rejected'),
 ('scale','model','k,alpha=scale,trans_b=True','k,alpha=1.,trans_b=True','test_a2_standard_same_tensors_bottom_right_mqa_and_scale'),
]
def main():
    directory=A/'mutations_a2';directory.mkdir(exist_ok=True);manifest=[]
    for name,module,old,new,test in MUTANTS:
        src=R/f'scripts/apa_sp4g_a2_{module}.py';s=src.read_text()
        if old not in s:raise Red('A2_MUTATION_SITE_MISSING: '+name)
        p=directory/f'{name}.py'
        with p.open('x') as f:f.write(s.replace(old,new,1))
        manifest.append(dict(name=name,module=module,path=str(p),source_sha256=sha(src),mutant_sha256=sha(p),test=test))
    publish(directory/'manifest.json',dict(threshold=.8,mutants=manifest))
    rows=[]
    for m in manifest:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',APA_SP4G_A2_MUTANT=m['module']+':'+m['path'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tensor_cuda/tests/test_apa_sp4g_a2.py','-k',m['test']],cwd=R,env=env,capture_output=True,text=True,timeout=30)
        p=directory/f'{m["name"]}.log'
        with p.open('x') as f:f.write(run.stdout+run.stderr)
        error=run.returncode not in (0,1) or 'ERROR ' in run.stdout
        rows.append(dict(name=m['name'],killed=run.returncode==1 and 'FAILED ' in run.stdout and not error,error=error,returncode=run.returncode,log=str(p.relative_to(R)),log_sha256=sha(p)))
    n=sum(not x['error'] for x in rows);k=sum(x['killed'] for x in rows)
    result=dict(killed=k,nonerror=n,rate=k/n if n else 0,threshold=.8,rows=rows,evidence_class='author mutation baseline, not blind verification')
    result['status']='PASS' if n==len(rows) and result['rate']>=.8 else 'RED'
    publish(directory/'results.json',result);print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise Red('A2_MUTATION_GATE_FAILED')
if __name__=='__main__':main()
