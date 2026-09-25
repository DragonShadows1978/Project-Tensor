#!/usr/bin/env python3
"""Author CPU mutation lane, NOT blind verification or CUDA mutation coverage.
Prior art: mutation testing, DeMillo/Lipton/Sayward (1978), taken; unverified
date/title, lead to check "mutation testing DeMillo Lipton Sayward 1978".
Ours: focused defects in the TF32 conversion/CPU model and strict gate.
"""
import importlib.util
import inspect
import sys
import numpy as np
import pt_tf32_1 as t

sys.dont_write_bytecode=True


def main():
    spec=importlib.util.spec_from_file_location('pt_tf32_author_tests',t.ROOT/'tests/test_pt_tf32_cpu.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    def checks():
        tests.test_tf32_rna_halfway_and_precision(1)
        tests.test_tf32_rna_halfway_and_precision(-1)
        assert t.mma_model(np.array([[1+2**-12]],np.float32),np.ones((1,1),np.float32)).item()==1.
        tests.test_tiled_forward_and_both_backward_owners(True,(17,31),(19,17))
        tests.test_tiled_forward_and_both_backward_owners(False,(17,31),(19,17))
        tests.test_selected_full_fp32_exact_scores_and_detached_kq()
        tests.test_spread_zero_and_nonfinite_fail_closed()
    mutations=[
        (t,'tf32_round','np.uint32(0x1000)','np.uint32(0)','truncate_tf32'),
        (t,'tf32_round','np.uint32(0xffffe000)','np.uint32(0xffff0000)','bf16_mantissa'),
        (t,'mma_model','convert = tf32_round if rounded else np.asarray','convert = np.asarray','no_tf32_conversion'),
        (t,'tiled_forward_model','abs(bulk)>=th[:,None]','abs(bulk)>th[:,None]','strict_selection'),
        (t,'tiled_forward_model','q@key.T*scale','mma_model(q,key.T)*scale','rounded_selected_score'),
        (t,'tiled_forward_model','+len(key))[None,:]<count[:,None]','+len(key))[None,:]<count[:,None]-1','causal_boundary'),
        (t,'tiled_forward_model','sum(1)/count-mean*mean','sum(1)/count+mean*mean','wrong_variance'),
        (t,'tiled_backward_model','range(kh*(H//KH),(kh+1)*(H//KH))','(kh*(H//KH),)','lost_group_head'),
        (t,'tiled_backward_model',"mma_model(sel.T,q,rounded)","mma_model((sel+bulk).T,q,rounded)",'unselected_dk'),
        (t.back,'gate','2*v[m]','4*v[m]','loosened_spread'),
    ]
    pins={str(p.relative_to(t.ROOT)):t.sha(p) for p in [t.ROOT/'scripts/pt_tf32_1.py',t.ROOT/'tests/test_pt_tf32_cpu.py',t.ROOT/'scripts/bp_kernel_2.py']}
    entries=[]
    for mod,name,old,new,label in mutations:
        source=inspect.getsource(getattr(mod,name))
        if source.count(old)!=1:raise ValueError('mutation anchor is not unique: '+label)
        entries.append(dict(label=label,module=mod.__name__,function=name,old=old,new=new))
    suffix='';number=1
    while (t.ART/f'MUTATION_REGISTRATION{suffix}.json').exists():
        number+=1;suffix=f'_{number:03d}'
    t.create_json(t.ART/f'MUTATION_REGISTRATION{suffix}.json',dict(evidence_class=__doc__,
        source_pins=pins,mutations=entries,min_nonerror_kill_fraction=.8,
        rule='Run passing baseline first; mutate one in-memory function; restore in finally; no source edits.'))
    checks();results=[]
    for (mod,name,old,new,label),entry in zip(mutations,entries):
        original=getattr(mod,name);namespace={}
        exec(compile(inspect.getsource(original).replace(old,new),f'<mutation:{label}>','exec'),original.__globals__,namespace)
        setattr(mod,name,namespace[name])
        try:
            checks();status='SURVIVED';detail=None
        except AssertionError as exc:
            status='KILLED';detail=str(exc)[:500]
        except Exception as exc:
            status='ERROR';detail=repr(exc)
        finally:
            setattr(mod,name,original)
        results.append(dict(label=label,status=status,detail=detail))
    checks()
    unchanged=all(t.sha(t.ROOT/p)==h for p,h in pins.items())
    kills=sum(x['status']=='KILLED' for x in results);errors=sum(x['status']=='ERROR' for x in results)
    nonerror=len(results)-errors;fraction=kills/nonerror if nonerror else 0.
    receipt=dict(evidence_class='author CPU mutation; not native kernel/blind coverage',
        results=results,killed=kills,errors=errors,nonerror=nonerror,kill_fraction=fraction,
        source_unchanged=unchanged,verdict='GREEN' if unchanged and not errors and fraction>=.8 else 'RED')
    t.create_json(t.ART/f'MUTATION_RECEIPT{suffix}.json',receipt)
    print(f'MUTATION {receipt["verdict"]}: killed={kills}/{nonerror}, errors={errors}, source_unchanged={unchanged}')
    return 0 if receipt['verdict']=='GREEN' else 1


if __name__=='__main__':sys.exit(main())
