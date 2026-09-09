"""Registered scalar-rule mutants, exercised on in-memory copies after G1 passes."""
import math
import numpy as np
from apa_sp2_common import ART, registration, write_new


def run():
    variants={
        'drop_2eq':lambda eps,e:-math.log(eps),
        'negative_log':lambda eps,e:math.log(eps)+2*e,
        'omit_log':lambda eps,e:2*e,
        'strict_refine_boundary':lambda eps,e:-math.log(eps)+2*e,
        'halve_eq':lambda eps,e:-math.log(eps)+e,
    }
    killed=[];survived=[]
    for name,fn in variants.items():
        failed=False
        for eps,e in [(1.,0.),(.1,2.),(.01,.5)]:
            correct=-math.log(eps)+2*e
            # An adversarial two-key q=[1], K=exact draw places the later key
            # between a defective cutoff and the sufficient-certificate cutoff.
            cut=fn(eps,e)
            gap=(max(cut,0)+correct)/2
            bulk=np.array([e,e-gap]);exact=np.array([0.,bulk[1]+e])
            prefix=np.maximum.accumulate(bulk)
            mask=bulk>prefix-cut if name=='strict_refine_boundary' else bulk>=prefix-cut
            relative=np.exp(exact-exact.max())
            if np.any((~mask)&(relative>=eps)):
                failed=True;break
        (killed if failed else survived).append(name)
    fraction=len(killed)/len(variants)
    result=dict(evidence_class='CPU mutation tests of registered mathematical rule',
                status='PASS' if fraction>=registration()['G1']['mutation_kill_fraction_min'] else 'FAIL',
                killed=killed,survived=survived,kill_fraction=fraction,
                source_mutation='in-memory functions only; production source unchanged')
    import time
    write_new(ART/f'mutations_{time.time_ns()}.json',result)
    print(result)
    assert result['status']=='PASS'


if __name__=='__main__':run()
