#!/usr/bin/env python3
"""Registered CPU semantic mutations in memory; original files never changed."""
from pathlib import Path
import json
import sys
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tensor_cuda/tests')]
from apa_sp1_cpu import registration,write_receipt
import apa_sp1_reference as ref


def corpus(candidate):
    for bulk in [[-5,-5,0,-1,2,2],[3,0,1,2],[1,0,10],[-2,1,-5,0]]:
        b=np.array([bulk],np.float32)
        for delta in [0,0.5,2]:
            np.testing.assert_array_equal(candidate.prefix_mask(b,delta),ref.prefix_mask(b,delta))
    rng=np.random.default_rng(614)
    q=rng.standard_normal((1,4,3,33),dtype=np.float32)
    k=rng.standard_normal((1,2,11,33),dtype=np.float32)
    kq=k+0.1*rng.standard_normal(k.shape,dtype=np.float32)
    v=rng.standard_normal((1,2,11,7),dtype=np.float32)
    sinks=np.array([0,3,8,10],np.float32)
    want=ref.tensor_reference(q,k,kq,v,1/np.sqrt(33),0.5,True,sinks,'prefix')
    got=candidate.tensor_reference(q,k,kq,v,1/np.sqrt(33),0.5,True,sinks,'prefix')
    np.testing.assert_allclose(got,want,atol=1e-3,rtol=1e-3)


def main():
    registration();corpus(ref)  # mandatory green baseline before mutations
    path=ROOT/'tensor_cuda/tests/apa_sp1_reference.py';source=path.read_text()
    needle='mask = bulk >= np.maximum.accumulate(bulk, axis=-1) - F(delta)'
    variants={
      'abs_bulk_selection':(needle,'mask = np.abs(bulk) >= np.maximum.accumulate(bulk, axis=-1) - F(delta)'),
      'exclusive_prefix':(needle,'mask = bulk >= np.concatenate([np.full_like(bulk[..., :1], -np.inf), np.maximum.accumulate(bulk, axis=-1)[..., :-1]], axis=-1) - F(delta)'),
      'strict_greater':(needle,'mask = bulk > np.maximum.accumulate(bulk, axis=-1) - F(delta)'),
      'reset_prefix_every_two':(needle,'mask = bulk >= np.maximum.accumulate(bulk.reshape(*bulk.shape[:-1], -1, 2), axis=-1).reshape(bulk.shape) - F(delta)'),
      'top_left_causal':('lengths = S - L + np.arange(start, end) + 1 if causal','lengths = np.arange(start, end) + 1 if causal'),
      'sink_double_fold':('den += np.exp(np.asarray(sinks, dtype=F) - maximum)','den += 2 * np.exp(np.asarray(sinks, dtype=F) - maximum)'),
    }
    # Reset mutant handles odd lengths without a runtime error: do two-key
    # segment maxima through a Python comprehension instead of reshape.
    variants['reset_prefix_every_two']=(needle,'mask = bulk >= np.stack([bulk[..., (j//2)*2:j+1].max(-1) for j in range(bulk.shape[-1])], axis=-1) - F(delta)')
    results=[]
    for name,(before,after) in variants.items():
        assert source.count(before)==1,(name,source.count(before))
        module=types.ModuleType('apa_sp1_mutant');module.__file__=str(path)
        exec(compile(source.replace(before,after),str(path)+'::'+name,'exec'),module.__dict__)
        try:corpus(module);status='SURVIVED'
        except AssertionError:status='KILLED'
        except Exception as e:status='ERROR: '+repr(e)
        results.append(dict(mutant=name,status=status))
    nonerrors=[r for r in results if not r['status'].startswith('ERROR')]
    killed=sum(r['status']=='KILLED' for r in nonerrors);fraction=killed/len(nonerrors)
    write_receipt(ROOT/'artifacts/apa_sp1/mutations.json',dict(evidence_class='CPU mutation unit test',
       baseline='PASS',results=results,kill_fraction_nonerror=fraction,gate='PASS' if fraction>=0.8 else 'FAIL',
       survivor_analysis='exclusive_prefix is decision-equivalent for delta>=0: a new record passes either cutoff, non-records see identical maxima. It cannot be killed by output/selection tests. Conservative denominator retains it among nonerror mutants.',
       file_safety='Mutations executed in isolated in-memory modules; original source bytes never modified. Author baseline, not independent blind verification.'))
    print(json.dumps(results));print('kill fraction',fraction);assert fraction>=0.8


if __name__=='__main__':main()
