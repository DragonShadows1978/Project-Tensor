#!/usr/bin/env python3
"""Lead-only APA forward/backward timing, not a GRAPA step projection.

Prior art: NVIDIA CUDA events (2007+, unverified - lead to check), existing
STE1 bp_wall timer; one warmup and two samples fixed by STE5 order. Ours:
g1/h pair scope and provenance, no new optimization.
"""
import argparse
import os
import time
import common as c
import numpy as np


def timing(switch):
    c.environment(); c.check_pins(); c.require_premise()
    assert os.environ.get('APA_STE5_LEAD_GPU')=='1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert os.environ.get('TC_APA_DK_STE')==str(switch)
    assert os.environ.get('TC_DETERMINISTIC')=='1'
    assert os.environ.get('TC_DET_EMBED_BWD')=='1'
    old=c.import_old_driver(); tc=old.load_tc('arm')
    tc._C.bp_kernel_2_set_variant('g1'); tc._C.bp_kernel_4_set_variant('h')
    assert tc._C.bp_kernel_2_get_variant()=='g1'
    assert tc._C.bp_kernel_4_get_variant()=='h'
    with np.load(c.fixture('grapa'),allow_pickle=False) as d:
        q,k,kq,v,do=[tc.tensor(d[x],dtype='bfloat16',device='cuda') for x in c.INPUTS]
    def pair():
        out,lse,thr=tc._C.apa_selective_fwd_train_variant(q,k,kq,v,128**-.5,c.Z,True,'h')
        grads=tc._C.apa_selective_bwd_variant(q,k,kq,v,do,lse,thr,out,128**-.5,True,'g1')
        return (out,lse,thr)+tuple(grads)
    warm=pair(); tc.synchronize()
    samples=[]; wall=[]
    for _ in range(2):
        start=time.perf_counter()
        tc._C.bp_wall_begin(); result=pair(); ms=float(tc._C.bp_wall_end())
        tc.synchronize(); wall.append((time.perf_counter()-start)*1000)
        samples.append(ms)
    assert all(np.isfinite(samples)) and min(samples)>0
    c.put(c.LEAD/f'timing-{switch}.json',{'switch':switch,'variant':'g1','forward_variant':'h',
        'shape':c.SHAPES['grapa'],'dtype':'bfloat16','causal':True,'zthr':c.Z,
        'TC_DETERMINISTIC':'1','TC_DET_EMBED_BWD':'1','warmups':1,'samples_ms':samples,
        'host_wall_ms':wall,'pid':os.getpid(),'so_sha256':c.digest(c.ARM_PACKAGE/c.SO_NAME),
        'driver_sha256':c.digest(__file__),'input_pins_sha256':c.digest(c.HERE/'INPUT_PINS.json'),'effective_pins_sha256':c.evidence_pins_digest(),
        'scope':'CUDA event time of one forward h + backward g1 pair; includes wrapper allocation/zeroing; n=2',
        'step_time':'GRAPA arm step time comes from GRAPA own smoke; not projected from this pair'})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--switch',type=int,choices=(0,1),required=True)
    timing(p.parse_args().switch)
