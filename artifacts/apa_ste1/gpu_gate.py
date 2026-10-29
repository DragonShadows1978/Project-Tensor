#!/usr/bin/env python3
"""Lead-only GPU captures and CPU receipt recomputation. Never auto-run a GPU.

Prior art: Bengio et al. 2013 STE; Dao et al. 2022 attention VJP; NVIDIA
CUDA events/Nsight Systems (2007+/2013+, years unverified - lead to check).
SHA-256 manifests: NIST FIPS 180-4 (2015). Ours: frozen fixtures, isolation,
full-width byte gates, base-error calibration and governor-bound receipts.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=Path('/mnt/ForgeRealm/engines/tf32-det-69b154cd2e9a/tensor_cuda')
ENGINE=HERE/'engine/tensor_cuda'
LEAD=HERE/'lead'
KEYS=('out','lse','thr','dq','dk','dv')
VARIANTS=('a','train','b','c','d','f','f_pass_a','g1','g2','g1_tf32','h_g1','h_tf32_g1_tf32')
SHAPES={'small1':(1,2,7,11,32,16,2),'small2':(1,4,17,19,32,16,2),'grapa':(1,16,4096,4096,128,64,16)}
spec=importlib.util.spec_from_file_location('apa_reference',HERE/'reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Fresh receipts only. No silent overwrite of calibration or an earlier run.
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def fixture_path(shape):return HERE/'inputs'/f'{shape}.npz'


def prepare():
    (HERE/'inputs').mkdir(exist_ok=True);LEAD.mkdir(exist_ok=True)
    records={}
    for name,shape in SHAPES.items():
        path=fixture_path(name)
        if path.exists(): raise FileExistsError(path)
        arrays=ref.fixture(shape,seed=412+list(SHAPES).index(name))
        if name=='grapa':
            # Exact singleton bulk dot in serial and warp orders; keeps the
            # refine-all gate about STE, without a rounded threshold tie.
            arrays[0][:,:,0,:]=.25;arrays[2][:,:,0,:]=.25
        # Seeded arrays are serialized once, then consumed by every engine arm.
        np.savez(path,**dict(zip(('q','k','kq','v','do'),[x.astype(np.float32) for x in arrays])))
        records[name]={'shape':shape,'sha256':digest(path),'bytes':path.stat().st_size,'causal':name!='small1'}
    put_json(HERE/'INPUTS.json',records)


def raw(x,dtype):
    a=np.ascontiguousarray(x.numpy())
    assert np.isfinite(a).all()
    if dtype=='bfloat16':
        u=a.astype(np.float32,copy=False).view(np.uint32)
        assert np.all((u & 65535)==0),'BF16 upcast must be exact'
        return (u>>16).astype(np.uint16)
    assert a.dtype==np.float32
    return a.view(np.uint32)


def decode(a):
    if a.dtype==np.uint16:return (a.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    assert a.dtype==np.uint32
    return a.view(np.float32).astype(np.float64)


def healthy(a):
    x=decode(a)
    return bool(np.isfinite(x).all() and np.any(x!=0) and np.any(x!=x.flat[0]))


def cases(suite):
    for name in (('small1','small2') if suite=='small' else ('grapa',)):
        for variant in (VARIANTS if suite=='small' else ('a',)):
            yield name,variant,'float32' if variant in ('g1_tf32','h_tf32_g1_tf32') else 'bfloat16'


def load_tc(which):
    # Baseline and arm MUST be loaded in different fresh processes.
    assert 'tensor_cuda' not in sys.modules
    for key in list(os.environ):
        if key.startswith('TC_') and key not in {'TC_APA_DK_STE','TC_DETERMINISTIC','TC_DET_EMBED_BWD'}:
            os.environ.pop(key)  # fixed registered defaults in each worker
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        assert os.environ.get(key)=='1',key
    path=BASE if which=='base' else ENGINE
    pin='69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c' if which=='base' else json.loads((HERE/'engine/ENGINE_MANIFEST.json').read_text())['so_sha256']
    assert digest(path/'tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so')==pin
    sys.path.insert(0,str(path))
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().parent==path/'tensor_cuda'
    assert tc._C.bp_kernel_2_get_variant()=='a'
    assert tc.get_deterministic()
    return tc


def ops(tc,t,scale,z,causal,variant):
    q,k,kq,v,do=t
    if variant in ('h_g1','h_tf32_g1_tf32'):
        fwd='h' if variant=='h_g1' else 'h_tf32'
        out,lse,thr=tc._C.apa_selective_fwd_train_variant(q,k,kq,v,scale,z,causal,fwd)
        variant='g1' if variant=='h_g1' else 'g1_tf32'
    else:out,lse,thr=tc.apa_selective_fwd_train(q,k,kq,v,scale,z,causal)
    if variant=='train':
        for x in (q,k,v):x.requires_grad=True;x.zero_grad()
        actual=tc.apa_selective_train(q,k,kq,v,scale,z,causal)
        (actual*do).sum().backward()
        out=actual;grads=(q.grad,k.grad,v.grad)
    elif variant=='a':grads=tc.apa_selective_bwd(q,k,kq,v,do,lse,thr,scale,causal)
    else:grads=tc._C.apa_selective_bwd_variant(q,k,kq,v,do,lse,thr,out,scale,causal,variant)
    tc.synchronize()
    return (out,lse,thr)+tuple(grads)


def check_capture(folder,suite):
    checks=[]
    for name,variant,dtype in cases(suite):
        with np.load(folder/f'{name}-{variant}.npz') as data:
            assert set(data.files)==set(KEYS+('q','k','kq','v','do'))
            for key in data.files:
                # c deliberately returns zero off dK/dV, and always zero dV.
                # They cannot meet L19: no byte-comparison gate may count c.
                if variant=='c' and key in ('dk','dv'):continue
                assert healthy(data[key]),f'zero/constant/nonfinite {name}/{variant}/{key}'
            checks.append({'case':name+'-'+variant,'sha256':digest(folder/f'{name}-{variant}.npz')})
    return checks


def capture(args):
    folder=LEAD/args.label;folder.mkdir(parents=True,exist_ok=False)
    assert args.switch==int(os.environ['TC_APA_DK_STE'])
    manifest=json.loads((HERE/'INPUTS.json').read_text())
    if args.which=='arm' and args.switch==1 and not args.all:
        assert (LEAD/'CALIBRATION.json').exists(),'freeze base errors before arm'
    tc=load_tc(args.which)
    z=-1e6 if args.all else float(tc.quant._norm_ppf(.85))
    start=time.monotonic()
    for name,variant,dtype in cases(args.suite):
        path=fixture_path(name);assert digest(path)==manifest[name]['sha256']
        with np.load(path) as data:
            tensors=[tc.tensor(data[k],dtype=dtype,device='cuda') for k in ('q','k','kq','v','do')]
        outputs=ops(tc,tensors,SHAPES[name][4]**-.5,z,name!='small1',variant)
        saved={k:raw(x,'float32' if k in ('lse','thr') else dtype) for k,x in zip(KEYS,outputs)}
        saved.update({k:raw(x,dtype) for k,x in zip(('q','k','kq','v','do'),tensors)})
        np.savez(folder/f'{name}-{variant}.npz',**saved)
    checked=check_capture(folder,args.suite)
    put_json(folder/'capture.json',{'PASS':True,'evidence_class':'raw tensor capture only, not gate verdict','which':args.which,'switch':args.switch,'suite':args.suite,'all_selected_zthr':args.all,'zthr':z,'cases':checked,'elapsed_s':time.monotonic()-start})


def ideal(data,name,variant,ste,z):
    q,k,kq,v,do=[decode(data[key]) for key in ('q','k','kq','v','do')]
    scale=SHAPES[name][4]**-.5
    state=ref.forward(q,k,kq,v,scale,z,name!='small1')
    # The ideal float64 primal drives the reference; baseline error includes
    # output-dot and tiled operand rounding of each existing variant.
    grads=ref.backward(q,k,kq,v,do,scale,state,ste,output_dot=variant in ('d','f','g1','g2','g1_tf32','h_g1','h_tf32_g1_tf32'))
    if variant=='c':
        off=ref.backward(q,k,kq,v,do,scale,state,False)
        grads=(grads[0],grads[1]-off[1] if ste else np.zeros_like(k),np.zeros_like(v))
    return dict(zip(KEYS,state[:3]+grads)),state


def calibration_data():
    assert governor('ste1-base-small'),'base governor outcome must be ran/rc0'
    checks=check_capture(LEAD/'base-small','small')
    result={'protocol':'max(8*pooled base off maxabs error,1e-5), by dtype/tensor; frozen before on','errors':{},'tolerances':{},'sources':checks,'per_case':{}}
    for name,variant,dtype in cases('small'):
        with np.load(LEAD/'base-small'/f'{name}-{variant}.npz') as data:
            expected,_=ideal(data,name,variant,False,float(json.loads((LEAD/'base-small/capture.json').read_text())['zthr']))
            errors={k:float(np.max(np.abs(decode(data[k])-expected[k]))) for k in KEYS}
        result['per_case'][name+'-'+variant]=errors
        pool=result['errors'].setdefault(dtype,{k:0. for k in KEYS})
        for k in KEYS:pool[k]=max(pool[k],errors[k])
    result['tolerances']={dt:{k:max(8*e,1e-5) for k,e in errs.items()} for dt,errs in result['errors'].items()}
    result['PASS']=True
    return result


def calibrate():
    put_json(LEAD/'CALIBRATION.json',calibration_data())


def governor(label):
    path=LEAD/'governor.tsv'
    if not path.exists():return False
    rows=[line.split('\t') for line in path.read_text().splitlines()]
    hits=[row for row in rows if len(row)==9 and row[2]==label]
    return len(hits)==1 and hits[0][-2:]==['ran','0']


def check_refine_all(data,name):
    threshold=decode(data['thr'])
    if name=='grapa':
        assert np.all(threshold[:,:,1:]<0),'all multi-key thresholds must be negative'
        q=decode(data['q']);kq=decode(data['kq'])
        assert np.all(q[:,:,0,:]==.25) and np.all(kq[:,:,0,:]==.25)
        bulk=np.float32(SHAPES[name][4]*.25*.25)*np.float32(SHAPES[name][4]**-.5)
        assert np.all(threshold[:,:,0]<=bulk),'singleton must be selected too'
    else:assert np.all(threshold<0),'negative threshold selects every visible key'


def exact(left,right,suite,keys=KEYS,refine_all=False):
    checks=[]
    for name,variant,dtype in cases(suite):
        # L19 forbids vacuous all-zero diagnostic comparisons; the c diagnostic
        # is checked numerically in L-G3, never credited toward a byte gate.
        if variant=='c':continue
        with np.load(LEAD/left/f'{name}-{variant}.npz') as a,np.load(LEAD/right/f'{name}-{variant}.npz') as b:
            if refine_all:
                check_refine_all(a,name);check_refine_all(b,name)
            for k in KEYS:
                assert healthy(a[k]) and healthy(b[k]),f'vacuous {name}/{variant}/{k}'
            for k in ('q','k','kq','v','do'):
                assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()
            for k in keys:
                checks.append({'case':name+'-'+variant,'tensor':k,'PASS':a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),'different_elements':int(np.count_nonzero(a[k]!=b[k]))})
    return checks


def correctness():
    cal=json.loads((LEAD/'CALIBRATION.json').read_text());checks=[]
    assert cal==calibration_data(),'frozen calibration must recompute from base raw files'
    for name,variant,dtype in cases('small'):
        folder=LEAD/'arm-on-small'
        z=json.loads((folder/'capture.json').read_text())['zthr']
        with np.load(folder/f'{name}-{variant}.npz') as a,np.load(LEAD/'arm-off-small'/f'{name}-{variant}.npz') as b:
            expected,state=ideal(a,name,variant,True,z)
            for k in KEYS:
                error=float(np.max(np.abs(decode(a[k])-expected[k])))
                checks.append({'case':name+'-'+variant,'tensor':k,'maxabs':error,'tolerance':cal['tolerances'][dtype][k],'PASS':error<=cal['tolerances'][dtype][k]})
            # Key 1 is exactly zero in kq, visible and strictly unselected in
            # every query; no cancellation across selected pairs can fake it.
            sel=state[3][:,:,:,1]
            off=decode(b['dk'])[:,:,1,:];on=decode(a['dk'])[:,:,1,:]
            witness=bool(not sel.any() and np.all(off==0) and np.any(on!=0))
            checks.append({'case':name+'-'+variant,'tensor':'unselected-key-1 witness','PASS':witness,'off_nonzero':int(np.count_nonzero(off)),'on_nonzero':int(np.count_nonzero(on))})
    return checks


def timing(args):
    assert args.switch==int(os.environ['TC_APA_DK_STE'])
    tc=load_tc('arm')
    with np.load(fixture_path('grapa')) as d:t=[tc.tensor(d[k],dtype='bfloat16') for k in ('q','k','kq','v','do')]
    q,k,kq,v,do=t;scale=128**-.5;z=float(tc.quant._norm_ppf(.85))
    out,lse,thr=tc.apa_selective_fwd_train(q,k,kq,v,scale,z,True)
    def call():return tc.apa_selective_bwd(q,k,kq,v,do,lse,thr,scale,True)
    call();tc.synchronize()  # one warmup, outside profiler capture
    cudart=ctypes.CDLL('/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so')
    assert cudart.cudaProfilerStart()==0
    samples=[]
    for _ in range(2):
        tc._C.bp_wall_begin();grads=call();ms=tc._C.bp_wall_end();samples.append(float(ms))
    tc.synchronize();assert cudart.cudaProfilerStop()==0
    put_json(LEAD/f'timing-{args.switch}-wrapper.json',{'PASS':all(np.isfinite(samples)) and min(samples)>0,'samples_ms':samples,'scope':'CUDA-event whole backward wrapper, includes allocation/zeroing; kernel-only durations are recomputed from Nsight SQLite','step':'unavailable: no standalone engine training step or registered GRAPA checkpoint/optimizer fixture; GRAPA times its step'})


def profile(args):
    prefix=LEAD/f'timing-{args.switch}'
    subprocess.run(['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda',
        '--capture-range=cudaProfilerApi','--capture-range-end=stop',
        '--output',str(prefix),sys.executable,'-B',str(Path(__file__).resolve()),
        'timing','--switch',str(args.switch)],check=True)
    subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',
        str(prefix)+'.nsys-rep'],check=True)
    put_json(LEAD/f'timing-{args.switch}-kernel.json',{'PASS':True,'raw':timing_sql(args.switch)})


def timing_sql(switch):
    db=LEAD/f'timing-{switch}.sqlite'
    with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as c:
        rows=c.execute('SELECT s.value,k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
    samples=[(int(end)-int(start))/1e6 for name,start,end in rows if 'apa_selective_bwd_kernel' in name]
    assert len(samples)==2 and all(np.isfinite(samples)) and min(samples)>0,(len(samples),samples)
    return {'samples_ms':samples,'median_ms':float(np.median(samples)),'sqlite_sha256':digest(db)}


def verify():
    gates={};deps={}
    def lane(name,fn,labels):
        try:
            checks=fn();ok=bool(checks) and all(x['PASS'] for x in checks)
            outcomes={label:governor(label) for label in labels}
            gates[name]={'PASS':bool(ok and all(outcomes.values())),'comparisons':checks,'governor_ran_rc0':outcomes}
        except Exception as e:gates[name]={'PASS':False,'error':repr(e)}
    lane('L-G1',lambda:exact('base-small','arm-off-small','small')+exact('base-grapa','arm-off-grapa','grapa'),['ste1-base-small','ste1-arm-off-small','ste1-base-grapa','ste1-arm-off-grapa'])
    lane('L-G2',lambda:exact('arm-off-small','arm-on-small','small',KEYS[:3])+exact('arm-off-grapa','arm-on-grapa','grapa',KEYS[:3]),['ste1-arm-off-small','ste1-arm-on-small','ste1-arm-off-grapa','ste1-arm-on-grapa'])
    lane('L-G3',correctness,['ste1-base-small','ste1-arm-off-small','ste1-arm-on-small'])
    lane('L-G4',lambda:exact('arm-off-all-small','arm-on-all-small','small',refine_all=True)+exact('arm-off-all-grapa','arm-on-all-grapa','grapa',refine_all=True),['ste1-arm-off-all-small','ste1-arm-on-all-small','ste1-arm-off-all-grapa','ste1-arm-on-all-grapa'])
    lane('L-G5',lambda:exact('arm-on-small','arm-repeat-small','small')+exact('arm-on-grapa','arm-repeat-grapa','grapa'),['ste1-arm-on-small','ste1-arm-repeat-small','ste1-arm-on-grapa','ste1-arm-repeat-grapa'])
    try:
        times={str(x):timing_sql(x) for x in (0,1)}
        wrappers={str(x):json.loads((LEAD/f'timing-{x}-wrapper.json').read_text()) for x in (0,1)}
        # Verdict rechecks raw samples, not the worker's PASS field.
        ok=all(len(w['samples_ms'])==2 and np.isfinite(w['samples_ms']).all() and min(w['samples_ms'])>0 for w in wrappers.values())
        outcomes={f'ste1-timing-{x}':governor(f'ste1-timing-{x}') for x in (0,1)}
        gates['L-G6']={'PASS':bool(ok and all(outcomes.values())),'kernels':times,'on_over_off':times['1']['median_ms']/times['0']['median_ms'],'wrapper':wrappers,'governor_ran_rc0':outcomes,'whole_step':'not timed; GRAPA owns the step measurement'}
    except Exception as e:gates['L-G6']={'PASS':False,'error':repr(e)}
    put_json(LEAD/'GPU_RECEIPT.json',{'PASS':all(g['PASS'] for g in gates.values()),'gates':gates,'evidence_class':'raw tensor bytes, float64 reference and kernel profiler records; requires governor ran rc0'})
    print(json.dumps({k:v['PASS'] for k,v in gates.items()},sort_keys=True))
    return 0 if all(g['PASS'] for g in gates.values()) else 1


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('calibrate');sub.add_parser('verify')
    c=sub.add_parser('capture');c.add_argument('--which',choices=['base','arm'],required=True);c.add_argument('--switch',type=int,choices=[0,1],required=True);c.add_argument('--suite',choices=['small','grapa'],required=True);c.add_argument('--all',action='store_true');c.add_argument('--label',required=True)
    for mode in ('timing','profile'):
        t=sub.add_parser(mode);t.add_argument('--switch',type=int,choices=[0,1],required=True)
    a=p.parse_args()
    if a.command in ('capture','timing','profile'):
        assert os.environ.get('APA_STE1_LEAD_GPU')=='1','GPU commands are lead-only'
        assert os.environ.get('CUDA_VISIBLE_DEVICES'),'lead must explicitly select a GPU'
    if a.command=='prepare':prepare()
    elif a.command=='capture':capture(a)
    elif a.command=='calibrate':calibrate()
    elif a.command=='timing':timing(a)
    elif a.command=='profile':profile(a)
    else:return verify()
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        # Ordinary failure writes a RED receipt and returns nonzero. A killed
        # process cannot write; the outer EXIT verifier reports missing raw/
        # governor outcome as RED instead.
        import traceback
        traceback.print_exc()
        label='failure-'+str(time.time_ns())
        put_json(LEAD/(label+'.json'),{'PASS':False,'argv':sys.argv,'error':repr(error)})
        sys.exit(1)
