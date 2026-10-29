"""CPU author baseline; required injected defects are not blind verification.

Prior art: DeMillo, Lipton and Sayward (1978), mutation testing; Python
subprocess/importlib (Python 3.6, 2016), isolation/error types. Unverified -
lead to check. Test cases fixed by STE5; ours adds real small-fixture parity.
"""
import copy
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import common as c
import reference5 as ref
import verify5 as v


def groups():
    sample={key:(np.array([.25,.5,1.,2.],np.float32).view(np.uint32)
            if key in ('lse','thr') else c.bf16([.25,.5,1.,2.])) for key in c.KEYS}
    result={cfg:[{key:x.copy() for key,x in sample.items()} for _ in range(c.N)] for cfg in c.CONFIGS}
    for row in result['on']:
        row['dk'][0] += np.uint16(1)
    return result


def test_clean_byte_fixture_passes():
    rows,gates=v.identity_case(groups())
    assert all(gates.values()) and len(rows)==252


def test_one_bit_g_g2_fails():
    d=groups(); d['off'][1]['dk'][2] ^= np.uint16(1)
    rows,gates=v.identity_case(d)
    assert gates['G-G1'] and gates['G-G2'] is False
    assert any(r['gate']=='G-G2' and not r['PASS'] for r in rows)


def test_full_float32_width():
    d=groups(); d['on'][2]['lse'][1] ^= np.uint32(1)
    rows,gates=v.identity_case(d)
    assert gates['G-G4'] is False and gates['G-G6'] is False


def test_signed_zero_distinct():
    a=c.bf16([0.,1.,2.]); b=a.copy(); b[0]=np.uint16(0x8000)
    assert v.compare(a,b)['bitwise'] is False and v.compare(a,b)['maxabs']==0


def test_base_failure_stops():
    d=groups(); d['base'][1]['dv'][0] ^= np.uint16(1)
    rows,gates=v.identity_case(d)
    assert gates=={'G-G1':False} and {r['gate'] for r in rows}=={'G-G1'}


def test_zero_switch_difference_fails_g_g4():
    d=groups(); d['on']=copy.deepcopy(d['off'])
    assert v.identity_case(d)[1]['G-G4'] is False


def signal(slope=1.):
    extra=np.array([1.,-2.,3.,-.5],np.float64)
    off=np.array([.5,-.25,1.,2.],np.float64)
    return ref.signal_check(off+slope*extra,off,off+extra,off,extra)


def test_clean_signal_passes():
    row=signal(); assert row['PASS'] and row['slope']==pytest.approx(1.)


def test_slope_097_fails():
    row=signal(.97)
    assert row['slope']==pytest.approx(.97) and row['cosine']==pytest.approx(1.)
    assert row['PASS'] is False


def test_bad_direction_fails():
    r=np.array([1.,-2.,3.,-.5]); off=np.array([.5,-.25,1.,2.])
    assert ref.signal_check(off+r[::-1],off,off+r,off,r)['PASS'] is False


def test_on_relative_error_condition_independent():
    r=np.array([1.,-2.,3.,-.5]); off=np.array([.5,-.25,1.,2.])
    actual_on=off+10+r
    result=ref.signal_check(actual_on,off+10,off+r,off+10,r)
    assert result['cosine']==pytest.approx(1.) and result['slope']==pytest.approx(1.)
    assert result['PASS'] is False


def test_zero_signal_rejected():
    r=np.array([1.,-2.,3.,-.5]); off=np.array([.5,-.25,1.,2.])
    assert ref.signal_check(off,off,off+r,off,r)['PASS'] is False


def test_missing_governor_fails(tmp_path):
    with pytest.raises(AssertionError,match='missing governor ran/rc0'):
        c.require_governor(tmp_path/'absent.tsv',[c.label('off','small')])


@pytest.mark.parametrize('ending',['ran\t1','refused\t0','ran\t0\textra'])
def test_wrong_governor_outcome(tmp_path,ending):
    p=tmp_path/'governor.tsv'
    p.write_text(f'time\t{c.ROOT}\t{c.label("off","small")}\thalted\tdeadline\t600\t0\t{ending}\n')
    with pytest.raises(AssertionError): c.require_governor(p,[c.label('off','small')])


def test_duplicate_governor_fails(tmp_path):
    p=tmp_path/'governor.tsv'
    row=f'time\t{c.ROOT}\t{c.label("off","small")}\thalted\tdeadline\t600\t0\tran\t0\n'
    p.write_text(row+row)
    with pytest.raises(AssertionError): c.require_governor(p,[c.label('off','small')])


def test_missing_governor_whole_verifier_never_passes(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'LEAD',tmp_path)
    output=tmp_path/'missing.json'
    assert v.verify(output)!=0
    row=json.loads(output.read_text())
    assert row['PASS'] is False and 'missing governor ran/rc0' in row['error']


@pytest.mark.parametrize('package',[c.ROOT/'tensor_cuda',c.ARM_PACKAGE.parent])
@pytest.mark.parametrize('value',['2','','1 ',' 1','true'])
def test_f4_real_extension_subprocess(package,value,tmp_path):
    # A stray fallback module must not mask an initializer error.
    (tmp_path/'_tensor_cuda.py').write_text('raise RuntimeError("STRAY FALLBACK WAS USED")\n')
    code='''import sys,json
sys.path[:0]=[sys.argv[1],sys.argv[2],sys.argv[3]]
try:
 import tensor_cuda
except Exception as exc:
 print(json.dumps({'type':type(exc).__name__,'message':str(exc)}))
else:
 raise AssertionError('invalid switch accepted')
'''
    env=dict(os.environ,TC_APA_DK_STE=value,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(package),str(tmp_path),str(Path(np.__file__).parents[1])],env=env,
                          text=True,capture_output=True,timeout=20,check=True)
    row=json.loads(result.stdout)
    assert row['type']=='ImportError',result
    assert 'TC_APA_DK_STE' in row['message'] and 'unset, 0, or 1' in row['message']
    assert 'ModuleNotFoundError' not in result.stderr


@pytest.mark.parametrize('value',['0','1',None])
def test_valid_switch_import_only(value):
    code='import sys;sys.path[:0]=[sys.argv[1],sys.argv[2]];import tensor_cuda;print("IMPORTED")'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    if value is None: env.pop('TC_APA_DK_STE',None)
    else: env['TC_APA_DK_STE']=value
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(c.ARM_PACKAGE.parent),str(Path(np.__file__).parents[1])],env=env,
                          text=True,capture_output=True,timeout=20,check=True)
    assert result.stdout.strip()=='IMPORTED'
    assert result.stderr.count(f'tensor_cuda: TC_APA_DK_STE={value or "0"}')==1


def test_source_bundle_identical():
    assert (c.ROOT/'tensor_cuda/tensor_cuda/__init__.py').read_bytes()==(c.ARM_PACKAGE/'__init__.py').read_bytes()


@pytest.mark.parametrize('name',['small1','small2'])
def test_real_fixture_reference_pair_parity(name):
    # Read-only real exports and independent REV1 oracle, not mock kernels.
    path=c.OLD/'lead/arm-off-small'/f'{name}-h_g1.npz'
    with np.load(path,allow_pickle=False) as f: cap={k:f[k] for k in f.files}
    ours=ref.oracle(c.input_words(name),name,saved_thr=cap['thr'])
    spec=importlib.util.spec_from_file_location('ste5_readonly_rev1',c.HERE.parent/'apa_ste_rev1/scripts/oracle_lib.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    independent=mod.reference(cap,name)
    for key in ('on','off','dq','dv'):
        np.testing.assert_allclose(ours[key],independent[key],rtol=2e-13,atol=1e-15)
    assert np.any(ours['certified_unselected'])
    # Existing data validates the CPU implementation, never a fresh G-gate.
    with np.load(c.OLD/'lead/arm-on-small'/f'{name}-h_g1.npz',allow_pickle=False) as f: on=f['dk']
    assert ref.signal_check(c.decode(on),c.decode(cap['dk']),ours['on'],ours['off'],ours['extra'])['PASS']


def test_ulp_subnormal_spacing():
    assert ref.ulp(np.array([0.,2.**-126,1.])).tolist()==[2.**-133,2.**-133,2.**-7]


@pytest.mark.parametrize('defect', [None, 'missing_binding', 'wrong_binding',
                                  'wrong_input_pin', 'one_bit_tensor'])
def test_real_capture_shape_premise_cli(tmp_path, defect):
    # Prior art: no prior art known to me; no new algorithm.
    # Faithful real exports, with only new-writer provenance supplied for this
    # CPU fixture. These are NOT re-captures or admissible GPU gate evidence.
    # Compare the full key set to the actual capture5 writer, so a future writer
    # schema change cannot silently leave this integration fixture behind.
    tree=ast.parse((HERE/'capture5.py').read_text())
    emitted=[node.args[1] for node in ast.walk(tree)
             if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
             and isinstance(node.func.value,ast.Name) and node.func.value.id=='c'
             and node.func.attr=='put']
    assert len(emitted)==1 and isinstance(emitted[0],ast.Dict)
    writer_keys={key.value for key in emitted[0].keys}
    exported=tmp_path/'captures'
    exported.mkdir()
    (exported/'governor.tsv').write_bytes((c.LEAD/'governor.tsv').read_bytes())
    for suite in ('small','grapa'):
        for run in range(c.N):
            source=c.folder('base',suite,run)
            target=exported/c.label('base',suite)/f'run-{run}'
            target.mkdir(parents=True)
            meta=json.loads((source/'capture.json').read_text())
            meta.update(driver_sha256=c.digest(HERE/'capture5.py'),
                        effective_pins_sha256=c.evidence_pins_digest())
            assert set(meta)==writer_keys
            if suite=='small' and run==0:
                if defect=='missing_binding':
                    del meta['effective_pins_sha256']
                elif defect=='wrong_binding':
                    meta['effective_pins_sha256']='0'*64
                elif defect=='wrong_input_pin':
                    meta['input_pins_sha256']='0'*64
            for case in meta['cases']:
                name=case['name']
                if defect=='one_bit_tensor' and suite=='small' and run==0 and name=='small1':
                    with np.load(source/f'{name}.npz',allow_pickle=False) as f:
                        words={k:f[k] for k in f.files}
                    words['dk'].flat[0] ^= np.uint16(1)
                    with (target/f'{name}.npz').open('xb') as f:
                        np.savez(f,**words)
                    case['sha256']=c.digest(target/f'{name}.npz')
                else:
                    (target/f'{name}.npz').symlink_to(source/f'{name}.npz')
            (target/'capture.json').write_text(json.dumps(meta))
    output=tmp_path/'premise.json'
    # Use the actual CLI entry point/--premise; redirect only the read root to
    # the faithful export. All pin, governor and tensor checks stay active.
    code='''import runpy,sys
sys.path.insert(0,sys.argv[1])
import common as c
c.LEAD=c.Path(sys.argv[2])
sys.argv=[str(c.HERE/'verify5.py'),'--premise','--output',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(HERE),str(exported),str(output)],
                          text=True,capture_output=True,timeout=120)
    report=json.loads(output.read_text())
    assert 'KeyError' not in report.get('error',''),result
    if defect is None:
        assert result.returncode==0 and report['PASS'],report
        assert report['gates']=={'G-G1':True}
        assert len(report['base_inventory'])==6 and len(report['identity'])==54
        assert all(r['PASS'] for r in report['identity'])
        assert all(r['PASS'] for r in report['governor_base'].values())
    elif defect=='one_bit_tensor':
        assert result.returncode==1 and not report['PASS']
        assert report['gates']=={'G-G1':False}
        assert any(not r['PASS'] for r in report['identity'])
    else:
        assert result.returncode==1 and not report['PASS'] and report['gates']=={}
        expected={'missing_binding':'re-capture required',
                  'wrong_binding':'wrong capture metadata',
                  'wrong_input_pin':'wrong capture metadata'}[defect]
        assert expected in report['error'],report
