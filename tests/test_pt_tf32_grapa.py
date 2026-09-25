"""CPU integration tests with a control double; no tensor/GPU operations.
Prior art: CC41 policy context tests (2026), taken; ours: TF32 mode restoration.
"""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pt_tf32_grapa as g


def fake():
    state=dict(gemm=False,bwd='g1',fwd='h')
    C=SimpleNamespace(get_tf32_gemm=lambda:state['gemm'],
                      set_tf32_gemm=lambda b:state.update(gemm=b))
    tc=SimpleNamespace(_C=C)
    fp=SimpleNamespace(FP32_BWD_FALLBACK={'g1':'f','g2':'f'},FP32_FWD_FALLBACK={'h':'a'})
    @contextlib.contextmanager
    def base(tc):
        b,f=state['bwd'],state['fwd']
        state.update(bwd=fp.FP32_BWD_FALLBACK[b],fwd=fp.FP32_FWD_FALLBACK[f])
        try:yield
        finally:state.update(bwd=b,fwd=f)
    fp.fp32_apa_variants=base
    return state,tc,fp


@pytest.mark.parametrize('old_mode',[False,True])
@pytest.mark.parametrize('error',[False,True])
def test_initial_and_replay_scope_restore_gemm_and_variants(old_mode,error):
    state,tc,fp=fake();state['gemm']=old_mode
    g.install_policy(tc,fp)
    for phase in ('initial','replay'):
        try:
            with fp.fp32_apa_variants(tc):
                assert state==dict(gemm=True,bwd='g1_tf32',fwd='h_tf32')
                if error:raise ValueError('block failed')
        except ValueError:
            assert error
        assert state==dict(gemm=old_mode,bwd='g1',fwd='h')
    with pytest.raises(RuntimeError,match='already installed'):g.install_policy(tc,fp)


def test_grapa_registered_geometry_and_engine_are_fork():
    r=g.registered()
    assert Path(r['engine']['so_path']).resolve().is_relative_to(g.ROOT)
    assert r['states']['onset']['batch']==25360
    assert r['pt_tf32_1']['gates']['onset_cos_min']==.99
    assert r['pt_tf32_1']['gates']['healthy_control_cos_min']==.999
    assert r['pt_tf32_1']['gates']['seconds_per_step_max']==6.5
    for arm in ('none','0-10'):
        argv=r['timing']['arms'][arm]
        assert argv[argv.index('--steps')+1]=='25295'
        assert '{ARM_DIR}' in argv[argv.index('--save-ckpt')+1]
