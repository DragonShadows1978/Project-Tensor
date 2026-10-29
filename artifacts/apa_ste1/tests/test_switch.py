"""CPU-only extension initialization contract, no CUDA allocations.
Prior art: Project-Tensor process env opt-ins (2026); test cases ours.
"""
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('value,expected',[(None,0),('0',0),('1',1)])
def test_switch_once(value,expected):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(ROOT/'tensor_cuda')
    if value is None: env.pop('TC_APA_DK_STE',None)
    else:env['TC_APA_DK_STE']=value
    code='import tensor_cuda; import tensor_cuda; import importlib; importlib.reload(tensor_cuda)'
    p=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
    assert p.stderr.splitlines()==[f'tensor_cuda: TC_APA_DK_STE={expected}']


@pytest.mark.parametrize('value',['','2','-1','01','true',' 1','1 '])
def test_invalid(value):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',TC_APA_DK_STE=value,PYTHONPATH=str(ROOT/'tensor_cuda'))
    p=subprocess.run([sys.executable,'-B','-c','import tensor_cuda'],env=env,capture_output=True,text=True,timeout=20)
    assert p.returncode!=0
    assert 'TC_APA_DK_STE must be unset, 0, or 1' in p.stderr
    assert not any(x.startswith('tensor_cuda: TC_APA_DK_STE=') for x in p.stderr.splitlines())
