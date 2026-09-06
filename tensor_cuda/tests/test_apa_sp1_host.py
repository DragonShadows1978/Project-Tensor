"""New binding/flag guards, empty CPU tensors avoid legacy from_host CUDA check.

Empty extents suffice because each tested guard precedes positive-geometry
validation; no arithmetic, data transfer or CUDA launch is claimed here.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from apa_sp1_gpu import load_runtime


@pytest.fixture
def runtime(monkeypatch):
    tc=load_runtime()
    monkeypatch.delenv('TC_APA_SP',raising=False)
    return tc


@pytest.mark.parametrize('flag',[None,'0','true','01','2',''])
def test_flag_off_is_a_hard_opt_in_guard(runtime,monkeypatch,flag):
    if flag is not None:monkeypatch.setenv('TC_APA_SP',flag)
    t=runtime.tensor(np.zeros((0,1,1,1),np.float32),device='cpu')
    with pytest.raises(RuntimeError,match='requires TC_APA_SP=1'):
        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)


@pytest.mark.parametrize('scale,delta',[(0,0),(-1,0),(np.nan,0),(np.inf,0),(1,-1),(1,np.inf),(1,np.nan)])
def test_invalid_scalars_rejected_before_device(runtime,monkeypatch,scale,delta):
    monkeypatch.setenv('TC_APA_SP','1')
    t=runtime.tensor(np.zeros((0,1,1,1),np.float32),device='cpu')
    with pytest.raises(RuntimeError,match='scale must be finite'):
        runtime._C.apa_selective_attention_sp(t,t,t,t,scale,delta)


def test_rank_and_cpu_device_rejected_without_launch(runtime,monkeypatch):
    monkeypatch.setenv('TC_APA_SP','1')
    t=runtime.tensor(np.zeros((0,1),np.float32),device='cpu')
    with pytest.raises(RuntimeError,match='rank four'):
        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)
    t=runtime.tensor(np.zeros((0,1,1,1),np.float32),device='cpu')
    with pytest.raises(RuntimeError,match='share CUDA device and dtype'):
        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)


def test_unsupported_dtype_rejected_without_launch(runtime,monkeypatch):
    monkeypatch.setenv('TC_APA_SP','1')
    t=runtime.tensor(np.zeros((0,1,1,1),np.int64),device='cpu',dtype='int64')
    with pytest.raises(RuntimeError,match='supports float32'):
        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)
