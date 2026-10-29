"""Registered CPU reference and branch-witness mutation checks.
Prior art: Burden & Faires, Numerical Analysis (2011), symmetric central
finite differences; unverified - lead to check. These fixtures are ours. STE and VJP citations in reference.py.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ste_reference', P/'reference.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def difference(fn, x):
    result=np.empty_like(x); eps=1e-6
    for index in np.ndindex(x.shape):
        a=x.copy();b=x.copy();a[index]+=eps;b[index]-=eps
        result[index]=(fn(a)-fn(b))/(2*eps)
    return result


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('kvh',[1,2])
def test_ste_derivatives(causal,kvh):
    q,k,kq,v,do=r.fixture((1,2,3,5,3,2,kvh))
    scale=3**-.5; st=r.forward(q,k,kq,v,scale,1.0364333894937896,causal)
    off=r.backward(q,k,kq,v,do,scale,st);on=r.backward(q,k,kq,v,do,scale,st,True)
    dk=difference(lambda x: np.sum(r.derivative_forward(q,x,k,kq,v,scale,st,causal)*do),k)
    dq=difference(lambda x: np.sum(r.forward(x,k,kq,v,scale,0.,causal,frozen=st[3])[0]*do),q)
    dv=difference(lambda x: np.sum(r.forward(q,k,kq,x,scale,0.,causal,frozen=st[3])[0]*do),v)
    for actual,expected in zip(on,(dq,dk,dv)):
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-5)
    assert np.array_equal(off[0],on[0]); assert np.array_equal(off[2],on[2])
    assert np.all(off[1][:,:,1,:]==0);assert np.any(on[1][:,:,1,:]!=0)
    # Selected-only STE mutant equals off: the registered witness kills it.
    assert not np.allclose(off[1][:,:,1,:],on[1][:,:,1,:],atol=1e-12,rtol=0)


@pytest.mark.parametrize('causal',[False,True])
def test_all_selected_dense(causal):
    q,k,kq,v,do=r.fixture((1,2,4,6,3,2,1));scale=3**-.5
    st=r.forward(q,k,kq,v,scale,-1e6,causal)
    dense=r.forward(q,k,k,v,scale,-1e6,causal)
    off=r.backward(q,k,kq,v,do,scale,st);on=r.backward(q,k,kq,v,do,scale,st,True)
    for a,b,c in zip(off,on,r.backward(q,k,k,v,do,scale,dense,True)):
        np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a,c,atol=2e-12,rtol=0)
    for idx in (0,1,3):
        inputs=[q,k,kq,v];expected=difference(lambda x: dense_loss(inputs,idx,x,do,scale,causal),inputs[idx])
        np.testing.assert_allclose(off[{0:0,1:1,3:2}[idx]],expected,atol=2e-7,rtol=2e-5)


def dense_loss(inputs,index,value,do,scale,causal):
    q,k,kq,v=inputs
    if index==0:q=value
    if index==1:k=value
    if index==3:v=value
    return np.sum(r.forward(q,k,k,v,scale,-1e6,causal)[0]*do)


@pytest.mark.parametrize('causal',[False,True])
def test_off_derivative(causal):
    q,k,kq,v,do=r.fixture((1,2,3,5,3,2,1));scale=3**-.5
    st=r.forward(q,k,kq,v,scale,1.0364333894937896,causal)
    dk=difference(lambda x: np.sum(r.forward(q,x,kq,v,scale,0.,causal,frozen=st[3])[0]*do),k)
    np.testing.assert_allclose(r.backward(q,k,kq,v,do,scale,st)[1],dk,atol=2e-7,rtol=2e-5)
