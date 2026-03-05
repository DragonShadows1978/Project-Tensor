import numpy as np
import tensor_gpu_v2 as tg
import os
import pickle
import sys

# Ensure we are using the local tensor_gpu_v2
sys.path.insert(0, os.path.abspath('.'))
tg.set_device('cpu')

def test_checkpoint_logic():
    print("Testing checkpoint backward logic...")
    x = tg.Tensor([1.0, 2.0], requires_grad=True)
    def fn(a):
        return a * a
    
    y = tg.checkpoint(fn, x)
    z = y.sum()
    
    # Passing a custom gradient to reveal the fix
    grad_out = np.array(0.5)
    z.backward(grad_out)
    
    # Grad of z = sum(x^2) w.r.t x is 2x.
    # If z.grad = 0.5, then x.grad = 0.5 * 2x = x.
    # x is [1.0, 2.0], so x.grad should be [1.0, 2.0]
    expected_grad = np.array([1.0, 2.0])
    
    if x.grad is not None and np.allclose(x.grad, expected_grad):
        print("checkpoint logic: PASS")
    else:
        print(f"checkpoint logic: FAIL, expected {expected_grad}, got {x.grad}")

def test_einsum_exception_swallowing():
    print("Testing einsum exception swallowing...")
    x = tg.Tensor([1.0, 2.0], requires_grad=True)
    y = tg.Tensor([3.0, 4.0], requires_grad=True)
    
    # Intentionally bad subscript for backward
    # The forward will pass, but backward will fail
    z = tg.einsum('i,i->', x, y)
    
    # Mocking a failure in backward by corrupting input_subscripts or similar
    # Actually, let's just use a case that might fail in the current implementation
    
    # The current implementation has:
    # grad_subscript = f"{output_str},{','.join(other_subs)}->{subs}"
    # if output_str is None or empty, it might produce ',i->i' which might be invalid or weird.
    
    z.backward()
    # If it swallowed an exception, we might not know unless we check x.grad
    if x.grad is not None and np.all(x.grad == 0):
        print("Einsum backward: POTENTIAL SILENT FAILURE (grad is zero)")
    else:
        print(f"Einsum backward: grad is {x.grad}")

def test_chunk_closure_bug():
    print("Testing chunk closure bug...")
    x = tg.Tensor(np.arange(4).reshape(2, 2).astype(float), requires_grad=True)
    chunks = tg.chunk(x, 2, dim=0)
    
    # chunks[0] should be x[0, :], chunks[1] should be x[1, :]
    # If there is a closure bug, both _backward functions will use chunks[1].grad
    
    # loss = chunks[0].sum() + chunks[1].sum() * 10
    # chunks[0].grad will be 1, chunks[1].grad will be 10.
    # If bug exists, both will apply 10 to their respective slices?
    # No, both _backward calls will use 'out.grad' where 'out' is the LAST chunk.
    # So chunks[0]._backward will use chunks[1].grad and apply it to x[0, :]
    # chunks[1]._backward will use chunks[1].grad and apply it to x[1, :]
    
    chunks[0].grad = np.array([1.0, 1.0])
    chunks[1].grad = np.array([10.0, 10.0])
    
    chunks[0]._backward()
    chunks[1]._backward()
    
    expected_grad = np.array([[1.0, 1.0], [10.0, 10.0]])
    if np.allclose(x.grad, expected_grad):
        print("Chunk closure: PASS")
    else:
        print(f"Chunk closure: FAIL, expected {expected_grad}, got {x.grad}")

def test_load_checkpoint_security():
    print("Testing load_checkpoint security...")
    path = "vuln_checkpoint.pkl"
    class Malicious:
        def __reduce__(self):
            return (os.system, ('echo VULNERABILITY EXPLOITED',))
    
    checkpoint = {
        'model_state_dict': {'param_0': np.array([1.0])},
        'version': '1.0',
        'malicious': Malicious()
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    class MockModel:
        def __init__(self): self._device = 'cpu'
        def parameters(self): return [tg.Tensor([0.0], requires_grad=False)]
        def load_state_dict(self, state): pass
    
    print("If you see 'VULNERABILITY EXPLOITED' below, the test FAILED (vulnerable).")
    try:
        tg.load_checkpoint(path, MockModel())
        print("load_checkpoint: EXECUTED (might be vulnerable if no exception)")
    except Exception as e:
        print(f"load_checkpoint raised exception: {e}")
    
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    test_checkpoint_logic()
    test_einsum_exception_swallowing()
    test_chunk_closure_bug()
    test_load_checkpoint_security()
