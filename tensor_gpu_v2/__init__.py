"""tensor_gpu_v2 — GPU-accelerated autograd tensor library built on CuPy.

Quick start::

    import tensor_gpu_v2 as tg

    x = tg.Tensor.randn(32, 784, device='cuda')
    net = tg.Linear(784, 10)
    loss = net(x).mean()
    loss.backward()

See README.md for full installation and usage instructions.
"""

from ._core import *
from ._core import _get_cached_kernel
from ._nn import *
from ._training import *
from ._ext import *
from ._quant import *

__version__ = '2.0.0'
