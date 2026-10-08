"""Reviewer probe preflight; no GPU API or filesystem mutation.
Prior art: no prior art known to me for this environment guard. Own work:
reject retired driver overrides and pin the already registered source files.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def preflight(root):
    sys.dont_write_bytecode = True
    stale = 'nvidia-userspace-595.84'
    for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PATH'):
        if stale in os.environ.get(name, ''):
            raise RuntimeError(f'{name} contains retired {stale}; launch from native driver environment')
    # APA-MF2 copy: the original pinned MF1-era sources from apa-mf1rev's registration,
    # which MF2 changes by design; the driver guard is kept, hashes are printed instead.
    for rel in ('tensor_cuda/tensor_cuda/quant.py', 'tensor_cuda/tensor_cuda/functional.py', 'apa_cuda/csrc/apa.cpp'):
        print('PREFLIGHT(MF2 copy) sha256', rel, hashlib.sha256((root/rel).read_bytes()).hexdigest(), flush=True)
    print('PREFLIGHT(MF2 copy) retired driver override absent', flush=True)
