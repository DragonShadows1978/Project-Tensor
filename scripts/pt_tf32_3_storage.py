"""PT-TF32-3 storage/IPC protocol, CPU-only import.

Prior art: POSIX pipes (1973), Python subprocess (2003), NumPy .npy array
format (2007) and SHA256 (NIST 2001), taken. Ours: bounded, exact gradient
transport without persistent dumps, canonical content receipts and 8 GiB rail.
No lossy gradient sketch and no changed cosine gate.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import numpy as np

MIN_FREE_BYTES=8*1024**3
MAX_GRAD_BYTES=2*1024**3
MAX_ARRAYS=4096


def space_status(path):
    path=Path(path).resolve()
    while not path.exists():path=path.parent
    free=shutil.disk_usage(path).free
    return dict(status='GREEN' if free>=MIN_FREE_BYTES else 'BLOCKED',
                reason=None if free>=MIN_FREE_BYTES else 'free space below 8 GiB',
                free_bytes=free,required_free_bytes=MIN_FREE_BYTES,filesystem_path=str(path))


def require_space(path):
    row=space_status(path)
    if row['status']!='GREEN':raise StorageBlocked(row)
    return row


class StorageBlocked(RuntimeError):
    def __init__(self,row):
        self.receipt=row
        super().__init__(json.dumps(row,sort_keys=True))


def array_header(grads):
    if not grads or len(grads)>MAX_ARRAYS:raise ValueError('empty/excessive gradients')
    rows=[];total=0
    for k in sorted(grads):
        a=np.asarray(grads[k])
        if a.dtype!=np.float32 or not a.flags.c_contiguous or not a.size:
            raise ValueError('gradients require nonempty contiguous FP32 arrays')
        total+=a.nbytes
        rows.append(dict(name=k,shape=list(a.shape),bytes=a.nbytes))
    if total>MAX_GRAD_BYTES:raise ValueError('gradient payload exceeds 2 GiB')
    return rows


def canonical_sha(grads):
    rows=array_header(grads)
    h=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())
    for row in rows:h.update(memoryview(grads[row['name']]).cast('B'))
    return h.hexdigest()


def send_grads(stream,grads):
    rows=array_header(grads)
    header=json.dumps(rows,separators=(',',':')).encode()
    stream.write(struct.pack('<Q',len(header)));stream.write(header)
    for row in rows:
        # Raw bytes have a validated length and FP32 dtype; no pickle loader.
        stream.write(memoryview(grads[row['name']]).cast('B'))
    stream.flush()


def read_exact(stream,size):
    value=bytearray(size);view=memoryview(value);at=0
    while at<size:
        n=stream.readinto(view[at:])
        if not n:raise EOFError('incomplete gradient pipe')
        at+=n
    return value


def receive_grads(stream):
    size=struct.unpack('<Q',read_exact(stream,8))[0]
    if not 1<=size<=1024**2:raise ValueError('invalid gradient header size')
    rows=json.loads(read_exact(stream,size));grads={};total=0
    if not isinstance(rows,list) or not 1<=len(rows)<=MAX_ARRAYS:raise ValueError('invalid gradient count')
    for row in rows:
        name=row['name'];shape=row['shape'];count=1
        if not isinstance(name,str) or name in grads or not isinstance(shape,list) or len(shape)>8:
            raise ValueError('invalid gradient metadata')
        for n in shape:
            if type(n)!=int or n<=0:raise ValueError('invalid gradient dimension')
            count*=n
        size=count*4;total+=size
        if size!=row['bytes'] or total>MAX_GRAD_BYTES:raise ValueError('oversized gradient payload')
        grads[name]=np.frombuffer(read_exact(stream,size),np.float32).reshape(shape)
    if stream.read(1):raise ValueError('trailing gradient payload')
    return grads


def dump_bytes(path):
    """Report real retained gradient/checkpoint bytes, separate from small logs."""
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and
               (p.name=='grad.npz' or '.ckpt' in p.name))
