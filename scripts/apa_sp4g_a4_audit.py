"""Create-only A4 source/payload audit and device visibility.
Prior art: SP3 (2026) SHA receipts; SP4G A3 Frobenius/bitwise comparison;
NVIDIA CUDA Runtime (2024) device count. Descriptive, no new gate/algorithm.
"""
import ctypes
import numpy as np
from apa_sp4g_a4_common import *
from apa_sp4g_a3_common import require_a3
from apa_sp4g_a3_math import compare

def main():
    rows=[]
    for i in (0,1):
        src=require_a3(f'diag_a3_call_l05_c{i}')['result'];m=read(R/src['manifest'])
        arrays=m['arrays']
        def array(name):return np.load(R/arrays[name]['path'],allow_pickle=False)
        rows.append(dict(call_index=i,source=src['manifest'],source_sha256=sha(R/src['manifest']),
          SP_fp32_cast_bf16_vs_SP_bf16=compare(array('sp_fp32_cast_bf16'),array('sp_bf16')),
          SP_fp32_projected_vs_actual_D_projected=compare(array('sp_fp32_projected'),array('actual_D_projected'))))
    publish(A/'a4_suspect_cell_audit.json',dict(evidence_class='source audit plus CPU comparison of historical GPU captures',
      source_finding='A2 explicitly cast Q/K/Kq/V to FP32; native dispatch preserves dtype; historical receipt lacks actual dtype pin. Missing-cast cause not established.',
      A2_A32_ppl=require_a4('diag_a2_fp32_A_2048_w0')['result']['ppl'],
      A2_D32_ppl=require_a4('diag_a2_fp32_2048_w0')['result']['ppl'],
      D_bf16_ppl=require_a4('ppl_D_2048_w0')['result']['ppl'],comparisons=rows,
      limitation='Two A-propagated layer5 calls only. Not a rerun or proof of dtype in the historical A2 worker.'))
    cu=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
    cu.cudaGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)];cu.cudaGetDeviceCount.restype=ctypes.c_int
    cu.cudaGetErrorString.argtypes=[ctypes.c_int];cu.cudaGetErrorString.restype=ctypes.c_char_p
    n=ctypes.c_int();rc=cu.cudaGetDeviceCount(ctypes.byref(n))
    publish(A/'a4_device_visibility.json',dict(cudaGetDeviceCount=rc,device_count=n.value,
      error=cu.cudaGetErrorString(rc).decode(),GPU_workers=0,model_loads=0,
      evidence_class='local CUDA runtime visibility only'))
    before=preserved()
    publish(A/'receipt_audit_A4.json',dict(status='PRESERVED',source_count=len(before['source_sha256']),
      receipt_count=len(before['receipt_sha256']),before_sha256=sha(A/'a4_before.json'),
      source_sha256=before['source_sha256'],receipt_sha256=before['receipt_sha256'],
      original_registration_sha256=REG_SHA,new_registration_sha256=REGISTRATION_SHA,
      historical_reds_preserved=True,large_A2_payloads_retained=True))
    print(json.dumps(rows,indent=2));print('CUDA',rc,n.value)

if __name__=='__main__':
    token=VALIDATION.set({})
    try:main()
    finally:VALIDATION.reset(token)
