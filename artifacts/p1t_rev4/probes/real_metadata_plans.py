"""PASS iff every real Conv/ConvTranspose/LSTM/norm/Round fixture descriptor plans its recorded output shape.
CPU-only: descriptors faithfully export real metadata, CUDA data addresses are placeholders NEVER dereferenced.
Prior art: ONNX operator geometry + DLPack metadata2017; ours: checking reviewed plans against fixture headers.
"""
import ctypes as C, json, sys
from pathlib import Path
ROOT=Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev4')
sys.path.insert(0,str(ROOT/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLDevice,DLDataType
import tc_signal as ts
n=ts._C
fixture=Path('/mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures')
m=json.loads((fixture/'manifest.json').read_text())
census=json.loads((ROOT/'artifacts/p1t_rev4/real_data_census.json').read_text())
weights={p['path']:p['arrays'] for p in census['weights']}
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
keep=[]
def cap(shape):
    shape=tuple(shape);sh=(C.c_int64*len(shape))(*shape);mt=DLManagedTensor();t=mt.dl_tensor
    t.data=0x1000;t.device=DLDevice(2,0);t.dtype=DLDataType(2,32,1);t.ndim=len(shape);t.shape=sh
    keep.extend([sh,mt]);return new(C.addressof(mt),b'dltensor',None)
rows=[]
for c in m['cases']:
    op=c['op_type'];attrs=c['attrs'];sh=c['shapes'];x=cap(sh['x'])
    w=weights.get(str(fixture/c['weights_file']),{}) if c.get('weights_file') else {}
    def wc(k):return cap(w[k+'.npy']['shape']) if k+'.npy' in w else None
    expected=tuple(sh.get('y',sh['x']))
    if op=='Conv':
        p=n.conv1d_plan(x,wc('W'),wc('B'),attrs['strides'][0],*attrs['pads'],attrs['dilations'][0],attrs['group']);got=tuple(p[3])
    elif op=='ConvTranspose':
        p=n.conv_transpose1d_plan(x,wc('W'),wc('B'),attrs['strides'][0],*attrs['pads'],attrs.get('output_padding',[0])[0],attrs['dilations'][0],attrs['group']);got=tuple(p[3])
    elif op=='LSTM':
        p=n.lstm_plan(x,wc('W'),wc('R'),wc('B'),cap(sh['initial_h']),cap(sh['initial_c']),attrs['hidden_size'],{'forward':0,'reverse':1,'bidirectional':2}[attrs['direction']]);got=tuple(p[6])
        assert tuple(p[7])==tuple(sh['y_h'])==tuple(sh['y_c'])
    elif op=='InstanceNormalization':got=tuple(n.instance_norm1d_plan(x,wc('scale'),wc('B'),attrs['epsilon'])[3])
    elif op=='LayerNormalization':got=tuple(n.layer_norm_plan(x,wc('scale'),wc('B'),attrs.get('axis',-1),attrs['epsilon'])[3])
    elif op=='GroupNorm':got=tuple(n.group_norm_plan(x,attrs['num_groups'],None,None,attrs['epsilon'])[3])
    elif op=='Round':got=tuple(n.rint_plan(x)[1])
    else:continue # Resize scale and Pad values need case-array values, covered by separate lead probes.
    assert got==expected,(c['id'],got,expected)
    rows.append({'id':c['id'],'shape':got,'ok':True})
(ROOT/'artifacts/p1t_rev4/real_metadata_plans.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS real metadata geometry',len(rows),'cases; no GPU/data/numerical claim')
