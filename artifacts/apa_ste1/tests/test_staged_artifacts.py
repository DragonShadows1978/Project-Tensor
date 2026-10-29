"""CPU artifact checks using the actual staged binaries and serialized inputs.
Prior art: SHA-256 NIST FIPS 180-4 (2015), shell argument parsing (POSIX 1988);
unverified - lead to check. No GPU execution or simulated GPU verdict.
"""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import numpy as np

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_engine_manifest_actual_bytes():
    m=json.loads((P/'engine/ENGINE_MANIFEST.json').read_text())
    assert m['schema']=='cc42-engine-manifest-v1'
    assert m['variant']=='apa-dk-ste' and m['switch']=='TC_APA_DK_STE' and m['default']==0
    assert m['base_engine_so_sha256']=='69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c'
    for name,pin in m['files'].items():assert sha(Path(m['package'])/name)==pin
    for name,pin in m['source_file_hashes'].items():assert sha(ROOT/name)==pin
    assert sha(Path(m['so_path']))==m['so_sha256']


def test_serialized_real_geometry_inputs():
    m=json.loads((P/'INPUTS.json').read_text())
    for name,info in m.items():
        path=P/'inputs'/f'{name}.npz';assert sha(path)==info['sha256']
        B,H,L,S,D,VD,KVH=info['shape']
        with np.load(path) as data:
            for key,shape in {'q':(B,H,L,D),'k':(B,KVH,S,D),'kq':(B,KVH,S,D),'v':(B,KVH,S,VD),'do':(B,H,L,VD)}.items():
                x=data[key];assert x.shape==shape and x.dtype==np.float32
                assert np.isfinite(x).all() and np.any(x!=0) and np.any(x!=x.flat[0])
            assert np.all(data['kq'][:,:,1,:]==0)
            if name=='grapa':
                assert info['shape']==[1,16,4096,4096,128,64,16]
                assert np.all(data['q'][:,:,0,:]==.25) and np.all(data['kq'][:,:,0,:]==.25)


def test_lead_commands_governed_and_bounded():
    script=P/'GPU_COMMANDS.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    jobs=[shlex.split(s) for s in script.read_text().splitlines() if s.startswith('forge-gpu-run ')]
    assert len(jobs)==14
    labels=[]
    for args in jobs:
        labels.append(args[args.index('--label')+1])
        assert args[args.index('--max')+1]=='600'
        at=args.index('timeout');assert args[at:at+3]==['timeout','--foreground','300s']
        assert 'APA_STE1_LEAD_GPU=1' in args and 'TC_DETERMINISTIC=1' in args
        assert 'TC_DET_EMBED_BWD=1' in args and 'CUDA_VISIBLE_DEVICES=0' in args
    assert len(set(labels))==len(labels)
    assert 'trap finish_receipt EXIT' in script.read_text()
