# Prior art: Project-Tensor AB-1/SPQ3/EB4 (2026) captures and accessor;
# immutable Git blobs (Git 2005) and SHA-256 (NIST 2015). No new algorithm.
from pathlib import Path
import subprocess,json,hashlib,shutil,sysconfig
A=Path('artifacts/apa_p1'); B=json.loads((A/'build.json').read_text())
def blob(path,ref='aaa9b4ea'):return subprocess.check_output(['git','show',ref+':'+path])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for arm,ref in [('lab','aaa9b4ea'),('production','0e1dea3')]:
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',ref,'--','tensor_cuda/tensor_cuda/'],text=True).splitlines()
    for p in paths:
        if p.endswith('.py'):
            dst=A/'refs'/arm/Path(p).relative_to('tensor_cuda');dst.parent.mkdir(parents=True,exist_ok=True)
            # Same additive Python surface for baseline regression; loader fails closed to _spo2=None.
            dst.write_bytes(blob(p,ref) if arm=='lab' else Path(p).read_bytes())
    if arm=='production':shutil.copy2(B['production_module'],A/'refs'/arm/'tensor_cuda'/Path(B['production_module']).name)
(A/'ab1_interop.cpp').write_bytes(blob('scripts/apa_ab1_interop.cpp'))
(A/'reference_g0_build.json').write_bytes(blob('artifacts/apa_ab1/g0/build.json'))
(A/'reference_g2.json').write_bytes(blob('artifacts/apa_ab1/g2/result.json'))
(A/'reference_capture_sha.json').write_bytes(blob('artifacts/apa_spq6/g1_captures_sha.json'))
root=Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree')
short_hash=json.loads((A/'reference_capture_sha.json').read_text())['files']; rows=[]
for model,layer in [('Qwen3-1.7B',5),('Qwen3-4B',7)]:
    long_manifest=json.loads(blob(f'artifacts/apa_eb4/p0/captures/{model}/manifest.json'))
    (A/f'reference_32k_{model}.json').write_text(json.dumps(long_manifest,indent=2)+'\n')
    for S in (2048,8192,32768):
        source=root/('tensor_fork_apa_wt_eb4/artifacts/apa_eb4/p0/captures' if S==32768 else 'tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures')/model
        files={}
        for key in 'qkv':
            name=f'L{layer:02d}_{key}.npy';p=source/name
            want=long_manifest['files'][name]['sha256'] if S==32768 else short_hash[f'{model}/{name}']['sha256']
            got=sha(p); assert got==want,(p,got,want)
            import numpy as np
            a=np.load(p,mmap_mode='r');assert a.dtype==np.uint16
            files[key]=dict(path=str(p),sha256=got,shape=list(a.shape),dtype=str(a.dtype))
        rows.append(dict(model=model,layer=layer,S=S,B=1,KVH=8,H=16 if model=='Qwen3-1.7B' else 32,D=128,files=files,query_position=S-1,source_query_position=32767 if S==32768 else S-1,fraction=0.02 if model=='Qwen3-1.7B' else 0.05,append_lengths=[1,63,64,65,S-4,S-3,S-2,S-1,S],scope='faithful uint16 bf16 export; after q_norm/k_norm and RoPE; prefix at S, last query row. One captured layer/model. Numerical identity, not model quality.'))
(A/'INPUT_CENSUS.json').write_text(json.dumps(rows,indent=2)+'\n')
labinterop=root/'tensor_fork_apa_wt_ab1/artifacts/apa_ab1/build/interop'/('_apa_ab1_interop'+sysconfig.get_config_var('EXT_SUFFIX'))
assert sha(labinterop)=='431784a53cde396c60305c59a276c8fa1c37d1572ea1c830285bcf753e61dc32'
print('Pinned six real-capture cases, lab Python at aaa9b4ea, production comparison runtime, and G0/G2 reference receipts.')
