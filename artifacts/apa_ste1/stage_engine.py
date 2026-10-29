"""CPU-only package/manifest staging; never publishes outside the lab fork.
Prior art: NIST FIPS 180-4 (2015) SHA-256; cc42-engine-manifest-v1 (2026).
Unverified - lead to check standard. Ours: STE variant/provenance fields.
"""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent;ROOT=P.parents[1]
base=json.loads(Path('/mnt/ForgeRealm/engines/tf32-det-69b154cd2e9a/ENGINE_MANIFEST.json').read_text())
engine=P/'engine';container=engine/'tensor_cuda';pkg=container/'tensor_cuda';pkg.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

files={}
for name in base['files']:
    src=ROOT/'tensor_cuda/tensor_cuda'/name;dest=pkg/name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);files[name]=sha(dest)
    if name!=base['so']:assert files[name]==base['files'][name],name
sources={str(p.relative_to(ROOT)):sha(p) for folder in ['tensor_cuda/src','tensor_cuda/include'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
sources['tensor_cuda/CMakeLists.txt']=sha(ROOT/'tensor_cuda/CMakeLists.txt')
manifest={
 'schema':'cc42-engine-manifest-v1','name':'apa-dk-ste','variant':'apa-dk-ste','switch':'TC_APA_DK_STE','default':0,
 'engine_dir':str(engine),'engine_path':str(container),'package':str(pkg),
 'files':files,'so':base['so'],'so_path':str(pkg/base['so']),'so_sha256':files[base['so']],
 'base_engine_so_sha256':base['so_sha256'],'base_engine_dir':base['engine_dir'],
 'source_commit':'f2c686e','source_commit_provenance':'user order; no git invoked; kernels baseline hash verified f1951814eae4f8183c3a2d3cd9b86eda9a5e0df74def7f07e5393e3c3f67fb2b',
 'source':{'path':str(ROOT/'tensor_cuda'),'package':str(ROOT/'tensor_cuda/tensor_cuda')},
 'source_file_hashes':sources,'source_references_checked':[str(ROOT/'tensor_cuda'),base['source']['path']],
 'restored_source_receipt':str(P/'receipts/restored_source.json'),
 'runpath':base['runpath'],'toolchain':{'cuda':'12.6','nvcc':'/usr/local/cuda-12.6/bin/nvcc','cxx':'/usr/bin/c++','gcc_version':'13.3.0','cuda_architectures':['89'],'pybind11':'2.12.0'},
 'build_command':'CUDA_VISIBLE_DEVICES= timeout --foreground 1050s bash artifacts/apa_ste1/build.sh',
 'build_script_sha256':sha(P/'build.sh'),'build_log':str(P/'receipts/build_final.log'),
 'installed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'CPU-built; GPU gates NOT RUN; cross-model review pending; not approved for training or promotion',
 'rollback':'default off; immutable base engine remains unchanged; no installation outside lab fork',
 'prior_art':'Bengio, Leonard and Courville (2013) STE identity derivative; Dao et al. (2022) FlashAttention recomputation (existing); NIST FIPS 180-4 (2015) SHA-256; unverified - lead to check',
}
(engine/'ENGINE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'so_sha256':manifest['so_sha256'],'package_files':len(files),'source_files':len(sources),'manifest':str(engine/'ENGINE_MANIFEST.json')}))
