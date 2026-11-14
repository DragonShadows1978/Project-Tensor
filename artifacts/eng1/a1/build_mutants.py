"""Fresh isolated source copies; CUDA12.6 builds do not use a device.
Prior art: ENG1 round1 build_mutants.py; mutation DeMillo/Lipton/Sayward (1978).
Taken: one replacement and pristine restoration receipts. Ours: separate fresh A1 scratch trees.
"""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
A=ROOT/'artifacts/eng1/a1'
MODULE='_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
mid=sys.argv[1];m=next(x for x in json.loads((A/'MUTANTS.json').read_text()) if x['id']==mid)
tree=Path('/tmp/eng1_a1_'+mid);src=tree/'tensor_cuda';dest=ROOT/'artifacts/eng1/mutants'/mid
assert not tree.exists() and not dest.exists(), 'fresh copies only'
src.mkdir(parents=True);dest.mkdir()
shutil.copy2(ROOT/'tensor_cuda/CMakeLists.txt',src/'CMakeLists.txt')
for d in ('src','include'):shutil.copytree(ROOT/'tensor_cuda'/d,src/d)
shutil.copytree(ROOT/'tensor_cuda/tensor_cuda',src/'tensor_cuda',ignore=shutil.ignore_patterns('*.so','__pycache__'))
source_map={str(p.relative_to(tree)):sha(p) for p in sorted(src.rglob('*')) if p.is_file()}
(dest/'pristine_sources.json').write_text(json.dumps(source_map,indent=1)+'\n')
target=tree/m['file'];pristine=target.read_text();assert pristine.count(m['old'])==1
target.write_text(pristine.replace(m['old'],m['new'],1));mutated_sha=sha(target)
(dest/'mutation.diff').write_text(''.join(__import__('difflib').unified_diff(pristine.splitlines(True),target.read_text().splitlines(True),fromfile=m['file'],tofile=mid+'/'+m['file'])))
env=dict(os.environ,CUDACXX='/usr/local/cuda-12.6/bin/nvcc',PATH='/usr/local/cuda-12.6/bin:'+os.environ['PATH'],TMPDIR='/tmp/eng1_a1_eng1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
commands=[('configure',['cmake','-S',str(src),'-B',str(tree/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89','-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src','-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],60),('build',['cmake','--build',str(tree/'build'),'-j2'],1050)]
rc=0
for name,cmd,bound in commands:
 start=utc()
 with (dest/(name+'.log')).open('x') as f:
  child=subprocess.run(['timeout',str(bound)+'s',*cmd],env=env,stdout=f,stderr=subprocess.STDOUT)
 rc=child.returncode
 rec=dict(command=cmd,start=start,child_exited=utc(),exit=rc,timeout=rc in (124,137),TMPDIR=env['TMPDIR'],CUDA_VISIBLE_DEVICES='')
 (dest/(name+'.json')).write_text(json.dumps(rec,indent=2)+'\n');print(mid,name,rc,flush=True)
 if rc:break
record=dict(id=mid,exit=rc,decision=m['decision'],file=m['file'],pristine_file_sha256=source_map[m['file']],mutated_file_sha256=mutated_sha)
if rc==0:
 shutil.copy2(src/'tensor_cuda'/MODULE,dest/MODULE);record['module_sha256']=sha(dest/MODULE)
target.write_text(pristine);record['restored_sha256']=sha(target)
assert record['restored_sha256']==record['pristine_file_sha256']
record['scratch_tree']=str(tree);record['recorded']=utc()
(dest/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
sys.exit(rc)
