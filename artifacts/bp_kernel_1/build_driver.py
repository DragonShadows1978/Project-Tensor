"""CPU-only authorized build receipt generator, foreground, disconnected."""
import subprocess,json,hashlib,time
from pathlib import Path
root=Path('/mnt/ForgeRealm/wt/pt-bk1');art=root/'artifacts/bp_kernel_1'
commands=[['cmake','-S','tensor_cuda','-B','tensor_cuda/build-bk1','-DCMAKE_BUILD_TYPE=Release',
 '-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc',
 '-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src',
 '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],['cmake','--build','tensor_cuda/build-bk1','-j4']]
r={'evidence_class':'CPU compilation only; no GPU execution','commands':[]};started=time.monotonic()
with (art/'build.log').open('x') as log:
 for cmd in commands:
  log.write('COMMAND '+ ' '.join(cmd)+'\n');log.flush()
  proc=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=570)
  r['commands'].append({'argv':cmd,'rc':proc.returncode})
  if proc.returncode:break
r['elapsed_seconds']=time.monotonic()-started
r['rc']=r['commands'][-1]['rc']
r['last_build_line']=(art/'build.log').read_text().splitlines()[-1]
r['binaries']=[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 for p in (root/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so')]
with (art/'engine_build_receipt.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(r['last_build_line']);print('build_rc='+str(r['rc']))
raise SystemExit(r['rc'])
