"""Prior art: BP-KERNEL-3 offline CMake build recipe (2026), taken unchanged except output directory and source receipt; no new algorithm."""
from pathlib import Path
import subprocess,time,json,hashlib,sys,os
r=Path(__file__).resolve().parents[2];a=r/'artifacts/bp_kernel_4'
reg=json.loads((a/'registration.json').read_text())
commands=[['cmake','-S','tensor_cuda','-B','tensor_cuda/build-bk4','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc','-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src','-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],['cmake','--build','tensor_cuda/build-bk4','-j4']]
started=time.monotonic();results=[];rc=1;last='not started'
with (a/'build.log').open('x') as log:
 for cmd in commands:
  run=subprocess.run(cmd,cwd=r,capture_output=True,text=True,timeout=max(1,480-(time.monotonic()-started)),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
  log.write(run.stdout+run.stderr);log.flush();results.append(dict(argv=cmd,rc=run.returncode));rc=run.returncode
  lines=run.stdout.strip().splitlines();last=lines[-1] if lines else run.stderr.strip().splitlines()[-1]
  if rc:break
bs=[dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in (r/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so')] if rc==0 else []
with (a/'engine_build_receipt.json').open('x') as f:json.dump(dict(evidence_class='CPU compilation only; no GPU execution',commands=results,elapsed_seconds=time.monotonic()-started,rc=rc,last_build_line=last,binaries=bs,source_pins={p:h for p,h in reg['pins'].items() if p.startswith('tensor_cuda/')}),f,indent=2)
print(last);print(f'BUILD_RC={rc}');sys.exit(rc)
