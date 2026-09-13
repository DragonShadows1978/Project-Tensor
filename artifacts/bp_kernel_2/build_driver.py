from pathlib import Path
import subprocess,time,json,hashlib
r=Path(__file__).resolve().parents[2]; a=r/'artifacts/bp_kernel_2'
commands=[['cmake','-S','tensor_cuda','-B','tensor_cuda/build-bk2','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc','-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src','-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],['cmake','--build','tensor_cuda/build-bk2','-j4']]
started=time.monotonic(); results=[]
with (a/'build.log').open('x') as log:
 for cmd in commands:
  run=subprocess.run(cmd,cwd=r,capture_output=True,text=True,timeout=480)
  log.write(run.stdout+run.stderr);log.flush(); results.append(dict(argv=cmd,rc=run.returncode))
  if run.returncode: break
last=(run.stdout+run.stderr).strip().splitlines()[-1]
bs=[dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in (r/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so')]
with (a/'engine_build_receipt.json').open('x') as f: json.dump(dict(evidence_class='CPU compilation only; no GPU execution',commands=results,elapsed_seconds=time.monotonic()-started,rc=run.returncode,last_build_line=last,binaries=bs),f,indent=2)
print(last)
raise SystemExit(run.returncode)
