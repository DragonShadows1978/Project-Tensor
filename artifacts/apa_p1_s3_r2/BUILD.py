"""A1 build script adapted to this round, with test-only hooks enabled.
Prior art: CMake (Kitware, 2000), nvcc (NVIDIA); no compiler optimization changes.
"""
from pathlib import Path
import os, sys, json, subprocess, time, hashlib, difflib, datetime

A = Path('artifacts/apa_p1_s3_r2')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR='/tmp/apap1s3r2_build')
Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
root = Path.cwd()
source = A/'cmake_source'
source.mkdir()
original = Path('tensor_cuda/CMakeLists.txt').read_text()
adapted = original.replace('  src/', '  '+str(root/'tensor_cuda/src')+'/')
adapted = adapted.replace('${CMAKE_CURRENT_SOURCE_DIR}/include', str(root/'tensor_cuda/include'))
adapted = adapted.replace('${CMAKE_CURRENT_SOURCE_DIR}/tensor_cuda', str(root/A/'build/lib'))
adapted += '\n# Test-only hook; never enabled by the shipped CMake default.\ntarget_compile_definitions(_tensor_cuda PRIVATE TC_APA_P1_TEST_HOOKS=1)\n'
(source/'CMakeLists.txt').write_text(adapted)
(A/'BUILD_PATH_ADAPTER.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True), fromfile='round1/CMakeLists.txt', tofile='A1/cmake_source/CMakeLists.txt')))
commands = [
 ['cmake','-S',str(source),'-B',str(A/'build'),'-DCMAKE_BUILD_TYPE=Release',
  '-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc',
  '-DCUDAToolkit_ROOT=/usr/local/cuda-12.6',
  '-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src',
  '-DFETCHCONTENT_FULLY_DISCONNECTED=ON','-DTC_APA_PACKED=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
 ['cmake','--build',str(A/'build'),'-j2']]
r = dict(attempt=1, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], status='RUNNING', commands=[])
record = A/'BUILD.json'
assert not record.exists()
record.write_text(json.dumps(r,indent=2)+'\n')
for i,cmd in enumerate(commands):
 start = time.monotonic()
 log = A/('configure.log' if i==0 else 'build.log')
 with log.open('x') as f:
  try: code = subprocess.run(['timeout','120s' if i==0 else '1650s',*cmd], stdout=f, stderr=subprocess.STDOUT).returncode
  except Exception as exc: code=125; f.write(repr(exc))
 r['commands'].append(dict(argv=cmd,exit=code,seconds=time.monotonic()-start,log=str(log)))
 if code: r['status']='FAIL'; break
else:
 module = next((A/'build/lib').glob('_tensor_cuda*.so'))
 r.update(status='PASS',module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
 (A/'MODULE.sha256').write_text(r['module_sha256']+'  '+str(module)+'\n')
r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record.write_text(json.dumps(r,indent=2)+'\n')
with (A/'LEDGER.md').open('a') as f: f.write('\n- S4 build attempt 1 '+r['status']+': BUILD.json, configure.log, build.log. Round-1 CMake flags; only source/output paths adapted, no numerical flags changed.\n')
print(json.dumps(r,indent=2))
sys.exit(0 if r['status']=='PASS' else 1)
