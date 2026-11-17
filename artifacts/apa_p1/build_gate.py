# Prior art: CMake (Kitware 2000) / nvcc (NVIDIA CUDA); standard build orchestration.
from pathlib import Path
import subprocess,os,time,json,hashlib,datetime,shutil
A=Path('artifacts/apa_p1'); root=Path.cwd(); build=root/'tensor_cuda/build-p1'
ref=Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ab1/artifacts/apa_ab1/build/main/_tensor_cuda.cpython-312-x86_64-linux-gnu.so')
prod=Path('/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(prod)=='d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24'
assert sha(ref)=='daa0b10c87222775a16b52a6811962b682319394c0f31675a924e68f5192933f'
for p,minfree in [(root,25_000_000_000),(Path('/'),20_000_000_000)]: assert shutil.disk_usage(p).free>=minfree,p
commands=[['cmake','-S','tensor_cuda','-B','tensor_cuda/build-p1','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc','-DCUDAToolkit_ROOT=/usr/local/cuda-12.6','-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src','-DFETCHCONTENT_FULLY_DISCONNECTED=ON','-DTC_APA_PACKED=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],['cmake','--build','tensor_cuda/build-p1','-j2']]
r=dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='RUNNING',production_module=str(prod),production_sha256=sha(prod),lab_module=str(ref),lab_sha256=sha(ref),nvcc='/usr/local/cuda-12.6/bin/nvcc',defines=['TC_APA_PACKED=1','TC_APA_EB2B=1','TC_APA_EB3=1','TC_APA_EB4=1','APA_EB2B_MUTANT=0','APA_FXTC1_VARIANT_DEFAULT=2'],commands=[],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
(A/'build.json').write_text(json.dumps(r,indent=2)+'\n')
for i,cmd in enumerate(commands):
    start=time.monotonic(); log=A/('configure.log' if i==0 else 'build.log')
    with log.open('w') as f:
        try: code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=120 if i==0 else 1450).returncode
        except subprocess.TimeoutExpired: code=124
    r['commands'].append(dict(argv=cmd,exit=code,seconds=time.monotonic()-start,log=str(log)))
    if code: r['status']='FAIL'; break
else:
    module=next((root/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    saved=build/module.name; shutil.copy2(module,saved)
    r.update(status='PASS',module=str(module),module_sha256=sha(module),saved_module=str(saved))
    (A/'module.sha256').write_text(sha(module)+'  '+str(module)+'\n')
r['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat(); (A/'build.json').write_text(json.dumps(r,indent=2)+'\n')
with (A/'LEDGER.md').open('a') as f:f.write('\n- G1 PASS: g1.json, g1_tracked.diff, g1_diff_stat.txt. 16 identical headers, five appended seams, zero removed lines.\n- G2 build '+r['status']+': build.json, configure.log, build.log. CUDA 12.6, sm89, Release; offline read-only pybind11 source. Main defines include explicit variant B and nonmutant EB2b; no K1 AOT wrapper. Lab module resolved via committed g0/build.json and verified checksum.\n')
print(json.dumps(r,indent=2)); raise SystemExit(0 if r['status']=='PASS' else 1)
