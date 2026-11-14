# L82 tooling import-site dispositions

Prior art: PSF Python import/environment semantics; REV1 F02. Every current `PYTHONPATH|sys.path|preload` hit in tool/*.py appears below.

| Site | Text | Disposition |
|---|---|---|
| arm_origin.py:1 | `"""Subprocess arm binding. Prior art: Python/PSF inherited environment and PYTHONPATH import search;` | prepend pinned package parent to inherited PYTHONPATH; fresh subprocess checks it and pins |
| arm_origin.py:11 | `print(json.dumps(dict(tensor_cuda_file=str(Path(tc.__file__).resolve()),engine_file=str(p),engine_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cwd=os.getcwd(),PYTHONPATH=os.environ.get('PYTHONPATH'))))` | prepend pinned package parent to inherited PYTHONPATH; fresh subprocess checks it and pins |
| arm_origin.py:17 | `env['PYTHONPATH']=parent+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')` | prepend pinned package parent to inherited PYTHONPATH; fresh subprocess checks it and pins |
| arm_origin.py:30 | `assert actual['PYTHONPATH']==record['PYTHONPATH'], 'subprocess environment mismatch'` | prepend pinned package parent to inherited PYTHONPATH; fresh subprocess checks it and pins |
| arm_origin.py:31 | `assert actual['PYTHONPATH'].split(os.pathsep)[0]==str(package), 'arm PYTHONPATH must be first'` | prepend pinned package parent to inherited PYTHONPATH; fresh subprocess checks it and pins |
| arm_origin.py:47 | `PYTHONPATH=env.get('PYTHONPATH'),expected_pin=pin)` | prepend pinned package parent to inherited PYTHONPATH; fresh subprocess checks it and pins |
| run_gpu_tests.py:6 | `control / regression; the regression suite runs the worktree's existing test files with the arm package preloaded` | preload stays for current process; inherited environment now bound before tests/subprocesses |
| run_gpu_tests.py:83 | `# Process-local preloading is insufficient for test_dlpack's fresh Python subprocesses (REV1 F02).` | preload stays for current process; inherited environment now bound before tests/subprocesses |
| run_gpu_tests.py:87 | `sys.path.insert(0, str(package))` | preload stays for current process; inherited environment now bound before tests/subprocesses |
| run_gpu_tests.py:88 | `sys.path.insert(1, str(ROOT))` | preload stays for current process; inherited environment now bound before tests/subprocesses |
| stage.py:5 | `census    collect each suite's test IDs on CPU with the arm package preloaded (ids_<census>.txt)` | collection child receives bound PYTHONPATH; tool bootstrap stays process-local |
| stage.py:95 | `sys.path.insert(0, package)` | collection child receives bound PYTHONPATH; tool bootstrap stays process-local |
| stage.py:298 | `sys.path.insert(0, str(TOOL))` | collection child receives bound PYTHONPATH; tool bootstrap stays process-local |
