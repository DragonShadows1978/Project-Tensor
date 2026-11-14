"""Subprocess arm binding. Prior art: Python/PSF inherited environment and PYTHONPATH import search;
SHA-256 (NIST FIPS180-4, 2015). Taken: ordinary subprocess origin checks. Ours: per-line sealed pin receipt.
External historical references unverified -- lead to check. No CUDA APIs called.
"""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path

ORIGIN_CODE = """import hashlib,json,os,tensor_cuda as tc
from pathlib import Path
p=Path(tc._C.__file__).resolve()
print(json.dumps(dict(tensor_cuda_file=str(Path(tc.__file__).resolve()),engine_file=str(p),engine_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cwd=os.getcwd(),PYTHONPATH=os.environ.get('PYTHONPATH'))))
"""

def bound_environment(package, inherited=None):
    env=dict(os.environ if inherited is None else inherited)
    parent=str(Path(package).resolve())
    env['PYTHONPATH']=parent+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def validate_origin(record, package, module_name, pin, cwd):
    package=Path(package).resolve()
    assert record['exit']==0 and not record['timeout'], 'subprocess origin check failed'
    actual=json.loads(record['stdout'])
    assert actual==record['observed'], 'origin stdout/record mismatch'
    assert actual['tensor_cuda_file']==str(package/'tensor_cuda/__init__.py'), 'subprocess package origin mismatch'
    assert actual['engine_file']==str(package/'tensor_cuda'/module_name), 'subprocess engine origin mismatch'
    assert actual['engine_sha256']==pin, 'subprocess pin mismatch'
    assert actual['cwd']==str(Path(cwd).resolve())==record['cwd'], 'subprocess cwd mismatch'
    assert actual['PYTHONPATH']==record['PYTHONPATH'], 'subprocess environment mismatch'
    assert actual['PYTHONPATH'].split(os.pathsep)[0]==str(package), 'arm PYTHONPATH must be first'
    return actual


def preflight(package, module_name, pin, cwd, output, env=None):
    env=dict(os.environ if env is None else env)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        p=subprocess.run([sys.executable,'-B','-c',ORIGIN_CODE],cwd=cwd,env=env,capture_output=True,text=True,timeout=60)
        rc,stdout,stderr,timed_out=p.returncode,p.stdout,p.stderr,False
    except subprocess.TimeoutExpired as exc:
        rc,stdout,stderr,timed_out=124,str(exc.stdout or ''),str(exc.stderr or ''),True
    try: observed=json.loads(stdout)
    except (ValueError,TypeError): observed=None
    rec=dict(start=start,child_exited=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit=rc,
             stdout=stdout,stderr=stderr,timeout=timed_out,observed=observed,cwd=str(Path(cwd).resolve()),
             PYTHONPATH=env.get('PYTHONPATH'),expected_pin=pin)
    # Receipt survives a failed check; refusal occurs before pytest/test imports.
    Path(output).write_text(json.dumps(rec,indent=2)+'\n')
    validate_origin(rec,package,module_name,pin,cwd)
    return rec
