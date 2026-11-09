# Prior art: pytest (Krekel et al., 2004+); GN1 arm preload; completed-child subprocess receipt.
# Runs frozen real tests, not replacement fixtures. No GPU paths selected.
import sys,os,json,subprocess,datetime
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'; tool=root/'artifacts/eng1/tool'
arms=sys.argv[1:] or ['candidate','baseline','C_COPY','C_UBSAN','M01','M02','M03','M04','M08','M09','M10','M11','M12','M13']
rows=[]
code="""import sys,pytest; sys.path.insert(0,sys.argv[1]); import tensor_cuda; raise SystemExit(pytest.main(sys.argv[2:]))"""
for arm in arms:
    package=tool/'arms'/arm/'tensor_cuda'
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    cmd=[sys.executable,'-B','-c',code,str(package),'-q','-p','no:cacheprovider','--import-mode=importlib',
         '--rootdir='+str(root),'--basetemp='+os.environ['TMPDIR']+'/cpu_'+arm,
         '--junitxml='+str(rev/('cpu_'+arm+'.xml')),'-k','cpu',str(package/'tests/test_eng1.py')]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=120,env=dict(os.environ))
    (rev/('cpu_'+arm+'.log')).write_text(p.stdout+p.stderr)
    rows.append(dict(arm=arm,command=cmd,start=start,child_exited=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     exit=p.returncode,timeout=False))
    print(arm,p.returncode,p.stdout.splitlines()[-1] if p.stdout.splitlines() else '')
(rev/'cpu_gates.json').write_text(json.dumps(rows,indent=2)+'\n')
