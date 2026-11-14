"""Run existing real CPU tests. Prior art: pytest hooks (Krekel et al. 2004), ENG2 cpu_runs.py; no new test algorithm."""
import ast,json,subprocess,sys,os
from pathlib import Path
root=Path.cwd(); out=root/"artifacts/eng2/rev1"; tool=root/"artifacts/eng2/tool"
src=ast.parse((tool.parent/"g2/cpu_runs.py").read_text())
child=next(ast.literal_eval(n.value) for n in src.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CHILD" for t in n.targets))
child=child.replace('"--junitxml=" + xml,', '"--basetemp=" + str(Path(os.environ["TMPDIR"])/("pytest_"+arm)), "--junitxml=" + xml,')
summary=[]
for arm in ["baseline","candidate","MF1","MF2"]:
    args=[sys.executable,"-B","-c",child,str(tool/"arms"/arm/"tensor_cuda"),arm,str(out/(arm+"_cpu.json")),str(out/(arm+"_cpu.xml")),str(tool)]
    r=subprocess.run(args,capture_output=True,text=True,timeout=120)
    (out/(arm+"_cpu.log")).write_text(r.stdout+r.stderr)
    assert r.returncode==0,(arm,r.returncode,r.stderr)
    d=json.loads((out/(arm+"_cpu.json")).read_text())
    fail=[k for k,v in d["outcomes"].items() if v!="passed"]
    summary.append(dict(arm=arm,collected=d["n"],pytest_exit=d["rc"],failed=len(fail),passed=d["n"]-len(fail),traced_failures=sum(bool((d["mutant_trace"] or {}).get(k)) for k in fail)))
(out/"cpu_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
