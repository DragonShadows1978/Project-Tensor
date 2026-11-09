"""G4 real installed runner, with scratch ownership files and CPU children only.
Prior art: REV2 hostile_rev2.py Part B/D; Meszaros (2007) fault injection;
GNU timeout/POSIX signals and BSD flock. A3 enumerates every logged outcome,
asserts the registered attribution rule and feeds the real parent/scorer.
"""
import datetime, fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path
from evidence import TOOL, lines, read
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
SCRATCH=Path(os.environ["TMPDIR"])/"runner_controls"
SCRATCH.mkdir(exist_ok=False)
BASE=TOOL/"runner_runs"
BASE.mkdir(exist_ok=False)
REAL=shutil.which("forge-gpu-run")
PY=sys.executable
RESULT=[]

def iso(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
def summary(events):
    (SCRATCH/"summary.log").write_text("".join(iso(t)+" | "+e+"\n" for t,e in events))
def absent():
    p=SCRATCH/"summary.log"
    if p.exists():p.rename(SCRATCH/("summary.saved."+str(time.time_ns())))
def env(runs,**extra):
    e=dict(os.environ,RUNS=str(runs),FORGE_GPU_SUMMARY=str(SCRATCH/"summary.log"),
           FORGE_GPU_WINDOW_FILE=str(SCRATCH/"GPU_WINDOW"),FORGE_GPU_LOCK=str(SCRATCH/"lock"),
           FORGE_GPU_POLL="0.05",FORGE_GPU_KILL_GRACE="0.2",REV2_NO_LOG="1",REV2_MODE="normal")
    e.pop("GN1_AUDIT",None);e.update(extra)
    assert all(Path(e[k]).resolve().is_relative_to(SCRATCH) for k in ("FORGE_GPU_SUMMARY","FORGE_GPU_WINDOW_FILE","FORGE_GPU_LOCK"))
    return e

def invoke(args,e,seconds=90):
    r=subprocess.run(["timeout",str(seconds)+"s",*args],cwd=TOOL,env=e,capture_output=True,text=True)
    return dict(exit=r.returncode,stdout=r.stdout[-2000:],stderr=r.stderr[-2000:])
def start(name):
    runs=BASE/name
    r=invoke([PY,"-B","start_gpu_run.py"],env(runs));assert r["exit"]==0,r
    return runs

def command(label,kind="valid",inner=20):
    line=lines()[label]
    target=([PY,str(TOOL/"rev2_forge.py")] if kind=="valid" else [PY,str(SCRATCH/"sleeper.py")])
    return ["timeout","--kill-after=0.2s",str(inner)+"s",*target,"run_gpu_tests.py",line["arm"],line["suite"]]

def args(label,tail,bound=30,minimum=0.1,maximum=20):
    line=lines()[label];ex=",".join(map(str,line["expected_child_exits"]))
    return [PY,"-B","run_gpu_line.py",label,ex,str(bound),REAL,"--min",str(minimum),"--max",str(maximum),"--margin","0.3","--label","a3-cpu-control","--",*tail]

def collect(runs,label,r):
    paths=sorted((runs/label).glob("attempt-*/parent_result.json"))
    r["receipt"]=str(paths[-1].relative_to(TOOL)) if paths else None
    r["parent"]=read(paths[-1]) if paths else None
    return r

def parent(runs,label,tail,**kw):
    extra=kw.pop("extra",{})
    r=invoke(args(label,tail,**kw),env(runs,**extra));return collect(runs,label,r)
def score(runs,label):
    r=invoke([PY,"-B","score_gpu.py"],env(runs))
    data=read(runs/"G5_SUMMARY.json")
    r["line"]=data["lines"][label];return r

def record(name,runs,label,r,outcome,cause=None,admission="admitted",retry=False):
    second=None
    if retry:
        absent()
        second=parent(runs,label,command(label))
    s=score(runs,label)
    p=r["parent"] or {}
    ok=p.get("forge_outcome")==outcome and p.get("timeout_cause")==cause and p.get("admission")==admission and r["exit"]==1
    if retry:ok &= second["exit"]==0 and s["line"]["passed"] and len(s["line"]["attempts"])==2
    else:ok &= not s["line"]["passed"] and s["exit"]==1
    RESULT.append(dict(control=name,target=f"{outcome}; {cause}; {admission}",passed=bool(ok),parent=r,retry=second,scorer=s))
    print(name,"PASS" if ok else "FAIL",flush=True)

def asynchronous_parent(runs,label,tail,**kw):
    # Foreground probe owns this subprocess and waits synchronously after the
    # required lock/window perturbation; no dispatched/background job exists.
    marker=SCRATCH/"started"
    if marker.exists():marker.rename(SCRATCH/("started.saved."+str(time.time_ns())))
    return subprocess.Popen(["timeout","90s",*args(label,tail,**kw)],cwd=TOOL,env=env(runs),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
def finish(p,runs,label):
    out,err=p.communicate(timeout=90)
    return collect(runs,label,dict(exit=p.returncode,stdout=out[-2000:],stderr=err[-2000:]))
def wait_for(predicate):
    deadline=time.monotonic()+15
    while not predicate():
        assert time.monotonic()<deadline,"CPU probe startup deadline"
        time.sleep(0.02)

(SCRATCH/"sleeper.py").write_text('import os,time,signal\nfrom pathlib import Path\nassert os.environ.get("CUDA_VISIBLE_DEVICES")==""\nif os.environ.get("IGNORE_TERM")=="1":signal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath(os.environ["TMPDIR"],"runner_controls/started").write_text("CPU child")\ntime.sleep(60)\n')
# no-window's two distinct exit forms (usage 64, denied window 75).
runs=start("no_window_75");summary([(time.time(),"RUN7 CYCLE 1 TRAINING")])
r=parent(runs,"baseline_focus",command("baseline_focus"))
record("Runner_no_window75_H5",runs,"baseline_focus",r,"no-window",admission="not_admitted",retry=True)
runs=start("no_window_64");absent()
line=lines()["candidate_focus"]
bad=[PY,"-B","run_gpu_line.py","candidate_focus","0","20",REAL,"--min","2","--max","1","--",*command("candidate_focus")]
r=collect(runs,"candidate_focus",invoke(bad,env(runs)))
record("Runner_no_window64_usage",runs,"candidate_focus",r,"no-window")
# no-lock with an actually held scratch lock.
runs=start("no_lock");summary([(time.time(),"COOLDOWN 2s")])
fd=os.open(SCRATCH/"lock",os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
try:r=parent(runs,"baseline_focus",command("baseline_focus"),minimum=0.5,maximum=2)
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
record("Runner_no_lock76",runs,"baseline_focus",r,"no-lock",admission="not_admitted",retry=True)
# B1: change the window while the real runner is blocked on the scratch lock.
runs=start("rechecked_out");summary([(time.time(),"COOLDOWN 3600s")])
fd=os.open(SCRATCH/"lock",os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
p=asynchronous_parent(runs,"baseline_focus",command("baseline_focus"))
try:
    wait_for(lambda:(runs/"baseline_focus/attempt-0001/stdout.log").exists())
    time.sleep(0.5)
    summary([(time.time(),"COOLDOWN 3500s")])
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
r=finish(p,runs,"baseline_focus")
record("B1_rechecked_out_retry",runs,"baseline_focus",r,"rechecked-out",admission="not_admitted",retry=True)
# B2 runner deadline; B3 clamp to a closed window; B4 parent's outer bound;
# B5 inner timeout. Each is attributed from the runner's raw nine-field row.
runs=start("runner_deadline");absent()
r=parent(runs,"candidate_focus",command("candidate_focus","sleep"),maximum=1.5)
record("B2_runner_deadline",runs,"candidate_focus",r,"timeout","runner-deadline")
runs=start("window_closed");summary([(time.time(),"COOLDOWN 3600s")])
p=asynchronous_parent(runs,"candidate_focus",command("candidate_focus","sleep"))
wait_for(lambda:(SCRATCH/"started").exists());summary([(time.time(),"RUN7 CYCLE 1 TRAINING")])
r=finish(p,runs,"candidate_focus")
record("B3_window_closed",runs,"candidate_focus",r,"window-closed","window-closed")
runs=start("outer_timeout");absent()
r=parent(runs,"candidate_focus",command("candidate_focus","sleep"),bound=2,maximum=60)
record("B4_outer_timeout",runs,"candidate_focus",r,"log-exit-mismatch","outer-timeout")
runs=start("inner_timeout");absent()
r=parent(runs,"candidate_focus",command("candidate_focus","sleep",inner=0.5))
record("B5_inner_timeout124",runs,"candidate_focus",r,"ran","inner-timeout")
runs=start("inner_kill");absent()
r=parent(runs,"candidate_focus",command("candidate_focus","sleep",inner=0.5),extra=dict(IGNORE_TERM="1"))
record("B5_inner_timeout137",runs,"candidate_focus",r,"ran","inner-timeout")
# Ran includes successful valid child, pytest failure and cannot-start 126/127.
for label in ("candidate_focus","baseline_focus"):
    runs=start("ran_"+label);absent()
    r=parent(runs,label,command(label));s=score(runs,label)
    ok=r["exit"]==0 and r["parent"]["forge_outcome"]=="ran" and r["parent"]["timeout_cause"] is None and s["line"]["passed"]
    RESULT.append(dict(control="Runner_ran_"+label,target="valid completed CPU stand-in; ran",passed=ok,parent=r,scorer=s))
for code in (126,127):
    runs=start("ran_cannot_start_"+str(code));absent()
    missing=SCRATCH/"nonexistent" if code==127 else SCRATCH/"nonexecutable"
    if code==126:missing.write_text("not executable")
    # Tail matches the parent's registered command binding but the runner
    # cannot execute the deliberately invalid CPU command at argv[0].
    r=parent(runs,"candidate_focus",[str(missing),"run_gpu_tests.py","candidate","focus"])
    record("Runner_ran_cannot_start"+str(code),runs,"candidate_focus",r,"ran")
result=dict(passed=all(r["passed"] for r in RESULT),controls=RESULT,installed_runner_sha256=__import__('hashlib').sha256(Path(REAL).read_bytes()).hexdigest(),CPU_only=True)
(TOOL/"G4_runner.json").write_text(json.dumps(result,indent=2)+"\n")
print("runner controls",len(RESULT),"passed",result["passed"])
sys.exit(0 if result["passed"] else 1)
