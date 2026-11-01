import os
os.environ["GN1_CONTROL"] = "1"  # A6 D1: CPU fixture root cannot accept.
"""REV4 reproductions and finite parser-spelling census through real A5 tools.

Prior art: REV4 hostile_rev4.py (2026), Meszaros fault injection (2007),
Python argparse allow_abbrev (PSF, 2015), GNU coreutils timeout(1).
The census enumerates every unambiguous option-name prefix and its supported
argument attachment forms; value domains and arbitrary repeated combinations
are infinite and represented by additional named grammar-class controls.
"""
import datetime
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
from control_transport import adapt, executable
from evidence import TOOL, lines, read, bound_reasons, timeout_cause

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
BASE = TOOL / "a5_runs"
BASE.mkdir(exist_ok=False)
SCRATCH = Path(os.environ["TMPDIR"]) / "a5_controls"
SCRATCH.mkdir(exist_ok=False)
REAL = shutil.which("forge-gpu-run")
PY = sys.executable
LINE = lines()["candidate_focus"]
RESULT = []


def env(runs, **extra):
    e = dict(os.environ, RUNS=str(runs), REV2_MODE="normal", REV2_NO_LOG="1",
             REV4_MA4=str(TOOL), FORGE_GPU_SUMMARY=str(SCRATCH/"summary.log"),
             FORGE_GPU_WINDOW_FILE=str(SCRATCH/"GPU_WINDOW"), FORGE_GPU_LOCK=str(SCRATCH/"lock"),
             FORGE_GPU_KILL_GRACE="10", FORGE_GPU_POLL="5")
    e.pop("GN1_AUDIT", None)
    for key, value in extra.items():
        if value is None:
            e.pop(key, None)
        else:
            e[key] = value
    return e


def run(argv, e, bound=80):
    with (BASE / "LAUNCHES.jsonl").open("a") as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               argv=argv, bound=bound, RUNS=e["RUNS"],
                               grace=e.get("FORGE_GPU_KILL_GRACE"), poll=e.get("FORGE_GPU_POLL"))) + "\n")
    p = subprocess.run(["timeout", "--kill-after=2s", f"{bound}s", *argv], cwd=TOOL,
                       env=e, capture_output=True, text=True)
    return dict(exit=p.returncode, stdout=p.stdout[-1500:], stderr=p.stderr[-3000:],
                traceback="Traceback" in p.stderr)


def start(name):
    runs = BASE/name
    assert run([PY, "-B", "start_gpu_run.py"], env(runs))["exit"] == 0
    return runs


def canonical(runs, **extra):
    old = [PY, "-B", "run_gpu_line.py", "candidate_focus", "0", str(LINE["outer"]),
           "forge-gpu-run", "--min", "250", "--max", "300", "--label", "candidate_focus", "--",
           "timeout", "--kill-after=10s", "240s", PY, str(TOOL/"rev2_forge.py"),
           "run_gpu_tests.py", "candidate", "focus"]
    return adapt(old, env(runs, **extra))


def parent(argv, e):
    r = run(argv, e)
    attempts = sorted((Path(e["RUNS"])/"candidate_focus").glob("attempt-*/parent_result.json"))
    try:
        r["parent"] = read(attempts[-1]) if attempts else None
    except (OSError, ValueError) as exc:
        r["parent"] = None
        r["prior_receipt_read_error"] = str(exc)
    return r


def score(runs):
    r = run([PY, "-B", "score_gpu.py"], env(runs))
    r["line"] = read(runs/"G5_SUMMARY.json")["lines"]["candidate_focus"]
    return r


def record(name, rule, ok, **data):
    RESULT.append(dict(control=name, target=rule, passed=bool(ok), **data))
    (TOOL/"G4_a5.json").write_text(json.dumps(dict(passed=all(r["passed"] for r in RESULT), controls=RESULT), indent=2)+"\n")
    print(name, "PASS" if ok else "FAIL", flush=True)


def refusal(name, runner=None, inner=None, parser=None):
    runs = start(name)
    argv, e = canonical(runs)
    at = argv.index("--")
    if runner is not None:
        argv[7:at] = runner
        at = argv.index("--")
    if inner is not None:
        argv[at+1:] = inner
    r = parent(argv, e)
    s = score(runs)
    binding = bound_reasons(r["parent"], LINE) if r["parent"] else []
    record(name, "D2 exact runner/inner argv refusal", r["parent"] is not None and bool(binding)
           and not s["line"]["passed"] and not r["traceback"] and not s["traceback"],
           parser=parser, bound_reasons=binding, parent=r, scorer=s)


def census():
    # Parser-only execution never calls runner main, takes a lock or reads a
    # window. Record source and accepted namespaces as operand evidence.
    ns = runpy.run_path(REAL, run_name="gn1_a8_parser")
    import contextlib
    import io
    options = ["--min", "--max", "--margin", "--self-leasing", "--label", "--status", "--help"]
    values = {"--min":"1", "--max":"900", "--margin":"30", "--label":"alternate"}
    runner = []
    for full in options:
        for n in range(3, len(full)+1):
            option = full[:n]
            forms = [[option, values[full]], [option+"="+values[full]]] if full in values else [[option]]
            for form in forms:
                trial = ["--min","250","--max","300","--label","candidate_focus",*form]
                if full != "--status":
                    trial += ["--","true"]
                accepted, detail = False, None
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        parsed = ns["options"](trial)
                    accepted = True
                    detail = vars(parsed)
                except SystemExit as exc:
                    accepted = exc.code == 0
                except ns["UsageError"]:
                    pass
                if accepted:
                    runner.append(dict(full=full, spelling=form, parsed=detail))
    # argparse also installs -h. All controls append aliases to the canonical
    # option list so later-value overrides are exercised exactly as REV4.
    runner.append(dict(full="--help", spelling=["-h"], parsed="builtin help"))
    timeout_specs = {"--preserve-status":None, "--foreground":None, "--kill-after":"10s",
                     "--signal":"TERM", "--verbose":None, "--help":None, "--version":None}
    timeout = []
    for full, value in timeout_specs.items():
        for n in range(3, len(full)+1):
            option = full[:n]
            forms = [[option,value],[option+"="+value]] if value is not None else [[option]]
            for form in forms:
                probe = subprocess.run(["timeout",*form,"1s","true"], capture_output=True, text=True, timeout=3)
                if probe.returncode == 0:
                    timeout.append(dict(full=full,spelling=form,probe_exit=0))
    for form in (["-k","10s"],["-k10s"],["-s","TERM"],["-sTERM"],["-v"],["-vk10s"],["-vsTERM"]):
        probe = subprocess.run(["timeout",*form,"1s","true"],capture_output=True,text=True,timeout=3)
        if probe.returncode == 0:
            timeout.append(dict(full="short/cluster",spelling=form,probe_exit=0))
    data = dict(runner=runner, timeout=timeout,
                runner_source_sha256=__import__("hashlib").sha256(Path(REAL).read_bytes()).hexdigest(),
                timeout_version=subprocess.check_output(["timeout","--version"],text=True).splitlines()[0],
                scope="all accepted option-name prefixes, separated/equals value forms, short attached/separate/cluster representatives; infinite values/repetitions are grammar classes")
    (TOOL/"OPTION_CENSUS.json").write_text(json.dumps(data,indent=2)+"\n")
    good=["--min","250","--max","300","--label","candidate_focus"]
    tail=["python3","run_gpu_tests.py","candidate","focus"]
    for i,row in enumerate(runner):
        refusal(f"Option_runner_{i:03d}",runner=good+row["spelling"],parser=row)
    for i,row in enumerate(timeout):
        refusal(f"Option_timeout_{i:03d}",inner=["timeout","--kill-after=10s",*row["spelling"],"240s",*tail],parser=row)
    # Additional equivalence classes: canonical options spelled with equals,
    # order, duplicate, number encoding; no parser-alias special casing.
    for name, opts in [
        ("equals_min",["--min=250","--max","300","--label","candidate_focus"]),
        ("equals_max",["--min","250","--max=300","--label","candidate_focus"]),
        ("equals_label",["--min","250","--max","300","--label=candidate_focus"]),
        ("reordered",["--max","300","--min","250","--label","candidate_focus"]),
        ("duplicate",good+good),
        ("float_min",["--min","250.0","--max","300","--label","candidate_focus"]),
        ("plus_max",["--min","250","--max","+300","--label","candidate_focus"]),
        ("zero_padded",["--min","0250","--max","300","--label","candidate_focus"])]:
        refusal("Grammar_runner_"+name,runner=opts)
    for i,duration in enumerate(["240","240.0s","4m","0.06666666666666667h","0.002777777777777778d","2.4e2s","0240s","+240s"]):
        refusal(f"Grammar_duration_{i}",inner=["timeout","--kill-after=10s",duration,*tail])


def rev4():
    good=["--min","250","--max","300","--label","candidate_focus"]
    tail=["python3","run_gpu_tests.py","candidate","focus"]
    refusal("REV4_max_equals",runner=good+["--max=900"])
    refusal("REV4_min_abbrev",runner=good+["--mi","1"])
    refusal("REV4_min_equals",runner=good+["--min=1"])
    refusal("REV4_inner_short_k",inner=["timeout","-k","240","900s",*tail])
    refusal("REV4_inner_kill_after_space",inner=["timeout","--kill-after","240","900s",*tail])
    for staging in ("unset_grace", "D3"):
        for name,mode in (("N2b","marker_only_sleep"),("N2c","ignore_term")):
            runs=start(name+"_"+staging)
            argv,e=canonical(runs)
            e["GN1_CPU_PAYLOAD"]=json.dumps([PY,str(TOOL/"rev4_forge.py")])
            e["REV2_MODE"]=mode
            e["REV3_SLEEP"]="60"
            if staging=="unset_grace":e.pop("FORGE_GPU_KILL_GRACE",None)
            argv[5]="2"
            t=time.monotonic();r=parent(argv,e);wall=time.monotonic()-t
            s=score(runs)
            att=runs/"candidate_focus/attempt-0001"
            pidfile=att/"orphan_pid.txt"
            state=None
            if pidfile.exists():
                pid=int(pidfile.read_text())
                try:state=Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[0]
                except FileNotFoundError:state="absent"
            alive=state not in (None,"absent","Z","X")
            record(name+"_"+staging,"D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child",
                   r["parent"] is not None and r["parent"]["timeout_cause"]=="outer-timeout"
                   and r["parent"]["exit"]==124 and not alive and not s["line"]["passed"],
                   parent=r,scorer=s,wall_seconds=wall,child_state=state,child_alive=alive)
    # Deliberately violate D3 grace ordering to exercise Python's raw -9
    # path with the real installed runner. The inner wrapper still bounds
    # the TERM-ignoring child; this is not accepted staging.
    runs=start("Raw_SIGKILL_normalisation")
    argv,e=canonical(runs,FORGE_GPU_KILL_GRACE="25",FORGE_GPU_POLL="5")
    e["GN1_CPU_PAYLOAD"]=json.dumps([PY,str(TOOL/"rev4_forge.py")])
    e["REV2_MODE"]="ignore_term";e["REV3_SLEEP"]="60";argv[5]="2"
    r=parent(argv,e);s=score(runs)
    pid=int((runs/"candidate_focus/attempt-0001/orphan_pid.txt").read_text())
    try:state=Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[0]
    except FileNotFoundError:state="absent"
    record("Raw_SIGKILL_normalisation","D3 raw negative return becomes137 before D4 outer-timeout attribution",
           r["parent"] is not None and r["parent"]["exit"]==137
           and r["parent"]["raw_exit"]==-9 and r["parent"]["timeout_cause"]=="outer-timeout" and state in ("absent","Z","X")
           and not s["line"]["passed"],parent=r,scorer=s,child_state=state)
    # N5 requires a synthetic row transport: the installed runner cannot be
    # instructed to emit no-lock/76 while still hanging after publication.
    runs=start("N5")
    argv,e=canonical(runs)
    body=f"#!{PY}\nimport os,sys\nos.execv({PY!r},[{PY!r},{str(TOOL/'rev4_forge.py')!r}]+sys.argv[1:])\n"
    argv[6]=executable(body,SCRATCH/"bin")
    argv[5]="2";e["REV2_MODE"]="nolock_row_hang";e["REV2_NO_LOG"]="0"
    r=parent(argv,e);s=score(runs)
    record("N5","D4 non-start row/124 returns outer-timeout, not-admitted receipt",
           r["parent"] is not None and not r["traceback"] and r["parent"]["admission"]=="not_admitted"
           and r["parent"]["timeout_cause"]=="outer-timeout" and bool(r["parent"]["reasons"])
           and not s["traceback"],parent=r,scorer=s)
    runs=start("N4")
    argv,e=canonical(runs);first=parent(argv,e)
    path=runs/"candidate_focus/attempt-0001/parent_result.json"
    text=path.read_text();path.rename(path.with_name("parent_result.full.json"));path.write_text(text[:len(text)//2])
    retry=parent(argv,e);s=score(runs)
    try:error=json.loads(retry["stderr"].strip())
    except ValueError:error={}
    record("N4","D5 corrupt prior receipt refuses with JSON error, no traceback",
           first["exit"]==0 and retry["exit"]==1 and bool(error.get("error"))
           and not retry["traceback"] and not s["traceback"] and not s["line"]["passed"],
           retry=retry,scorer=s,json_error=error)
    # Real publication uses link: complete retained temporary and final name
    # share an inode, and the final data parses after the child exited.
    runs=start("atomic_publication");argv,e=canonical(runs);r=parent(argv,e);s=score(runs)
    path=runs/"candidate_focus/attempt-0001/parent_result.json"
    temps=list(path.parent.glob("parent_result.*.tmp"))
    record("Atomic_publication","D5 complete linked receipt, no-overwrite publication",
           r["exit"]==0 and s["line"]["passed"] and len(temps)==1
           and temps[0].stat().st_ino==path.stat().st_ino and temps[0].read_bytes()==path.read_bytes(),
           parent=r,scorer=s,link_count=path.stat().st_nlink)


rev4()
census()
sys.exit(0 if all(r["passed"] for r in RESULT) else 1)
