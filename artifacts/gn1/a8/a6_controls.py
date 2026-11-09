"""REV5/D0 controls through A6's real parent and scorer, CPU only.

Prior art: REV5 independently authored reproductions (2026), Meszaros fault
injection/test stubs (2007), POSIX execvp PATH lookup (year unverified),
Linux procfs exe/cmdline/environ (history unverified), CPython json recursion
limit, Python subprocess negative-signal convention and POSIX128+signal.
Unverified — lead to check: POSIX execvp PATH; Linux procfs history;
coreutils timeout history. No novel algorithm claimed.
"""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from evidence import TOOL, lines, read, sha
from program_binding import RUNNER, TIMEOUT, PYTHON, PATH, pins

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
BASE = TOOL / "a6_runs"
BASE.mkdir(exist_ok=False)
SCRATCH = Path(os.environ["TMPDIR"]) / "a6_controls"
SCRATCH.mkdir(exist_ok=False)
ROWS = []


def env(runs, control=True, **extra):
    e = dict(os.environ, RUNS=str(runs), PATH=PATH, GN1_CONTROL="1" if control else "0",
             GN1_CPU_FIXTURE="1", REV2_NO_LOG="1", REV2_MODE="normal",
             FORGE_GPU_SUMMARY=str(SCRATCH/"absent_summary"),
             FORGE_GPU_WINDOW_FILE=str(SCRATCH/"GPU_WINDOW"), FORGE_GPU_LOCK=str(SCRATCH/"lock"),
             FORGE_GPU_KILL_GRACE="10", FORGE_GPU_POLL="5")
    e.pop("GN1_AUDIT", None)
    e.update(extra)
    return e


def invoke(argv, e, seconds=90):
    with (BASE/"LAUNCHES.jsonl").open("a") as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               argv=argv, bound=seconds, environment={k:v for k,v in e.items()
                               if k == "PATH" or k.startswith(("FORGE_GPU_", "GN1_"))}))+"\n")
    r = subprocess.run([TIMEOUT, "--kill-after=2s", f"{seconds}s", *argv],
                       cwd=TOOL, env=e, capture_output=True, text=True)
    return dict(exit=r.returncode, stdout=r.stdout[-2000:], stderr=r.stderr[-3000:])


def start(name, control=True):
    runs = BASE/name
    r = invoke([PYTHON, "-B", "start_gpu_run.py"], env(runs, control))
    assert r["exit"] == 0, r
    return runs


def command(label):
    row = lines()[label]
    return [RUNNER, "--min", str(row["minimum"]), "--max", str(row["maximum"]),
            "--label", label, "--", TIMEOUT, "--kill-after=10s", f'{row["inner"]}s',
            PYTHON, "run_gpu_tests.py", row["arm"], row["suite"]]


def parent(runs, label="candidate_focus", cmd=None, launcher=None, control=True, **extra):
    row = lines()[label]
    args = [PYTHON, "-B", launcher or "run_gpu_line.py", label,
            ",".join(map(str, row["expected_child_exits"])), str(row["outer"]),
            *(cmd or command(label))]
    r = invoke(args, env(runs, control, **extra))
    paths = sorted((runs/label).glob("attempt-*/parent_result.json"))
    r["receipt"] = str(paths[-1].relative_to(TOOL)) if paths else None
    r["parent"] = read(paths[-1]) if paths else None
    return r


def score(runs, control=True, label="candidate_focus"):
    r = invoke([PYTHON, "-B", "score_gpu.py"], env(runs, control))
    r["summary"] = read(runs/"G5_SUMMARY.json")
    r["line"] = r["summary"]["lines"].get(label, {})
    return r


def record(name, target, ok, **data):
    ROWS.append(dict(control=name, target=target, passed=bool(ok), **data))
    (TOOL/"G4_a6.json").write_text(json.dumps(dict(passed=all(r["passed"] for r in ROWS), controls=ROWS),indent=2)+"\n")
    print(name, "PASS" if ok else "FAIL", flush=True)


def executable(name, body):
    p = SCRATCH/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"#!{PYTHON}\n"+body)
    p.chmod(0o755)
    return str(p)


# Every staged argv, unchanged, under the real runner and interpreter. The
# child explicitly dispatches a CPU fixture only for a control-mode START.
runs = start("honest_all_labels")
for label in lines():
    r = parent(runs, label)
    assert r["exit"] == 0, (label, r)
s = score(runs)
record("CPU_control_root", "D1 all 58 lines recompute; control root never establishes acceptance",
       s["exit"] == 1 and not s["summary"]["passed"]
       and "not a device run: every line requires positive child device evidence" in s["summary"]["errors"]
       and len(s["summary"]["lines"]) == 58 and all(v["passed"] for v in s["summary"]["lines"].values()), scorer=s)
(TOOL/"honest_labels_verified.json").write_text(json.dumps(dict(passed=all(v["passed"] for v in s["summary"]["lines"].values()),
    root=str(runs.relative_to(TOOL)), labels=58, runner=pins()["runner"],
    exact_staged_argv=True, control=True, gpu_acceptance=False, summary=str((runs/"G5_SUMMARY.json").relative_to(TOOL))),indent=2)+"\n")

# REV5 three deep-nesting payloads and raw-exit edits, in the full honest root.
for name, relative in [("NEW_parent_result_deep_nesting", "candidate_focus/attempt-0001/parent_result.json"),
                       ("NEW_start_json_deep_nesting", "START.json"),
                       ("NEW_regression_deep_child_result", "candidate_regression/attempt-0001/child_result.json")]:
    p = runs/relative
    old = p.read_bytes()
    try:
        p.write_text("["*200000+"]"*200000)
        s = score(runs, label="candidate_regression" if "regression" in name else "candidate_focus")
        record(name, "D2 Exception catches RecursionError; summary written and affected line/root refuses",
               s["exit"] == 1 and "Traceback" not in s["stderr"]
               and "recursion" in str(s["summary"]).lower(), scorer=s)
    finally:
        p.write_bytes(old)

p = runs/"candidate_focus/attempt-0001/parent_result.json"
old = p.read_bytes()
for name, raw in [("raw_exit_minus9_exit0", -9), ("raw_exit_removed", None),
                  ("raw_exit_bool", True), ("raw_exit_string", "0")]:
    try:
        d = json.loads(old)
        if raw is None:
            d.pop("raw_exit")
        else:
            d["raw_exit"] = raw
        p.write_text(json.dumps(d))
        s = score(runs)
        record(name, "D3 integer raw_exit and independently recomputed normalisation",
               s["exit"] == 1 and not s["line"]["passed"] and "raw_exit" in str(s["line"]), scorer=s)
    finally:
        p.write_bytes(old)

# Exercise D1 production checks against real parent operands. CPU payloads
# remain explicit fixture-runner programs, never a production GPU child.
# Here GN1_CPU_FIXTURE is cleared so a declared non-control child cannot be
# mistaken for permitted CPU dispatch in run_gpu_tests.
fixture = executable("fixture_runner", f"import os,sys\nos.execv({PYTHON!r},[{PYTHON!r},'-B',{str(TOOL/'rev2_forge.py')!r}]+sys.argv[1:])\n")
runner_wrapper = executable("wrapped_runner", f"import os,sys\nos.execv({RUNNER!r},[{RUNNER!r}]+sys.argv[1:sys.argv.index('--')]+['--max','900','--self-leasing']+sys.argv[sys.argv.index('--'):] )\n")
timeout_wrapper = executable("timeout_wrapper", f"import os,sys\na=sys.argv[1:];a[1]='900s'\nos.execv({TIMEOUT!r},[{TIMEOUT!r}]+a)\n")
python_wrapper = executable("python_wrapper", f"import os,sys\nos.execv({PYTHON!r},[{PYTHON!r},'-B',{str(TOOL/'rev2_forge.py')!r}]+sys.argv[1:])\n")
shadow_timeout = executable("shadow/timeout", f"import os,sys\na=sys.argv[1:];a[1]='900s'\nos.execv({TIMEOUT!r},[{TIMEOUT!r}]+a)\n")
shadow_python = executable("shadow/python3", f"import os,sys\na=sys.argv[1:]\nif a[:1]==['run_gpu_tests.py']:a=['-B',{str(TOOL/'rev2_forge.py')!r}]+a\nos.execv({PYTHON!r},[{PYTHON!r}]+a)\n")


def attack(name, mutate, extra=None):
    rroot = start(name, control=False)
    cmd = command("candidate_focus")
    # CPU payload is an interpreter wrapper. Other substitutions are applied
    # on top; every case still reaches the exact named D1 check.
    cmd[cmd.index("--")+4] = python_wrapper
    mutate(cmd)
    r = parent(rroot, cmd=cmd, control=False, GN1_CPU_FIXTURE="0", **(extra or {}))
    s = score(rroot, control=False)
    text = str(s["line"])
    record(name, "D1 pinned argv/program/environment refusal", r["parent"] is not None
           and s["exit"] == 1 and not s["line"].get("passed", True)
           and ("binding:" in text or "pin mismatch" in text), parent=r, scorer=s)
    return rroot, r, s


attack("Q2_wrapped_runner_command0", lambda c: c.__setitem__(0, runner_wrapper))
attack("D0c_runner_substitute", lambda c: c.__setitem__(0, fixture))
attack("D0c_timeout_wrapper", lambda c: c.__setitem__(c.index("--")+1, timeout_wrapper))
attack("D0c_interpreter_wrapper", lambda c: None)
attack("Q2_path_shadow_inner_timeout", lambda c: c.__setitem__(c.index("--")+1, "timeout"),
       dict(PATH=str(Path(shadow_timeout).parent)+":"+PATH))
attack("D0c_PATH_shadow", lambda c: None, dict(PATH=str(Path(shadow_python).parent)+":"+PATH))

# Isolate every pin and env equality using a real-parent honest fixture
# receipt, then alter exactly one recorded operand. This proves each deciding
# rule instead of relying on a different substitution to reject the case.
template = read(runs/"candidate_focus/attempt-0001/parent_result.json")
def production_operand(name, field, change):
    rroot = start(name, control=True)
    r = parent(rroot)
    att = rroot/"candidate_focus/attempt-0001/parent_result.json"
    d = read(att)
    start_path = rroot/"START.json"
    sr = read(start_path); sr["control"] = False; start_path.write_text(json.dumps(sr))
    d["control"] = False
    d["programs"] = dict(runner=pins()["runner"], timeout=pins()["timeout"],
        interpreter=pins()["interpreter"], outer_timeout=pins()["timeout"],
        runner_interpreter=pins()["interpreter"], shebang_env=pins()["shebang_env"],
        parent_interpreter=pins()["interpreter"], parent_kernel_exe=pins()["interpreter"]["resolved_path"])
    d["environment"] = dict(PATH=PATH, FORGE_GPU_KILL_GRACE="10", FORGE_GPU_POLL="5",
                            FORGE_GPU_RUN_LOG=str(att.parent/"forge.log"))
    change(d)
    att.write_text(json.dumps(d))
    s = score(rroot, control=False)
    record(name, field, s["exit"] == 1 and not s["line"]["passed"] and field in str(s["line"]), parent=r, scorer=s)

for key in ("runner", "timeout", "interpreter", "runner_interpreter", "parent_interpreter", "outer_timeout", "shebang_env"):
    production_operand("Recorded_pin_"+key, "program pin mismatch: "+key,
        lambda d, k=key: d["programs"][k].__setitem__("sha256", "0"*64))

for key in ("PATH", "FORGE_GPU_SUMMARY", "FORGE_GPU_WINDOW_FILE", "FORGE_GPU_LOCK",
            "FORGE_GPU_KILL_GRACE", "FORGE_GPU_POLL", "FORGE_GPU_RUN_LOG", "FORGE_GPU_UNKNOWN"):
    production_operand("Recorded_env_"+key, "child environment differs from staged environment",
        lambda d, k=key: d["environment"].__setitem__(k, "hostile"))

# Actual inherited environment overrides are recorded by the real parent.
# Synthetic runner avoids ever resolving a production ownership path.
for key in ("FORGE_GPU_SUMMARY", "FORGE_GPU_WINDOW_FILE", "FORGE_GPU_LOCK",
            "FORGE_GPU_KILL_GRACE", "FORGE_GPU_POLL", "FORGE_GPU_RUN_LOG"):
    rroot = start("Actual_env_"+key)
    value = str(SCRATCH/("override_"+key)) if key.endswith(("SUMMARY", "FILE", "LOCK", "LOG")) else "1"
    r = parent(rroot, cmd=[fixture,*command("candidate_focus")[1:]], **{key:value})
    s = score(rroot)
    actual = r["parent"]["environment"][key]
    expected = str(Path(r["parent"]["attempt_dir"])/"forge.log") if key == "FORGE_GPU_RUN_LOG" else value
    record("Actual_env_"+key, "D1 child-received environment recorded; control root rejected",
           actual == expected and s["exit"] == 1 and not s["summary"]["passed"], parent=r, scorer=s)

# D4 runs the real parent main, faulting after the child has returned. The
# injected launcher lives in task scratch; A6 source/seal stay byte-identical.
for name, fault in [("REV5_postexit_seal_failure", "seal"),
                    ("D4_write_retry", "write_once"), ("D4_double_write_incomplete", "write_twice")]:
    launch = SCRATCH/(name+".py")
    launch.write_text("import os,sys\nfrom pathlib import Path\n"+f"sys.path.insert(0,{str(TOOL)!r})\nimport run_gpu_line as P\n"+
        ("real=P.sha\ndef bad(p):\n if Path(p).name=='staging_seal.json':raise PermissionError('REV5 post-exit seal failure')\n return real(p)\nP.sha=bad\n" if fault == "seal" else
         "real=P.tempfile.mkstemp\ncalls=0\ndef bad(*a,**k):\n global calls\n calls+=1\n"+
         (" if calls<=1:raise OSError('D4 injected write failure')\n" if fault == "write_once" else " raise OSError('D4 injected persistent write failure')\n")+
         " return real(*a,**k)\nP.tempfile.mkstemp=bad\n")+"sys.exit(P.main())\n")
    rroot = start(name)
    r = parent(rroot, launcher=str(launch))
    s = score(rroot)
    if fault == "seal":
        ok = r["parent"] is not None and "REV5 post-exit seal failure" in str(r["parent"]["reasons"])
    elif fault == "write_once":
        ok = r["parent"] is not None and "parent receipt write failed" in str(r["parent"]["reasons"])
    else:
        ok = r["parent"] is None and "incomplete: missing parent completion receipt" in str(s["line"])
    record(name, "D4 protected seal/retry/minimal receipt; double failure incomplete fallback",
           ok and r["exit"] == 1 and s["exit"] == 1 and "Traceback" not in r["stderr"], parent=r, scorer=s)

# Restore the honest root's derived summary after all hostile file edits.
restored = score(BASE/"honest_all_labels")
assert all(v["passed"] for v in restored["summary"]["lines"].values())
assert not restored["summary"]["passed"]
sys.exit(0 if all(r["passed"] for r in ROWS) else 1)
