#!/usr/bin/env python3
"""GN1-REV6 own cases against A8's REAL parent (run_gpu_line.py via parent_wrap6.py),
REAL child (run_gpu_tests.py) and REAL scorer (score_gpu.py / evidence.py /
device_evidence.py), byte-identical in the seal-verified /tmp/gn1_rev6_m mirror.
The installed runner always has summary, window and lock in /tmp/gn1_rev6_forge.
CUDA_VISIBLE_DEVICES is empty except where a Q2 case sets it ("-1") as an input.

Parts (argv selects; default all):
  d  Q2 device evidence: live GPU-hidden production lines (CVD "" and "-1"),
     staged-script CVD guard (unset, empty), evidence ownership/timing, data-only
     classify/check_line_device units seeded from the REAL hidden capture,
     a right-path wrong-bytes child.
  m  Q3 modes: START/parent/env mode mismatches, audit root, fixture in a
     production root, and a root-level demote-only unit of score_gpu.main.
  i  Q4 live analogue of the Import_helper_pin seal edits (import module bytes).
  f  Q1 new cases: F01 runner copy at another path and wrapped runner with
     production inner argv; F02 PATH-shadowed interpreter for the runner's
     /usr/bin/env shebang; F03 deep child_result (focus, and regression under a
     control root so regression_reasons is reached); F05 double post-exit failure.
Every parent/scorer exit is recorded after the process exits (L45).

Prior art: fault injection / test stubs (Meszaros 2007); REV5 hostile_rev5.py
(2026) helpers reused via hostile_rev6; CUDA_VISIBLE_DEVICES semantics (NVIDIA
CUDA C Programming Guide, environment variables; unverified - lead to check:
"CUDA_VISIBLE_DEVICES invalid index hides devices"); POSIX execvp PATH lookup and
env(1) shebang resolution for the interpreter shadow. Ours: the case list.
"""
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostile_rev6 as H  # noqa: E402

OUT = Path(__file__).resolve().parent
RESULTS = []
_ORIG_BASE_ENV = H.base_env


def _base_env(**extra):
    # Q2 input only: REV6_ALLOW_CVD sets CUDA_VISIBLE_DEVICES after REV5's
    # empty-CVD assertion (which stays in force for every other launch).
    allow = extra.pop("REV6_ALLOW_CVD", None)
    if allow is None:
        return _ORIG_BASE_ENV(**extra)
    e = _ORIG_BASE_ENV(**{k: v for k, v in extra.items() if k != "CUDA_VISIBLE_DEVICES"})
    assert allow in ("", "-1"), allow
    e["CUDA_VISIBLE_DEVICES"] = allow
    return e


H.base_env = _base_env
REAL_PATH = "/usr/bin:/bin"
LABEL = "candidate_focus"
PROD_INNER = ["/usr/bin/timeout", "--kill-after=10s", "240s", "/usr/bin/python3", "run_gpu_tests.py", "candidate", "focus"]


def record(case, question, expected, ok, **data):
    RESULTS.append(dict(case=case, question=question, expectation=expected, outcome_as_expected=bool(ok), **data))
    print(case, "AS-EXPECTED" if ok else "UNEXPECTED", flush=True)


def reasons(sc):
    return (sc.get("line") or {}).get("reasons") or []


def has(sc, text):
    return any(text in r for r in reasons(sc))


def child_files(runs, label=LABEL):
    att = runs / label / "attempt-0001"
    out = {}
    for name in ("child_result.json", "invocation.json", "parent_result.json", "junit.xml"):
        p = att / name
        out[name] = p.stat().st_mtime_ns if p.exists() else None
    return att, out


# ---------------------------------------------------------------- part d (Q2)
def live_hidden(name, cvd):
    runs = H.start(name, PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH, CUDA_VISIBLE_DEVICES=cvd,
                 REV6_ALLOW_CVD=cvd)
    sc = H.score(runs, LABEL)
    att, mt = child_files(runs)
    child = json.loads((att / "child_result.json").read_text()) if mt["child_result.json"] else {}
    inv = json.loads((att / "invocation.json").read_text()) if mt["invocation.json"] else {}
    parent = json.loads((att / "parent_result.json").read_text()) if mt["parent_result.json"] else {}
    ev = child.get("device_evidence") or {}
    own = dict(
        invocation_equals_child=inv.get("device_evidence") == ev and bool(ev),
        parent_has_no_device_evidence="device_evidence" not in json.dumps(parent),
        invocation_written_before_junit=(mt["invocation.json"] or 0) <= (mt["junit.xml"] or 0) and mt["junit.xml"] is not None,
        captured_within_child=bool(parent) and parent["start"] <= ev.get("captured_at", "") <= parent["child_exited"],
        program_is_real_child=ev.get("program", {}).get("resolved_path") == str(H.MA8 / "run_gpu_tests.py"),
    )
    ok = (sc["exit"] == 1 and sc["errors"] == ["not a device run: every line requires positive child device evidence"]
          and (sc.get("line") or {}).get("mode") == "cpu" and has(sc, "mode binding: parent declaration disagrees")
          and ev.get("device_name") is None and ev.get("CUDA_VISIBLE_DEVICES") == cvd
          and ev.get("native_status", {}).get("cudaGetDevice") not in (None, "0") and all(own.values()))
    record("Q2_live_gpu_hidden_cvd_" + (cvd or "empty"), "Q2",
           "production line on CPU: child records no device; line mode cpu + mode binding; root refused for absent device evidence; evidence child-owned, pre-test",
           ok, parent_exit=r["exit"], parent_reasons=(r["parent"] or {}).get("reasons"), scorer=sc,
           evidence=ev, ownership=own, mtimes=mt)
    return child


def staged_guard(name, cvd):
    runs = H.MA8 / ("rev6_staged_" + name)
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES",)}
    env.update(RUNS=str(runs), TMPDIR="/tmp/gn1_a8_rev6_never_created", PATH=REAL_PATH)
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    p = subprocess.run(["/usr/bin/bash", str(H.MA8 / "STAGED_GPU_COMMANDS.sh"), LABEL], cwd=H.MA8, env=env,
                       capture_output=True, text=True, timeout=60)
    created = runs.exists() or Path("/tmp/gn1_a8_rev6_never_created").exists()
    record("Q2_staged_script_cvd_" + name, "Q2",
           "staged script refuses before START when CUDA_VISIBLE_DEVICES is unset or empty; nothing created",
           p.returncode != 0 and not created and "nonempty CUDA_VISIBLE_DEVICES" in p.stderr,
           exit=p.returncode, stderr=p.stderr[-600:], created=created)


def unit_cases(real_child):
    sys.path.insert(0, str(H.MA8))
    os.chdir(H.MA8)
    import evidence  # mirror copy, seal verified below
    seal = evidence.verify_seal()
    base = copy.deepcopy(real_child)
    pos = copy.deepcopy(base)
    pos["CUDA_VISIBLE_DEVICES"] = "0"
    pos["device_evidence"].update(CUDA_VISIBLE_DEVICES="0", device_name="Unit Device", device_ordinal="0",
                                  native_status={k: "0" for k in ("cudaRuntimeGetVersion", "cudaDriverGetVersion",
                                                                   "cudaGetDevice", "cudaGetDeviceProperties_v2")},
                                  query_error=None)
    prod = dict(control=False, audit=False)

    def ev(**kw):
        d = copy.deepcopy(pos)
        for k, v in kw.items():
            if v is KeyError:
                d["device_evidence"].pop(k, None)
            else:
                d["device_evidence"][k] = v
        return d
    cases = [
        ("accept_path_positive_production", pos, prod, True),
        ("real_hidden_capture_production", base, prod, False),
        ("positive_parent_control_true", pos, dict(control=True, audit=False), False),
        ("positive_parent_audit_true", pos, dict(control=False, audit=True), False),
        ("positive_parent_control_string", pos, dict(control="false", audit=False), False),
        ("positive_parent_control_missing", pos, dict(audit=False), False),
        ("positive_parent_audit_int0", pos, dict(control=False, audit=0), False),
        ("cvd_unset", dict(ev(CUDA_VISIBLE_DEVICES=None), CUDA_VISIBLE_DEVICES=None), prod, False),
        ("cvd_empty", dict(ev(CUDA_VISIBLE_DEVICES=""), CUDA_VISIBLE_DEVICES=""), prod, False),
        ("cvd_evidence_vs_child_mismatch", dict(ev(CUDA_VISIBLE_DEVICES="0"), CUDA_VISIBLE_DEVICES="1"), prod, False),
        ("device_name_empty", ev(device_name=""), prod, False),
        ("device_name_whitespace", ev(device_name="   "), prod, False),
        ("device_name_none_statuses_zero", ev(device_name=None), prod, False),
        ("device_name_not_str", ev(device_name=7), prod, False),
        ("getdevice_status_100", ev(native_status=dict(pos["device_evidence"]["native_status"], cudaGetDevice="100")), prod, False),
        ("properties_status_1", ev(native_status=dict(pos["device_evidence"]["native_status"], cudaGetDeviceProperties_v2="1")), prod, False),
        ("native_status_missing_call", ev(native_status={k: "0" for k in ("cudaRuntimeGetVersion", "cudaDriverGetVersion", "cudaGetDevice")}), prod, False),
        ("native_status_int_zero", ev(native_status={k: 0 for k in ("cudaRuntimeGetVersion", "cudaDriverGetVersion", "cudaGetDevice", "cudaGetDeviceProperties_v2")}), prod, False),
        ("query_error_set", ev(query_error="x"), prod, False),
        ("runtime_version_zero", ev(cuda_runtime_version="0"), prod, False),
        ("driver_version_nondecimal", ev(cuda_driver_version="13.0"), prod, False),
        ("runtime_version_int", ev(cuda_runtime_version=12060), prod, False),
        ("ordinal_negative", ev(device_ordinal="-1"), prod, False),
        ("ordinal_missing", ev(device_ordinal=KeyError), prod, False),
        ("ordinal_unicode_digit", ev(device_ordinal="٠"), prod, False),
        ("engine_sha_wrong", ev(engine_sha256="0" * 64), prod, False),
        ("provider_other", ev(provider="other"), prod, False),
        ("captured_at_bad", ev(captured_at="yesterday"), prod, False),
        ("program_right_path_wrong_bytes", ev(program=dict(pos["device_evidence"]["program"], sha256="1" * 64)), prod, False),
        ("program_standin_path", ev(program=dict(resolved_path=str(H.MA8 / "rev4_forge.py"), sha256=pos["device_evidence"]["program"]["sha256"])), prod, False),
        ("program_missing", ev(program=KeyError), prod, False),
        ("evidence_list", dict(pos, device_evidence=[]), prod, False),
        ("evidence_absent", {k: v for k, v in pos.items() if k != "device_evidence"}, prod, False),
        ("evidence_absent_control_parent", {k: v for k, v in pos.items() if k != "device_evidence"}, dict(control=True, audit=False), False),
    ]
    rows = []
    for name, data, parent, accept in cases:
        try:
            mode, errors = evidence.check_line_device(data, seal, parent)
            accepted = mode == "device" and not errors
            err = None
        except Exception as exc:  # a raise refuses the line in recompute_line/score_gpu
            mode, errors, accepted, err = None, [], False, repr(exc)
        rows.append(dict(case=name, accept_expected=accept, mode=mode, errors=errors, raised=err, accepted=accepted,
                         as_expected=accepted is accept))
    record("Q2_Q3_unit_check_line_device", "Q2/Q3",
           "only a complete positive record under a production declaration is device-mode with no errors",
           all(r["as_expected"] for r in rows), units=rows, n=len(rows))


def wrong_bytes_child():
    child = H.MA8 / "run_gpu_tests.py"
    original = child.read_bytes()
    runs = H.start("Q2_wrong_bytes_child", PATH=REAL_PATH)
    try:
        child.write_bytes(original + b"\n# rev6 wrong-bytes probe\n")
        r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH)
        sc = H.score(runs, LABEL)
    finally:
        child.write_bytes(original)
    restored = subprocess.run([H.PY, "-B", "-c", "import evidence; evidence.verify_seal(); print('ok')"], cwd=H.MA8,
                              capture_output=True, text=True, timeout=60)
    ok = (r["exit"] == 1 and "tool seal mismatch: run_gpu_tests.py" in r["stderr"] and r["attempts"] == []
          and sc["exit"] == 1 and any("tool seal mismatch: run_gpu_tests.py" in e for e in (sc["errors"] or [])))
    record("Q2_child_right_path_wrong_bytes", "Q2",
           "parent refuses before launch and scorer refuses the root: tool seal mismatch on run_gpu_tests.py",
           ok and restored.stdout.strip() == "ok", parent=r, scorer=sc, seal_restored=restored.stdout.strip())


def part_d():
    real = live_hidden("Q2_hidden_empty", "")
    live_hidden("Q2_hidden_minus1", "-1")
    staged_guard("unset", None)
    staged_guard("empty", "")
    unit_cases(real)
    wrong_bytes_child()


# ---------------------------------------------------------------- part m (Q3)
def root_unit():
    sys.path.insert(0, str(H.MA8))
    os.chdir(H.MA8)
    import evidence
    import score_gpu
    rows = []
    labels = list(evidence.lines())
    for name, start, line_mode, want in (
            ("all_device_production", dict(control=False, audit=False), "device", True),
            ("all_device_control_root", dict(control=True, audit=False), "device", False),
            ("all_device_audit_root", dict(control=False, audit=True), "device", False),
            ("one_cpu_line_production", dict(control=False, audit=False), "one_cpu", False),
            ("one_line_no_mode_production", dict(control=False, audit=False), "one_missing", False)):
        runs = H.BASE / ("M_root_unit_" + name)
        runs.mkdir(parents=True, exist_ok=True)

        def fake_line(r, label, seal, st, stack=(), _m=line_mode):
            row = dict(passed=True, reasons=[], actual_killers=[], outcomes={}, mode="device")
            if _m == "one_cpu" and label == labels[0]:
                row["mode"] = "cpu"
            if _m == "one_missing" and label == labels[0]:
                row.pop("mode")
            return row
        score_gpu.recompute_line = fake_line
        score_gpu.verify_start = lambda r, _s=start: ({}, _s)
        os.environ["RUNS"] = str(runs)
        rc = score_gpu.main()
        summ = json.loads((runs / "G5_SUMMARY.json").read_text())
        rows.append(dict(case=name, start=start, line_mode=line_mode, exit=rc, passed=summ["passed"], errors=summ["errors"],
                         as_expected=summ["passed"] is want))
    record("Q3_root_demote_only_unit", "Q3",
           "score_gpu.main with every line device-mode and passing: production accepts; control/audit roots, a cpu line or a line without mode refuse",
           all(r["as_expected"] for r in rows), units=rows)


def part_m():
    runs = H.start("Q3_start_prod_parent_control", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH, GN1_CONTROL="1")
    record("Q3_parent_control_env_vs_production_START", "Q3", "parent refuses pre-launch: START control-mode mismatch",
           r["exit"] == 1 and "START control-mode mismatch" in r["stderr"] and r["attempts"] == [], parent=r)
    runs = H.start("Q3_start_control_parent_prod", PATH=REAL_PATH, GN1_CONTROL="1")
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH, GN1_CONTROL=None)
    record("Q3_parent_production_env_vs_control_START", "Q3", "parent refuses pre-launch: START control-mode mismatch",
           r["exit"] == 1 and "START control-mode mismatch" in r["stderr"] and r["attempts"] == [], parent=r)
    runs = H.start("Q3_audit_root", PATH=REAL_PATH, GN1_AUDIT="1")
    sc = H.score(runs)
    record("Q3_audit_root_scored", "Q3", "audit root refused at root level",
           sc["exit"] == 1 and "audit run cannot establish GPU acceptance" in (sc["errors"] or []), scorer=sc)
    runs = H.start("Q3_fixture_in_production_root", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH, GN1_CPU_FIXTURE="1",
                 REV2_NO_LOG="1")
    sc = H.score(runs, LABEL)
    record("Q3_cpu_fixture_requested_in_production_root", "Q3",
           "child refuses the fixture branch (START not control): no child_result, line and root refused",
           r["exit"] == 1 and "child_result.json" not in r["last_attempt_files"] and (sc.get("line") or {}).get("passed") is False
           and sc["exit"] == 1, parent=r, scorer=sc)
    runs = H.start("Q3_control_root_real_child_fixture", PATH=REAL_PATH, GN1_CONTROL="1")
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH, GN1_CONTROL="1",
                 GN1_CPU_FIXTURE="1", REV2_NO_LOG="1")
    sc = H.score(runs, LABEL)
    record("Q3_control_root_line_passes_root_refused", "Q3",
           "declared CPU control line passes its line checks (mode cpu); root refused for absent device evidence",
           (sc.get("line") or {}).get("passed") is True and (sc.get("line") or {}).get("mode") == "cpu"
           and "not a device run: every line requires positive child device evidence" in (sc["errors"] or []),
           parent=r, scorer=sc)
    root_unit()


# ---------------------------------------------------------------- part f (Q1)
def part_f():
    copy_runner = H.TOOLS / "forge-gpu-run-bytecopy"
    shutil.copyfile(Path(H.REAL_FORGE).resolve(), copy_runner)
    copy_runner.chmod(0o755)
    runs = H.start("F01_runner_bytecopy_other_path", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [str(copy_runner), *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH)
    sc = H.score(runs, LABEL)
    record("F01_new_runner_bytecopy_other_path", "Q1 R5-F01", "refused: runner command[0] differs from pinned runner",
           has(sc, "program/environment binding: runner command[0] differs from pinned runner") and sc["exit"] == 1,
           parent_exit=r["exit"], scorer=sc)
    runs = H.start("F01_wrapped_runner_prod_inner", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [str(H.TOOLS / "wrapped-runner"), *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH)
    sc = H.score(runs, LABEL)
    record("F01_new_wrapped_runner_production_inner", "Q1 R5-F01",
           "refused by runner identity even with production inner argv",
           has(sc, "runner command[0] differs from pinned runner") and not has(sc, "inner argv differs") and sc["exit"] == 1,
           parent_exit=r["exit"], scorer=sc)
    shadow = H.TOOLS / "pyshadow"
    shadow.mkdir(exist_ok=True)
    H.tool("python3", "#!/usr/bin/python3\nimport os, sys\nos.execv('/usr/bin/python3', ['/usr/bin/python3'] + sys.argv[1:])\n", shadow)
    runs = H.start("F02_runner_interpreter_shadow", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=str(shadow) + ":" + REAL_PATH)
    sc = H.score(runs, LABEL)
    record("F02_new_PATH_shadow_runner_shebang_interpreter", "Q1 R5-F02",
           "refused: the runner's /usr/bin/env python3 resolves a shadow; pin mismatch runner_interpreter",
           has(sc, "program pin mismatch: runner_interpreter") and sc["exit"] == 1, parent_exit=r["exit"],
           programs=None, scorer=sc)
    runs = H.start("F02_unknown_forge_var", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH, FORGE_GPU_REV6_UNKNOWN="1",
                 FORGE_GPU_KILL_GRACE="18")
    sc = H.score(runs, LABEL)
    att = runs / LABEL / "attempt-0001" / "parent_result.json"
    envrec = json.loads(att.read_text())["environment"] if att.exists() else None
    record("F02_new_live_unknown_var_and_grace18", "Q1 R5-F02",
           "child environment recorded with the live values and refused: child environment differs from staged environment",
           has(sc, "child environment differs from staged environment") and envrec is not None
           and envrec.get("FORGE_GPU_REV6_UNKNOWN") == "1" and envrec.get("FORGE_GPU_KILL_GRACE") == "18",
           parent_exit=r["exit"], recorded_environment=envrec, scorer=sc)
    # F03: deep child_result.json on a focus line (evaluate + recompute_line read path).
    runs = H.start("F03_focus_deep_child", PATH=REAL_PATH)
    r = H.parent(runs, LABEL, [str(H.TOOLS / "fixture-runner"), *H.CANON, "--", *PROD_INNER], REV5_FIX="deep_child")
    sc = H.score(runs, LABEL)
    record("F03_new_focus_deep_child_result", "Q1 R5-F03",
           "parent receipt with a recorded reason; scorer writes a summary, no traceback, line refused",
           r["parent"] is not None and not r["traceback"] and not sc["traceback"] and sc["G5_summary_written"]
           and (sc.get("line") or {}).get("passed") is False, run=r, scorer=sc)
    # F03: REV5's regression case under a declared control root so regression_reasons is reached.
    ctrl = dict(GN1_CONTROL="1")
    runs = H.start("F03_regression_deep_child_control", **ctrl)
    pre = []
    for l in ["baseline_focus", "candidate_focus", "baseline_regression"]:
        ln = H.LINES[l]
        inner = ["timeout", "--kill-after=10s", "%ds" % ln["inner"], "python3", "run_gpu_tests.py", ln["arm"], ln["suite"]]
        pr = H.parent(runs, l, [*H.SR(), "--min", str(ln["minimum"]), "--max", str(ln["maximum"]), "--label", l, "--", *inner], **ctrl)
        pre.append(dict(label=l, exit=pr["exit"], reasons=(pr["parent"] or {}).get("reasons")))
    ln = H.LINES["candidate_regression"]
    inner = ["timeout", "--kill-after=10s", "%ds" % ln["inner"], "python3", "run_gpu_tests.py", ln["arm"], ln["suite"]]
    r = H.parent(runs, "candidate_regression", [str(H.TOOLS / "fixture-runner"), "--min", str(ln["minimum"]), "--max",
                                                str(ln["maximum"]), "--label", "candidate_regression", "--", *inner],
                 REV5_FIX="deep_child", **ctrl)
    sc = H.score(runs, "candidate_regression")
    reached = any("recursion" in x for x in ((r["parent"] or {}).get("reasons") or []))
    record("F03_rerun_regression_deep_child_under_control", "Q1 R5-F03",
           "prerequisite lines pass; parent reaches regression_reasons and records the RecursionError; scorer summary, no traceback",
           all(p["exit"] == 0 for p in pre) and r["parent"] is not None and not r["traceback"] and reached
           and not sc["traceback"] and sc["G5_summary_written"] and (sc.get("line") or {}).get("passed") is False,
           prerequisites=pre, run=r, scorer=sc)
    # F05: seal unreadable AND attempt directory read-only while the child runs.
    runs = H.start("F05_double_failure")
    seal = H.MA8 / "staging_seal.json"
    smode = os.stat(seal).st_mode & 0o777
    p = H.parent(runs, LABEL, [*H.SR(), *H.CANON, "--", *H.INNER], popen=True, REV2_MODE="hang_after_result", REV3_SLEEP="5")
    att = runs / LABEL / "attempt-0001"
    for _ in range(100):
        if (att / "child_result.json").exists():
            break
        time.sleep(0.1)
    os.chmod(seal, 0)
    amode = os.stat(att).st_mode & 0o777
    os.chmod(att, 0o555)
    try:
        out, err = p.communicate(timeout=120)
    finally:
        os.chmod(seal, smode)
        os.chmod(att, amode)
    r = H.collect(runs, LABEL, dict(exit=p.returncode, stdout=out[-1200:], stderr=err[-2500:]))
    sc = H.score(runs, LABEL)
    record("F05_new_seal_unreadable_and_attempt_readonly", "Q1 R5-F05",
           "no receipt can be published: parent exits 1 with a JSON error, no traceback; scorer marks the line incomplete",
           r["exit"] == 1 and not r["traceback"] and isinstance(r["stderr_json_line"], dict)
           and "parent_result.json" not in r["last_attempt_files"] and "incomplete" in json.dumps(sc.get("line"))
           and (sc.get("line") or {}).get("passed") is False, run=r, scorer=sc,
           seal_mode_restored=oct(os.stat(seal).st_mode & 0o777))


# ---------------------------------------------------------------- part i (Q4)
def part_i():
    """Live analogue of the 3 class (iii) Import_helper_pin targets: change the
    import module's BYTES (a wrong program, in scope) instead of the seal entry."""
    mod = H.MROOT / "scripts/apa_om135_core.py"
    original = mod.read_bytes()
    runs = H.start("Q4_import_module_bytes_changed", PATH=REAL_PATH)
    try:
        mod.write_bytes(original + b"\n# rev6 import-module bytes probe\n")
        st = H.run([H.PY, "-B", "start_gpu_run.py"], H.base_env(RUNS=str(H.BASE / "Q4_import_module_start_after_change")))
        r = H.parent(runs, LABEL, [H.REAL_FORGE, *H.CANON, "--", *PROD_INNER], PATH=REAL_PATH)
        sc = H.score(runs, LABEL)
    finally:
        mod.write_bytes(original)
    restored = subprocess.run([H.PY, "-B", "-c", "import evidence; evidence.verify_seal(); print('ok')"], cwd=H.MA8,
                              capture_output=True, text=True, timeout=60)
    msg = "import module seal mismatch: scripts/apa_om135_core.py"
    record("Q4_live_import_module_bytes_changed", "Q4",
           "START, parent (pre-launch) and scorer (root) all refuse with the import-module seal mismatch",
           st["exit"] != 0 and msg in st["stderr"] and r["exit"] == 1 and msg in r["stderr"] and r["attempts"] == []
           and sc["exit"] == 1 and msg in (sc["errors"] or []) and restored.stdout.strip() == "ok",
           start=st, parent=r, scorer=sc, seal_restored=restored.stdout.strip())


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parts = sys.argv[1:] or ["d", "m", "f", "i"]
    H.BASE.mkdir(parents=True)
    for d in (H.TOOLS, H.FORGE, H.OBS):
        d.mkdir(exist_ok=True)
    H.make_tools()
    for part in parts:
        try:
            globals()["part_" + part]()
        except Exception:
            import traceback
            record("PART_%s_harness_error" % part, "-", "harness completes", False, error=traceback.format_exc()[-3000:])
    redact = lambda s: s.replace(H.REAL_FORGE, "forge-gpu-run").replace(str(Path.home()), "$HOME")
    out = OUT / ("rev6_cases_" + "".join(parts) + ".json")
    out.write_text(redact(json.dumps(dict(base=str(H.BASE), results=RESULTS), indent=1, default=str)) + "\n")
    print("unexpected:", [r["case"] for r in RESULTS if not r["outcome_as_expected"]])


if __name__ == "__main__":
    main()
