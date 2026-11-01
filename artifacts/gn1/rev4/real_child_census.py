#!/usr/bin/env python3
"""GN1-REV4 Q4: run A4's staged candidate_focus line exactly as written (same script
and argv; only the runner's summary, window and lock redirected to scratch, and
the guard prefix substituted) on CPU with CUDA hidden, through A4's real parent and
real child (pinned engine import), and read the runtime census: which recorded
modules are sealed, tampered, or unsealed. The line is expected to fail on CPU.
Prior art: Python sys.modules origin census (PSF docs); no other prior art known to me.
"""
import json, os, subprocess, sys, time
from pathlib import Path
OUT = Path(__file__).resolve().parent
MA4 = Path("/tmp/gn1_rev4_m/W/artifacts/gn1/a4")
FORGE = Path("/tmp/gn1_rev4_forge"); FORGE.mkdir(exist_ok=True)
if (FORGE / "summary.log").exists():
    os.replace(FORGE / "summary.log", FORGE / ("summary.moved." + str(time.time())))
runs = MA4 / ("rev4_real_" + time.strftime("%H%M%S"))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR="/tmp/gn1_rev4_tmp", CUDA_VISIBLE_DEVICES="",
           REV4_MA4=str(MA4), RUNS=str(runs), FORGE_GPU_SUMMARY=str(FORGE / "summary.log"),
           FORGE_GPU_WINDOW_FILE=str(FORGE / "GPU_WINDOW"), FORGE_GPU_LOCK=str(FORGE / "lock"))
for k in ("GN1_AUDIT", "FORGE_GPU_KILL_GRACE", "FORGE_GPU_POLL"):
    env.pop(k, None)
assert subprocess.run([sys.executable, "-B", "start_gpu_run.py"], cwd=MA4, env=env).returncode == 0
argv = [sys.executable, "-B", str(OUT / "parent_wrap4.py"), "candidate_focus", "0", "330", "forge-gpu-run", "--min", "250",
        "--max", "300", "--label", "gn1-a4-candidate_focus", "--", "timeout", "--kill-after=10s", "240s", "python3",
        "run_gpu_tests.py", "candidate", "focus"]
t0 = time.monotonic()
p = subprocess.run(argv, cwd=MA4, env=env, capture_output=True, text=True, timeout=360)
att = runs / "candidate_focus" / "attempt-0001"
rec = json.loads((att / "parent_result.json").read_text()) if (att / "parent_result.json").exists() else None
child = json.loads((att / "child_result.json").read_text()) if (att / "child_result.json").exists() else {}
res = dict(parent_exit=p.returncode, wall_s=round(time.monotonic() - t0, 1), stderr_tail=p.stderr[-800:],
           files=sorted(x.name for x in att.iterdir()) if att.exists() else [],
           child_pytest_exit=child.get("pytest_exit"), collected=len(child.get("collected_ids", [])),
           runtime_modules=len(child.get("runtime_modules", [])),
           passed=(rec or {}).get("passed"), reasons=[r[:200] for r in (rec or {}).get("reasons", [])][:6],
           tamper=[r for r in (rec or {}).get("reasons", []) if "tamper" in r or "unreadable" in r],
           unsealed=[u["path"] for u in (rec or {}).get("unsealed", [])],
           admission=(rec or {}).get("admission"), forge_outcome=(rec or {}).get("forge_outcome"))
home = str(Path.home())
(OUT / "real_child_census.json").write_text(json.dumps(res, indent=2).replace(home, "$HOME") + "\n")
print(json.dumps(res, indent=1)[:3000])
