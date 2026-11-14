#!/usr/bin/env python3
"""ENG1-REV2 supervisor: run one CPU command, record its exit only after the child has exited (L45).

Usage: python3 -B run.py LABEL cmd args...   (cwd = the worktree root)
Environment forced: CUDA_VISIBLE_DEVICES='' (L58), one thread (L25), TMPDIR=/tmp/eng1_rev2_work/tmp,
PYTHONDONTWRITEBYTECODE=1 (no __pycache__ writes into the worktree). Wrapped in nice -n 19 and timeout 550 s.
Prior art: GNU coreutils timeout/nice; the parent-after-child receipt pattern of GN1 A8 / ENG1 tooling.
"""
import datetime, json, os, subprocess, sys
from pathlib import Path

REV2 = Path(__file__).resolve().parent
label, cmd = sys.argv[1], sys.argv[2:]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TMPDIR="/tmp/eng1_rev2_work/tmp",
           PYTHONDONTWRITEBYTECODE="1")
env.pop("PYTHONPATH", None)
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
p = subprocess.run(["nice", "-n", "19", "timeout", "550s"] + cmd, env=env, capture_output=True, text=True)
end = datetime.datetime.now(datetime.timezone.utc).isoformat()
(REV2 / f"{label}.stdout").write_text(p.stdout)
(REV2 / f"{label}.stderr").write_text(p.stderr)
rec = dict(label=label, cmd=cmd, cwd=os.getcwd(), start=start, child_exited=end, exit=p.returncode,
           timeout=p.returncode == 124, CUDA_VISIBLE_DEVICES=env["CUDA_VISIBLE_DEVICES"])
(REV2 / f"{label}.exit.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
sys.exit(0)
