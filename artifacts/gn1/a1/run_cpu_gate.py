"""Synchronous bounded CPU gate; parent writes status after child returns.
Prior art: GNU coreutils timeout and Python subprocess.run (PSF, 2003+);
L45 parent receipts from this order. Wiring only; no novel algorithm.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
name, seconds, *command = sys.argv[1:]
assert name and command and 0 < int(seconds) < 1200
out = root / "artifacts/gn1/a1"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/gn1_a1_cpu",
           PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
Path(env["TMPDIR"]).mkdir(exist_ok=True)
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (out / (name + ".log")).open("x") as log:
    child = subprocess.run(["timeout", "--kill-after=10s", seconds + "s", *command],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
receipt = dict(name=name, command=command, bound_seconds=int(seconds),
               start=start, end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               exit=child.returncode, timeout=child.returncode in (124,137),
               CUDA_VISIBLE_DEVICES="", TMPDIR=env["TMPDIR"])
(out / (name + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
sys.exit(child.returncode)
