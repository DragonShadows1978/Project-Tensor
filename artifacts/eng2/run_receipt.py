#!/usr/bin/env python3
"""ENG2 receipt wrapper (copied from artifacts/eng1/run_receipt.py; ENG2 changes: the bound is enforced by GNU
timeout around the whole process group, and every command stays under 10 minutes).
Usage: run_receipt.py NAME BOUND_SECONDS OUTDIR -- command...
Prior art: GNU timeout / Python subprocess; GN1/ENG1 receipt shape. No new algorithm.
"""
import datetime, json, os, subprocess, sys
from pathlib import Path


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


name, bound, outdir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
cmd = sys.argv[sys.argv.index("--") + 1:]
assert 0 < bound <= 590, "every command stays under 10 minutes"
outdir.mkdir(parents=True, exist_ok=True)
start = utc()
log_path = outdir / (name + ".log")
mode = "a" if log_path.exists() else "w"
with log_path.open(mode) as log:
    log.write("\n=== %s %s\n" % (start, " ".join(cmd)))
    log.flush()
    rc = subprocess.run(["timeout", "--kill-after=10s", "%ds" % bound, *cmd], stdout=log,
                        stderr=subprocess.STDOUT).returncode
end = utc()
rec = dict(name=name, command=cmd, bound_seconds=bound, start=start, end=end, exit=rc, timeout=rc in (124, 137),
           CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), TMPDIR=os.environ.get("TMPDIR"),
           CUDACXX=os.environ.get("CUDACXX"), cwd=os.getcwd().replace(str(Path.home()), "$HOME"))
receipts = outdir / (name + ".json")
history = json.loads(receipts.read_text()) if receipts.exists() else []
history.append(rec)
receipts.write_text(json.dumps(history, indent=2) + "\n")
print(json.dumps(rec))
sys.exit(rc)
