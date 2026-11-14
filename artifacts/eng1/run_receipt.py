#!/usr/bin/env python3
"""ENG1 receipt wrapper: run one command under a bound, write a JSON receipt after the child exits.
Usage: run_receipt.py NAME BOUND_SECONDS OUTDIR -- command...
Prior art: GNU timeout / Python subprocess; the receipt shape follows GN1's configure/build receipts
(artifacts/gn1/configure_cuda126.json). No new algorithm.
"""
import datetime, json, os, subprocess, sys
from pathlib import Path

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

name, bound, outdir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
cmd = sys.argv[sys.argv.index("--") + 1:]
assert 0 < bound <= 1190, "every command stays under 20 minutes"
outdir.mkdir(parents=True, exist_ok=True)
start = utc()
timed_out = False
with (outdir / (name + ".log")).open("w") as log:
    try:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=bound).returncode
    except subprocess.TimeoutExpired:
        rc, timed_out = 124, True
end = utc()
rec = dict(name=name, command=cmd, bound_seconds=bound, start=start, end=end, exit=rc, timeout=timed_out,
           CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), TMPDIR=os.environ.get("TMPDIR"),
           CUDACXX=os.environ.get("CUDACXX"), cwd=os.getcwd().replace(str(Path.home()), "$HOME"))
(outdir / (name + ".json")).write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec))
sys.exit(rc)
