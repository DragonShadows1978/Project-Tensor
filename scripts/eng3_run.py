"""ENG3 helper: run one bounded command and append a JSON receipt (name, argv, bound, start/end UTC, exit, timeout,
the environment keys that matter, cwd). Used for the builds; the build itself runs in its own systemd scope.

Usage: python3 -B -P scripts/eng3_run.py <receipt.json> <name> <bound_seconds> -- <argv...>

Prior art: no prior art known to me beyond timeout(1) and ENG2's run receipts (artifacts/eng2/build/*.json in the eng2
worktree), whose field names this copies.
"""
import datetime
import json
import os
import subprocess
import sys


def main():
    receipt, name, bound = sys.argv[1], sys.argv[2], int(sys.argv[3])
    argv = sys.argv[sys.argv.index("--") + 1:]
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_path = receipt[:-5] + "_" + name + ".log" if receipt.endswith(".json") else receipt + ".log"
    with open(log_path, "ab") as log:
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=bound)
            to = False
        except subprocess.TimeoutExpired:
            p.terminate()  # our own child only
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            rc, to = 124, True
    end = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = []
    if os.path.exists(receipt):
        with open(receipt) as f:
            rows = json.load(f)
    rows.append({
        "name": name, "command": argv, "bound_seconds": bound, "start": start, "end": end, "exit": rc,
        "timeout": to, "log": log_path,
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TMPDIR", "CUDACXX")},
        "cwd": os.getcwd(),
    })
    with open(receipt, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{name}: exit {rc} timeout {to}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
