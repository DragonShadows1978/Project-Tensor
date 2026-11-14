#!/usr/bin/env python3
"""CPU audit only. Prior art: subprocess/GNU timeout; installed forge log format.
Stand-in never leases or calls GPU. Uses the real nine-field outcome format.
"""
import datetime
import os
from pathlib import Path
import subprocess
import sys
assert os.environ.get("GN1_AUDIT") == "1"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
args = sys.argv[1:]
command = args[args.index("--")+1:]
assert command[0] in ("timeout", "/usr/bin/timeout")
rc = subprocess.run(command).returncode
fields = [datetime.datetime.now(datetime.timezone.utc).isoformat(), os.getcwd(), "audit", "halted", "-", "0", "0", "ran", str(rc)]
with Path(os.environ["FORGE_GPU_RUN_LOG"]).open("a") as f:
    f.write("\t".join(fields)+"\n")
sys.exit(rc)
