#!/usr/bin/env python3
"""CPU audit only (copied from ENG1's tool; adapted from GN1 A8; ENG2 change: ENG2_AUDIT). Prior art: subprocess/GNU
timeout; installed forge log format.
The stand-in never leases or calls the GPU. It writes the real nine-field outcome row after its child exits.
"""
import datetime
import os
from pathlib import Path
import subprocess
import sys
# ENG2-A1 change (the only one): this copy drives a NON-audit child on CPU for the mutant-scoring check, so it
# requires ENG2_AUDIT unset and the CPU test subset selected through PYTEST_ADDOPTS (pytest's own option variable).
assert os.environ.get("ENG2_AUDIT") is None and os.environ.get("PYTEST_ADDOPTS") == "-k cpu"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
args = sys.argv[1:]
command = args[args.index("--")+1:]
assert command[0] in ("timeout", "/usr/bin/timeout")
rc = subprocess.run(command).returncode
fields = [datetime.datetime.now(datetime.timezone.utc).isoformat(), os.getcwd(), "audit", "halted", "-", "0", "0", "ran", str(rc)]
with Path(os.environ["FORGE_GPU_RUN_LOG"]).open("a") as f:
    f.write("\t".join(fields)+"\n")
sys.exit(rc)
