#!/usr/bin/env python3
"""CPU audit stand-in; no GPU leasing, GPU API or real forge invocation.
Prior art: Python subprocess.run (PSF), argument-preserving synchronous wrapper.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get("GN1_AUDIT") == "1"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
args = sys.argv[1:]
assert "--max" in args
split = args.index("--")
command = args[split + 1:]
assert command[0] == "timeout", command
print(json.dumps(dict(standin=True, cwd=os.getcwd(), argv=args)), flush=True)
result = subprocess.run(command)
print(json.dumps(dict(standin_child_exit=result.returncode)), flush=True)
sys.exit(result.returncode)
