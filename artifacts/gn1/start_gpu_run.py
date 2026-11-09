"""Lead start/seal record. Prior art: SHA-256 (NIST 2001), content pinning;
L59 start record and L67 free-space rails from the GN1 order. GN1 wiring only.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[1]
runs = Path(os.environ["RUNS"]).resolve()
assert str(runs) != "/" and runs != ROOT and runs != TOOL
runs.mkdir(parents=True, exist_ok=True)
seal = json.loads((TOOL / "staging_seal.json").read_text())
for file, expected in seal["tool_files"].items():
    assert hashlib.sha256((TOOL / file).read_bytes()).hexdigest() == expected, "changed tool " + file
order = ROOT / "orders/GN1_GROUPNORM_FIX.md"
assert hashlib.sha256(order.read_bytes()).hexdigest() == seal["order_sha256"], "changed order"
record = dict(date=subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
              order_log=subprocess.check_output(["git", "log", "-1", "--format=%H %s", "--", str(order)], cwd=ROOT, text=True).strip(),
              order_sha256=seal["order_sha256"], audit=os.environ.get("GN1_AUDIT") == "1",
              seal_sha256=hashlib.sha256((TOOL / "staging_seal.json").read_bytes()).hexdigest())
path = runs / "START.json"
if path.exists():
    old = json.loads(path.read_text())
    assert old["seal_sha256"] == record["seal_sha256"] and old["audit"] == record["audit"]
else:
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
