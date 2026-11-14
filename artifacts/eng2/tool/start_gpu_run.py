"""Immutable START binding (copied from ENG1's tool; adapted from GN1 A8; ENG2 change: ENG2_AUDIT). Prior art: SHA-256
NIST (2001); GN1 L59/D3."""
import datetime
import json
import os
from evidence import SEAL_NAME, TOOL, guard, verify_seal, verify_start, sha, verify_arm

runs = guard(os.environ["RUNS"])
seal = verify_seal()
for arm in seal["arms"]:
    verify_arm(arm, seal)
runs.mkdir(parents=True, exist_ok=True)
record = dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              order_commit=seal["order_commit"], order_sha256=seal["order_sha256"],
              audit=os.environ.get("ENG2_AUDIT") == "1", run_root=str(runs),
              seal_sha256=sha(TOOL / SEAL_NAME))
if (runs / "START.json").exists():
    _, old = verify_start(runs)
    assert old["audit"] == record["audit"]
else:
    with (runs / "START.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
