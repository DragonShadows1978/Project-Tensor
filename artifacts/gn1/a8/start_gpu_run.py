"""Immutable START binding. Prior art: SHA-256 NIST (2001); GN1 L59/D3."""
import datetime
import json
import os
from evidence import TOOL, guard, verify_seal, verify_start, sha, verify_arm

runs = guard(os.environ["RUNS"])
seal = verify_seal()
for arm in seal["arms"]:
    verify_arm(arm, seal)
runs.mkdir(parents=True, exist_ok=True)
record = dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              order_commit=seal["order_commit"], order_sha256=seal["order_sha256"],
              audit=os.environ.get("GN1_AUDIT") == "1", control=os.environ.get("GN1_CONTROL") == "1", run_root=str(runs),
              seal_sha256=sha(TOOL / "staging_seal.json"))
if (runs / "START.json").exists():
    _, old = verify_start(runs)
    assert old["audit"] == record["audit"] and old["control"] == record["control"]
else:
    with (runs / "START.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2)+"\n")
print(json.dumps(record))
