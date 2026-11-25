"""BWD-F5 CPU drive stand-ins (never staged on a GPU line; no CUDA, no engine import). Two roles, one file:

  python3 -B -P scripts/bwdf5_drive_stub.py sanitizer --tool memcheck --log-file <log> <child argv...>
      behaves like `compute-sanitizer ... --log-file <log> <child argv>`: writes the log the behavior table asks for,
      then runs the child and exits with the child's status (a signal death becomes the same signal on this process).
  python3 -B -P scripts/bwdf5_drive_stub.py child <mode> <arm> <rid> <out_dir>
      behaves like one row's child: writes <rid>.row.json and prints ROW, or dies / hangs as the table asks.

The behavior table is JSON in BWDF5_STUB: {rid: {"log": ok|errors|empty|missing|noapi, "child": ok|exit255|segv|hang|
norow|exit0norow}}. Used only by scripts/bwdf5_cpu_drive.py to drive the runner's recording of every way a child can end.
Prior art: no prior art known to me (a test double).
"""
import json
import os
import signal
import subprocess
import sys
import time

LOGS = {"ok": "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n",
        "errors": "========= COMPUTE-SANITIZER\n========= Invalid __global__ read of size 4 bytes\n"
                  "========= ERROR SUMMARY: 2 errors\n",
        "empty": "",
        "noapi": "========= COMPUTE-SANITIZER\n========= Error: Target application terminated before first "
                 "instrumented API call\n"}


def table():
    return json.loads(os.environ.get("BWDF5_STUB", "{}"))


def sanitizer(argv):
    i = argv.index("--log-file")
    log, cmd = argv[i + 1], argv[i + 2:]
    rid = cmd[cmd.index("child") + 3]
    kind = table().get(rid, {}).get("log", "ok")
    if kind != "missing":
        open(log, "w").write(LOGS[kind])
    p = subprocess.run(cmd)
    if p.returncode < 0:
        os.kill(os.getpid(), -p.returncode)
    sys.exit(p.returncode)


def child(mode, arm, rid, out):
    kind = table().get(rid, {}).get("child", "ok")
    open(os.path.join(out, f"{rid}.pid"), "w").write(str(os.getpid()))
    if kind == "hang":
        time.sleep(600)
    if kind == "segv":
        os.kill(os.getpid(), signal.SIGSEGV)
    if kind == "exit255":
        sys.exit(255)
    if kind == "exit0norow":
        return
    row = {"id": rid, "module_sha256": "stub", "result": "RETURNED"}
    if kind == "norow":
        print("no row here")
        sys.exit(1)
    open(os.path.join(out, f"{rid}.row.json"), "w").write(json.dumps(row))
    print("ROW " + json.dumps(row))


if __name__ == "__main__":
    if sys.argv[1] == "sanitizer":
        sanitizer(sys.argv[2:])
    else:
        child(*sys.argv[2:6])
